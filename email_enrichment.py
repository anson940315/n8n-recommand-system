# -*- coding: utf-8 -*-
"""Best-effort public email enrichment for scored leads.

The crawler only visits official company websites discovered from 104 company
profiles. It does not guess personal addresses; if no public email is found,
the row is kept for manual follow-up.
"""

from __future__ import annotations

import argparse
import html
import re
import time
import urllib3
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests

import company_104_client as company_104


DEFAULT_INPUT_FILE = Path("outputs/n8n_top100_leads.csv")
DEFAULT_OUTPUT_FILE = Path("outputs/n8n_top100_leads_enriched.csv")
EMAIL_COLUMNS = ("Email", "email", "contact_email", "電子郵件")
EMAIL_REGEX = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.+-])", re.I)
HREF_REGEX = re.compile(r"<a\b[^>]*?\bhref=[\"']([^\"']+)[\"'][^>]*>", re.I)

CONTACT_LINK_KEYWORDS = (
    "contact",
    "contacts",
    "contact-us",
    "contactus",
    "about",
    "about-us",
    "support",
    "service",
    "聯絡",
    "聯繫",
    "客服",
    "關於",
    "公司",
)
COMMON_CONTACT_PATHS = (
    "/contact",
    "/contact-us",
    "/contacts",
    "/about",
    "/about-us",
    "/support",
    "/service",
    "/聯絡我們",
    "/關於我們",
)
SKIP_DOMAINS = (
    "104.com.tw",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
    "line.me",
)
BAD_LOCAL_PARTS = (
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "example",
    "test",
)
BAD_LOCAL_PREFIXES = (
    "%",
    "u00",
    "x00",
)
BAD_TLDS = {"png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js", "pdf"}
GOOD_LOCAL_KEYWORDS = (
    "info",
    "contact",
    "service",
    "sales",
    "support",
    "business",
    "marketing",
    "hr",
    "admin",
)
STRONG_CONTACT_EMAIL_CONTEXTS = (
    "公司聯絡信箱",
    "公司信箱",
    "聯絡信箱",
    "聯絡電子信箱",
    "電子信箱",
    "客服信箱",
    "服務信箱",
    "業務信箱",
    "合作信箱",
    "採購信箱",
    "公關信箱",
    "媒體信箱",
    "contact email",
    "contact e-mail",
    "business email",
    "service email",
)
WEAK_CONTACT_EMAIL_CONTEXTS = (
    "聯絡我們",
    "聯絡方式",
    "客服",
    "服務",
    "業務",
    "合作",
    "採購",
    "公關",
    "媒體",
    "信箱",
    "email",
    "e-mail",
    "mail",
)
NEGATIVE_EMAIL_CONTEXTS = (
    "訂閱",
    "退訂",
    "電子報",
    "隱私",
    "履歷",
    "求職",
    "招募",
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def read_leads(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 leads 檔案: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def existing_email(row: pd.Series, *, gmail_only: bool = False) -> str:
    for column in EMAIL_COLUMNS:
        value = clean_text(row.get(column))
        if not value:
            continue
        parsed = extract_emails(value, gmail_only=gmail_only)
        if parsed:
            return parsed[0]
    return ""


def normalize_url(value: Any) -> str:
    raw_text = clean_text(value)
    if not raw_text:
        return ""

    url_match = re.search(r"https?://[^\s,，;；]+", raw_text, flags=re.I)
    domain_match = re.search(
        r"(?:www\.)?[A-Za-z0-9][A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s,，;；]*)?",
        raw_text,
    )
    if url_match:
        text = url_match.group(0)
    elif domain_match:
        text = domain_match.group(0)
    else:
        return ""

    if not re.match(r"^https?://", text, flags=re.I):
        text = f"https://{text}"

    try:
        parsed = urlparse(text)
    except ValueError:
        return ""

    if not parsed.netloc:
        return ""

    host = parsed.netloc.lower()
    if "." not in host:
        return ""

    if any(domain in host for domain in SKIP_DOMAINS):
        return ""

    return text


def fetch_missing_website(session_104: requests.Session, row: pd.Series) -> str:
    current = normalize_url(row.get("104_website"))
    if current:
        return current

    cust_no = clean_text(row.get("104_custNo"))
    if not cust_no:
        return ""

    company_data = company_104.fetch_company_profile(session_104, cust_no)
    record = company_104.company_content_to_record(company_data)
    return normalize_url(record.get("104_website"))


def request_text(session: requests.Session, url: str, timeout: int = 10) -> str:
    headers = {
        "User-Agent": company_104.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        response = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        response = session.get(url, headers=headers, timeout=timeout, allow_redirects=True, verify=False)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "text/plain" not in content_type and "application/xhtml" not in content_type:
        return ""
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def same_domain(url: str, base_host: str) -> bool:
    host = urlparse(url).netloc.lower()
    if not host:
        return False
    return host == base_host or host.endswith("." + base_host)


def contact_urls(home_url: str, html_text: str, max_pages: int) -> list[str]:
    parsed = urlparse(home_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    base_host = parsed.netloc.lower().removeprefix("www.")

    urls = [home_url]
    for path in COMMON_CONTACT_PATHS:
        urls.append(urljoin(base, path))

    for raw_href in HREF_REGEX.findall(html_text):
        href = html.unescape(raw_href)
        if href.lower().startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(home_url, href)
        normalized_host = urlparse(absolute).netloc.lower().removeprefix("www.")
        if normalized_host != base_host:
            continue
        lower = absolute.lower()
        if any(keyword.lower() in lower for keyword in CONTACT_LINK_KEYWORDS):
            urls.append(absolute)

    deduped: list[str] = []
    for url in urls:
        if url not in deduped:
            deduped.append(url)
        if len(deduped) >= max_pages:
            break
    return deduped


def is_valid_email(email: str) -> bool:
    email = email.strip().strip(".,;:()[]{}<>\"'")
    if not email or "@" not in email:
        return False

    local, domain = email.lower().split("@", 1)
    if not local or not domain:
        return False
    if any(local.startswith(prefix) for prefix in BAD_LOCAL_PREFIXES):
        return False
    if re.search(r"%[0-9a-f]{2}", local, flags=re.I):
        return False
    if re.search(r"u00[0-9a-f]{2}", local, flags=re.I):
        return False
    if any(part in local for part in BAD_LOCAL_PARTS):
        return False
    if domain.split(".")[-1] in BAD_TLDS:
        return False
    if domain.endswith((".example", ".test")):
        return False
    return True


def decode_email_text(text: str) -> str:
    decoded = html.unescape(text or "")
    decoded = unquote(decoded)
    replacements = {
        r"\\u003c": "<",
        r"\\u003e": ">",
        r"\\u0026": "&",
        r"\\u0040": "@",
        "u003c": "<",
        "u003e": ">",
        "u0026": "&",
        "u0040": "@",
    }
    for needle, replacement in replacements.items():
        decoded = decoded.replace(needle, replacement)
    return decoded


def clean_email_candidate(email: str) -> str:
    cleaned = decode_email_text(email).strip().strip(".,;:()[]{}<>\"'")
    return cleaned.lower()


def extract_emails(text: str, *, gmail_only: bool = False) -> list[str]:
    emails: list[str] = []
    decoded_text = decode_email_text(text)
    for match in EMAIL_REGEX.findall(decoded_text):
        email = clean_email_candidate(match)
        if gmail_only and not email.endswith("@gmail.com"):
            continue
        if is_valid_email(email) and email not in emails:
            emails.append(email)
    return emails


def email_context_score(text: str, email_start: int, context_start: int = 0) -> int:
    decoded = decode_email_text(text)
    start = max(context_start, email_start - 140, 0)
    before = decoded[start:email_start].lower()
    context = re.sub(r"<[^>]+>", " ", before)
    context = re.sub(r"\s+", " ", context)

    score = 0
    if any(keyword.lower() in context for keyword in STRONG_CONTACT_EMAIL_CONTEXTS):
        score += 100
    if any(keyword.lower() in context for keyword in WEAK_CONTACT_EMAIL_CONTEXTS):
        score += 35
    if "mailto:" in before[-40:]:
        score += 15
    if any(keyword.lower() in context for keyword in NEGATIVE_EMAIL_CONTEXTS):
        score -= 60
    return score


def email_context_scores(text: str, *, gmail_only: bool = False) -> dict[str, int]:
    scores: dict[str, int] = {}
    decoded_text = decode_email_text(text)
    previous_email_end = 0
    for match in EMAIL_REGEX.finditer(decoded_text):
        email = clean_email_candidate(match.group(1))
        if gmail_only and not email.endswith("@gmail.com"):
            previous_email_end = match.end(1)
            continue
        if not is_valid_email(email):
            previous_email_end = match.end(1)
            continue

        score = email_context_score(decoded_text, match.start(1), previous_email_end)
        if score > scores.get(email, -999):
            scores[email] = score
        previous_email_end = match.end(1)
    return scores


def email_score(
    email: str,
    site_host: str,
    context_scores: dict[str, int] | None = None,
) -> int:
    local, domain = email.lower().split("@", 1)
    score = (context_scores or {}).get(email, 0)
    if any(keyword in local for keyword in GOOD_LOCAL_KEYWORDS):
        score += 30
    if domain in site_host or site_host.endswith(domain) or domain.endswith(site_host):
        score += 20
    if local in {"info", "contact", "service", "sales"}:
        score += 10
    return score


def choose_best_email(
    emails: list[str],
    website: str,
    context_scores: dict[str, int] | None = None,
) -> str:
    if not emails:
        return ""
    site_host = urlparse(website).netloc.lower().removeprefix("www.")
    indexed_emails = list(enumerate(emails))
    _, best_email = max(
        indexed_emails,
        key=lambda item: (
            email_score(item[1], site_host, context_scores),
            -item[0],
        ),
    )
    return best_email


def crawl_website_for_email(
    session: requests.Session,
    website: str,
    *,
    max_pages: int,
    request_delay: float,
    gmail_only: bool = False,
) -> tuple[str, list[str], str]:
    home_url = normalize_url(website)
    if not home_url:
        return "", [], "missing_website"

    try:
        home_html = request_text(session, home_url)
    except requests.RequestException:
        return "", [], "homepage_request_failed"

    urls = contact_urls(home_url, home_html, max_pages=max_pages)
    found: list[str] = []
    source_by_email: dict[str, str] = {}
    context_scores: dict[str, int] = {}

    for position, url in enumerate(urls):
        try:
            text = home_html if position == 0 else request_text(session, url)
        except requests.RequestException:
            continue

        page_emails = extract_emails(text, gmail_only=gmail_only)
        page_context_scores = email_context_scores(text, gmail_only=gmail_only)
        for email in page_emails:
            if email not in found:
                found.append(email)
            if email not in source_by_email:
                source_by_email[email] = url
            if page_context_scores.get(email, 0) > context_scores.get(email, -999):
                context_scores[email] = page_context_scores[email]

        if request_delay > 0 and position < len(urls) - 1:
            time.sleep(request_delay)

    best = choose_best_email(found, home_url, context_scores=context_scores)
    if best:
        return best, found, source_by_email.get(best, home_url)
    return "", [], "email_not_found"


def enrich_lead_emails(
    leads: pd.DataFrame,
    *,
    fetch_missing_websites: bool = True,
    max_pages_per_site: int = 4,
    request_delay: float = 0.5,
    gmail_only: bool = False,
) -> pd.DataFrame:
    enriched = leads.copy()
    for column in ["Email", "104_website", "email_source_url", "email_status", "email_candidates"]:
        if column not in enriched.columns:
            enriched[column] = ""

    session_104 = requests.Session()
    website_session = requests.Session()

    total = len(enriched)
    for position, index in enumerate(enriched.index, start=1):
        row = enriched.loc[index]
        company_name = clean_text(row.get("企業名稱"))
        current_email = existing_email(row, gmail_only=gmail_only)
        if current_email:
            enriched.at[index, "Email"] = current_email
            enriched.at[index, "email_status"] = "existing_email"
            print(f"[{position}/{total}] {company_name}: existing email")
            continue

        website = normalize_url(row.get("104_website"))
        if not website and fetch_missing_websites:
            print(f"[{position}/{total}] {company_name}: fetching 104 website")
            website = fetch_missing_website(session_104, row)
            enriched.at[index, "104_website"] = website
        else:
            print(f"[{position}/{total}] {company_name}: crawling website")

        if not website:
            enriched.at[index, "email_status"] = "missing_website"
            continue

        email, candidates, source = crawl_website_for_email(
            website_session,
            website,
            max_pages=max_pages_per_site,
            request_delay=request_delay,
            gmail_only=gmail_only,
        )
        enriched.at[index, "Email"] = email
        enriched.at[index, "email_candidates"] = " | ".join(candidates)
        enriched.at[index, "email_source_url"] = source if email else ""
        enriched.at[index, "email_status"] = "found" if email else source

        print(
            f"    website={website} status={enriched.at[index, 'email_status']} "
            f"email={email or '-'}"
        )

        if request_delay > 0:
            time.sleep(request_delay)

    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich top leads with public website emails.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--limit", type=int, default=0, help="Process only first N rows; 0 means all.")
    parser.add_argument("--max-pages-per-site", type=int, default=4)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument(
        "--gmail-only",
        action="store_true",
        help="Only keep @gmail.com addresses. This is conservative and may miss company-domain emails.",
    )
    parser.add_argument(
        "--no-fetch-missing-websites",
        action="store_true",
        help="Do not call 104 again when 104_website is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    leads = read_leads(input_file)
    if args.limit > 0:
        leads = leads.head(args.limit).copy()

    enriched = enrich_lead_emails(
        leads,
        fetch_missing_websites=not args.no_fetch_missing_websites,
        max_pages_per_site=args.max_pages_per_site,
        request_delay=args.request_delay,
        gmail_only=args.gmail_only,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_file, index=False, encoding="utf-8-sig")
    enriched.to_excel(output_file.with_suffix(".xlsx"), index=False)

    found = int(enriched["Email"].map(clean_text).astype(bool).sum())
    print(f"Done. Found {found}/{len(enriched)} emails.")
    print(f"Wrote: {output_file}")
    print(f"Wrote: {output_file.with_suffix('.xlsx')}")


if __name__ == "__main__":
    main()
