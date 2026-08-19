# -*- coding: utf-8 -*-
"""Small 104 company API client shared by the random crawler."""

from __future__ import annotations

import random
import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
import requests


SEARCH_API_URL = "https://www.104.com.tw/jobs/search/api/jobs"
LEGACY_COMPANY_CONTENT_API_URL = "https://www.104.com.tw/company/ajax/content/{cust_no}"
COMPANY_CONTENT_API_URL = "https://www.104.com.tw/api/companies/{cust_no}/content"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/146.0.0.0 Safari/537.36"
)

COMPANY_SUFFIX_PATTERNS = (
    "股份有限公司台灣分公司",
    "股份有限公司臺灣分公司",
    "有限公司台灣分公司",
    "有限公司臺灣分公司",
    "股份有限公司",
    "有限公司",
    "台灣分公司",
    "臺灣分公司",
    "分公司",
    "股份",
)

TAIWAN_CITIES = (
    "台北市",
    "新北市",
    "桃園市",
    "台中市",
    "台南市",
    "高雄市",
    "基隆市",
    "新竹市",
    "嘉義市",
    "新竹縣",
    "苗栗縣",
    "彰化縣",
    "南投縣",
    "雲林縣",
    "嘉義縣",
    "屏東縣",
    "宜蘭縣",
    "花蓮縣",
    "台東縣",
    "澎湖縣",
    "金門縣",
    "連江縣",
)

SNACK_WELFARE_KEYWORDS = (
    "零食",
    "點心",
    "下午茶",
    "happyhour",
    "happy hour",
    "咖啡",
    "咖啡吧",
    "飲料",
    "茶水",
    "員工餐",
    "伙食",
    "餐費",
    "部門聚餐",
    "聚餐",
)


def random_sleep(min_seconds: float = 2.0, max_seconds: float = 4.0) -> None:
    delay = random.uniform(min_seconds, max_seconds)
    print(f"    waiting {delay:.1f}s before next 104 request...")
    time.sleep(delay)


def build_headers(referer: str) -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Referer": referer,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }


def get_json(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    timeout: int = 15,
) -> dict[str, Any] | None:
    random_sleep()
    try:
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except ValueError:
        print(f"    response is not valid JSON: {url}")
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        print(f"    HTTP error {status_code}: {url}")
    except requests.RequestException as exc:
        print(f"    request failed: {exc}")
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def join_values(values: Any) -> str:
    if isinstance(values, list):
        return "、".join(clean_text(value) for value in values if clean_text(value))
    return clean_text(values)


def normalize_company_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name or "")).strip().lower()
    text = text.replace("臺", "台")
    text = re.sub(r"\s+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[_()（）【】\[\]「」『』,，.。．·]", "", text)
    return text


def strip_parenthetical_text(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name or ""))
    text = re.sub(r"（[^）]*）|\([^)]*\)", "", text)
    return text.strip()


def remove_latin_if_chinese_remains(text: str) -> str:
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text

    without_latin = re.sub(r"[a-zA-Z]+", "", text)
    without_latin = re.sub(r"\s+", "", without_latin)
    if len(re.findall(r"[\u4e00-\u9fff]", without_latin)) >= 2:
        return without_latin
    return text


def choose_best_company_name_segment(text: str) -> str:
    segments = [segment.strip() for segment in re.split(r"[_＿]+", text) if segment.strip()]
    if not segments:
        return text
    return max(
        segments,
        key=lambda segment: (len(re.findall(r"[\u4e00-\u9fff]", segment)), len(segment)),
    )


def company_core_name(name: str) -> str:
    text = strip_parenthetical_text(str(name or ""))
    text = choose_best_company_name_segment(text)
    text = re.split(r"[-－–—/／|｜]", text, maxsplit=1)[0]
    text = remove_latin_if_chinese_remains(text)
    text = normalize_company_name(text)

    for suffix in COMPANY_SUFFIX_PATTERNS:
        normalized_suffix = normalize_company_name(suffix)
        if text.endswith(normalized_suffix):
            text = text[: -len(normalized_suffix)]
            break
    return text


def is_company_name_match(target_name: str, candidate_name: str) -> tuple[bool, str]:
    target = normalize_company_name(target_name)
    candidate = normalize_company_name(candidate_name)
    target_core = company_core_name(target_name)
    candidate_core = company_core_name(candidate_name)

    if candidate == target:
        return True, "exact"
    if target_core and candidate_core and candidate_core == target_core:
        return True, "core_exact"
    if target in candidate or candidate in target:
        return True, "partial"
    if target_core and candidate_core:
        if target_core in candidate_core or candidate_core in target_core:
            return True, "core_partial"
        if SequenceMatcher(None, target_core, candidate_core).ratio() >= 0.86:
            return True, "core_fuzzy"

    return False, "no_company_name_match"


def company_search_keywords(company_name: str) -> list[str]:
    display_name = strip_parenthetical_text(company_name)
    display_name = choose_best_company_name_segment(display_name)
    first_segment = re.split(r"[-－–—/／|｜]", display_name, maxsplit=1)[0]
    no_latin = remove_latin_if_chinese_remains(first_segment)

    keywords = [
        str(company_name).strip(),
        display_name,
        first_segment,
        no_latin,
        company_core_name(company_name),
    ]

    deduped: list[str] = []
    for keyword in keywords:
        if keyword and keyword not in deduped:
            deduped.append(keyword)
    return deduped


def extract_cust_no_from_job(job: dict[str, Any]) -> str | None:
    link = job.get("link")
    if isinstance(link, dict):
        company_url = link.get("cust")
        if company_url:
            match = re.search(r"/company/([^/?#'\" ]+)", str(company_url))
            if match:
                return match.group(1)

    for key in ("custUrl", "companyUrl"):
        company_url = job.get(key)
        if not company_url:
            continue
        match = re.search(r"/company/([^/?#'\" ]+)", str(company_url))
        if match:
            return match.group(1)

    cust_no = job.get("custNo")
    if cust_no:
        return str(cust_no).strip()
    return None


def parse_first_number(value: Any) -> float | None:
    text = clean_text(value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def parse_capital_ntd(value: Any) -> float | None:
    text = unicodedata.normalize("NFKC", clean_text(value))
    if not text or re.search(r"暫不提供|未提供|無|nan", text, flags=re.IGNORECASE):
        return None

    text = re.sub(r"[,，\s新台幣臺幣台幣元NTD$]", "", text)
    total = 0.0
    has_unit = False

    yi_match = re.search(r"(\d+(?:\.\d+)?)億", text)
    if yi_match:
        total += float(yi_match.group(1)) * 100_000_000
        has_unit = True

    wan_match = re.search(r"(?:億)?(\d+(?:\.\d+)?)萬", text)
    if wan_match:
        total += float(wan_match.group(1)) * 10_000
        has_unit = True

    if has_unit:
        return total
    return parse_first_number(text)


def extract_address_parts(address: Any, area_desc: Any = None) -> tuple[str | None, str | None]:
    text = unicodedata.normalize("NFKC", clean_text(address)).replace("臺", "台")
    area_text = unicodedata.normalize("NFKC", clean_text(area_desc)).replace("臺", "台")
    source = text or area_text
    if not source:
        return None, None

    city = None
    for city_name in TAIWAN_CITIES:
        if city_name in source:
            city = city_name
            break
    if not city:
        return None, None

    district = None
    city_index = source.find(city)
    if city_index >= 0:
        tail = source[city_index + len(city) :]
        district_match = re.match(r"([\u4e00-\u9fff]{1,6}(?:區|鄉|鎮|市))", tail)
        if district_match:
            district = district_match.group(1)

    if not district and area_text:
        area_tail = area_text.replace(city, "", 1)
        district_match = re.match(r"([\u4e00-\u9fff]{1,6}(?:區|鄉|鎮|市))", area_tail)
        if district_match:
            district = district_match.group(1)

    return city, district


def find_snack_welfare_keywords(*texts: Any) -> str:
    combined = "\n".join(clean_text(text).lower() for text in texts if clean_text(text))
    found = []
    for keyword in SNACK_WELFARE_KEYWORDS:
        if keyword.lower() in combined and keyword not in found:
            found.append(keyword)
    return "、".join(found)


def fetch_company_profile(session: requests.Session, cust_no: str) -> dict[str, Any]:
    url = COMPANY_CONTENT_API_URL.format(cust_no=cust_no)
    headers = build_headers(f"https://www.104.com.tw/company/{cust_no}")
    data = get_json(session, url, headers=headers)

    if not data:
        legacy_url = LEGACY_COMPANY_CONTENT_API_URL.format(cust_no=cust_no)
        data = get_json(session, legacy_url, headers=headers)

    if not data:
        return {}

    company_data = data.get("data", {})
    if not isinstance(company_data, dict):
        return {}
    return company_data


def company_content_to_record(company_data: dict[str, Any]) -> dict[str, Any]:
    profile = clean_text(company_data.get("profile"))
    welfare = clean_text(company_data.get("welfare"))
    welfare_tags = join_values(company_data.get("tagNames"))
    legal_welfare_tags = join_values(company_data.get("legalTagNames"))
    address = clean_text(company_data.get("address")) or None
    area_desc = clean_text(company_data.get("addrNoDesc")) or None
    city, district = extract_address_parts(address, area_desc)
    snack_keywords = find_snack_welfare_keywords(welfare, welfare_tags)

    return {
        "104_official_company_name": clean_text(company_data.get("custName")) or None,
        "公司簡介": profile,
        "福利制度": welfare,
        "104_welfare_tags": welfare_tags or None,
        "104_legal_welfare_tags": legal_welfare_tags or None,
        "104_has_snack_related_welfare": bool(snack_keywords),
        "104_welfare_snack_keywords": snack_keywords or None,
        "104_capital": clean_text(company_data.get("capital")) or None,
        "104_capital_ntd": parse_capital_ntd(company_data.get("capital")),
        "104_industry": clean_text(company_data.get("industryDesc")) or None,
        "104_industry_no": company_data.get("industryNo"),
        "104_indcat": clean_text(company_data.get("indcat")) or None,
        "104_employee_count": clean_text(company_data.get("empNo")) or None,
        "104_employee_count_number": parse_first_number(company_data.get("empNo")),
        "104_address": address,
        "104_address_city": city,
        "104_address_district": district,
        "104_addr_no_desc": area_desc,
        "104_website": clean_text(company_data.get("custLink")) or None,
        "104_phone": clean_text(company_data.get("phone")) or None,
        "104_hr_name": clean_text(company_data.get("hrName")) or None,
        "104_news": clean_text(company_data.get("news")) or None,
    }
