import random
import re
import time
import unicodedata
import urllib3
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from openpyxl.styles import Alignment

import company_104_client as company_104


# 避免商工 API SSL 憑證問題
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# 基本設定
# ============================================================

# 正式目標：最後保留至少 500 筆「104 有資料 + 商工查得到 + 有所營事業代碼」的企業
TARGET_VALID_N = 500

# 最多檢查多少間 104 候選公司。
# 如果跑完不足 500 筆，可以把這裡加大，例如 3500 或 5000。
MAX_CANDIDATES_TO_CHECK = 2500

# 最後輸出檔
FINAL_CSV_FILE = "outputs/random_104_gcis_filtered_profiles.csv"
FINAL_XLSX_FILE = "outputs/random_104_gcis_filtered_profiles.xlsx"

# 商工 API
GCIS_COMPANY_SEARCH_API = "https://data.gcis.nat.gov.tw/od/data/api/6BBA2268-1367-4B42-9CCA-BC17499EBE8C"
GCIS_COMPANY_BUSINESS_API = "https://data.gcis.nat.gov.tw/od/data/api/236EE382-4942-41A9-BD03-CA0709025E7C"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}

# 從 104 隨機抓活躍公司用的關鍵字
KEYWORDS = [
    "工程師",
    "行銷",
    "業務",
    "行政",
    "人資",
    "財務",
    "會計",
    "軟體",
    "資料",
    "專案",
    "產品",
    "客服",
    "營運",
    "設計",
    "採購",
    "物流",
    "餐飲",
    "零售",
    "金融",
    "顧問",
    "助理",
    "管理",
    "科技",
    "製造",
    "電商",
]

# 最後輸出欄位：對齊 customer_profiles 欄位，不新增、不修改
ORIGINAL_COLUMNS = [
    "企業名稱",
    "104_custNo",
    "104_matched_company_name",
    "104_profile_status",
    "104_official_company_name",
    "公司簡介",
    "福利制度",
    "104_welfare_tags",
    "104_legal_welfare_tags",
    "104_has_snack_related_welfare",
    "104_welfare_snack_keywords",
    "104_address",
    "104_address_city",
    "104_address_district",
    "104_addr_no_desc",
    "104_website",
    "local_employee_count",
    "local_capital_ntd",
    "industry",
    "農、林、漁、牧業",
    "礦業及土石採取業",
    "製造業",
    "電力及燃氣供應業",
    "營造業",
    "批發、零售及餐飲業",
    "運輸及倉儲業",
    "金融及保險業",
    "專業、科學及技術服務業",
    "文化、運動、娛樂及其他服務業",
]

LETTER_TO_INDUSTRY = {
    "A": "農、林、漁、牧業",
    "B": "礦業及土石採取業",
    "C": "製造業",
    "D": "電力及燃氣供應業",
    "E": "營造業",
    "F": "批發、零售及餐飲業",
    "G": "運輸及倉儲業",
    "H": "金融及保險業",
    "I": "專業、科學及技術服務業",
    "J": "文化、運動、娛樂及其他服務業",
}

INDUSTRY_COLUMNS = list(LETTER_TO_INDUSTRY.values())


# ============================================================
# 小工具
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value))
    text = text.replace("臺", "台")
    text = re.sub(r"\s+", "", text)
    return text


def company_history_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"（[^）]*）|\([^)]*\)", "", text)
    text = re.sub(r"[_＿]+", "", text)
    text = re.sub(r"[-－–—/／|｜].*$", "", text)
    text = re.sub(r"股份有限公司|有限公司|台灣分公司|臺灣分公司|分公司|股份", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def sleep_gcis() -> None:
    time.sleep(random.uniform(0.5, 1.0))


def get_json_list(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    label: str = "",
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """
    呼叫商工 API。
    注意：這裡 verify=False 是因為你的電腦跑商工 API 時會遇到 SSL 憑證驗證錯誤。
    """
    sleep_gcis()

    try:
        response = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=timeout,
            verify=False,
        )

        if response.status_code != 200:
            print(f"    {label} HTTP status={response.status_code}")
            return []

        if not response.text.strip():
            return []

        data = response.json()

    except requests.RequestException as exc:
        print(f"    {label} request failed: {exc}")
        return []
    except ValueError:
        print(f"    {label} response is not JSON")
        return []

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        for key in ["data", "records", "result", "results"]:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

        return [data]

    return []


def parse_number(value: Any) -> float | None:
    text = clean_text(value).replace(",", "")
    if not text:
        return None

    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None

    return float(match.group(0))


# ============================================================
# 104：抓候選公司
# ============================================================

def collect_104_candidates_once(
    session: requests.Session,
    seen_cust_no: set[str],
    excluded_company_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    從 104 職缺搜尋隨機抓一頁公司。
    """
    keyword = random.choice(KEYWORDS)
    page = random.randint(1, 30)

    print(f"[104 Search] keyword={keyword}, page={page}")

    params = {
        "keyword": keyword,
        "mode": "s",
        "order": "16",
        "page": str(page),
    }

    headers = company_104.build_headers("https://www.104.com.tw/")
    data = company_104.get_json(
        session,
        company_104.SEARCH_API_URL,
        headers=headers,
        params=params,
    )

    if not data:
        return []

    payload = data.get("data", [])

    if isinstance(payload, dict):
        jobs = payload.get("list", [])
    elif isinstance(payload, list):
        jobs = payload
    else:
        jobs = []

    excluded_company_keys = excluded_company_keys or set()
    candidates = []

    for job in jobs:
        if not isinstance(job, dict):
            continue

        company_name = clean_text(job.get("custName"))
        cust_no = company_104.extract_cust_no_from_job(job)

        if not company_name or not cust_no:
            continue

        if cust_no in seen_cust_no:
            continue

        if company_history_key(company_name) in excluded_company_keys:
            continue

        seen_cust_no.add(cust_no)

        candidates.append(
            {
                "企業名稱": company_name,
                "104_custNo": cust_no,
                "104_matched_company_name": company_name,
            }
        )

    return candidates


# ============================================================
# 商工：查公司與所營事業資料
# ============================================================

def company_query_keywords(*names: Any) -> list[str]:
    """
    從 104 公司名稱產生商工查詢關鍵字。
    """
    keywords = []

    for name in names:
        text = clean_text(name)
        if not text:
            continue

        try:
            generated = company_104.company_search_keywords(text)
        except Exception:
            generated = [text]

        for keyword in generated:
            keyword = clean_text(keyword)
            if not keyword:
                continue

            # 太短容易誤配
            if len(normalize_text(keyword)) < 3:
                continue

            if keyword not in keywords:
                keywords.append(keyword)

    return keywords


def choose_best_gcis_company(
    target_names: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    從商工查詢結果中，挑出最像 104 公司名稱的公司。
    """
    valid_matches = []

    for row in rows:
        gcis_name = clean_text(row.get("Company_Name"))
        business_no = clean_text(row.get("Business_Accounting_NO"))
        status = clean_text(row.get("Company_Status"))
        status_desc = clean_text(row.get("Company_Status_Desc"))

        if not gcis_name or not business_no:
            continue

        if status and status != "01":
            continue

        if status_desc and "核准設立" not in status_desc:
            continue

        best_type = None

        for target in target_names:
            target = clean_text(target)
            if not target:
                continue

            matched, match_type = company_104.is_company_name_match(target, gcis_name)
            if matched:
                best_type = match_type
                break

        if best_type:
            row["_match_type"] = best_type
            valid_matches.append(row)

    if not valid_matches:
        return None

    priority = {
        "exact": 0,
        "core_exact": 1,
        "partial": 2,
        "core_partial": 3,
        "core_fuzzy": 4,
    }

    valid_matches.sort(
        key=lambda row: priority.get(clean_text(row.get("_match_type")), 99)
    )

    return valid_matches[0]


def search_gcis_company(
    session: requests.Session,
    target_names: list[str],
) -> dict[str, Any] | None:
    """
    用公司名稱查商工。
    只使用 Company_Status eq 01 的條件，避免商工 API 回空白。
    """
    keywords = company_query_keywords(*target_names)

    for keyword in keywords:
        filters = [
            f"Company_Name eq {keyword} and Company_Status eq 01",
            f"Company_Name like {keyword} and Company_Status eq 01",
        ]

        for filter_text in filters:
            params = {
                "$format": "json",
                "$filter": filter_text,
                "$skip": "0",
                "$top": "20",
            }

            rows = get_json_list(
                session,
                GCIS_COMPANY_SEARCH_API,
                params,
                label="GCIS company search",
            )

            if not rows:
                continue

            best = choose_best_gcis_company(target_names, rows)

            if best:
                return best

    return None


def fetch_gcis_business_rows(
    session: requests.Session,
    business_no: str,
) -> list[dict[str, Any]]:
    """
    用統編查商工所營事業資料。
    """
    params = {
        "$format": "json",
        "$filter": f"Business_Accounting_NO eq {business_no}",
        "$skip": "0",
        "$top": "10",
    }

    return get_json_list(
        session,
        GCIS_COMPANY_BUSINESS_API,
        params,
        label="GCIS business",
    )


def extract_business_item_codes(rows: list[dict[str, Any]]) -> list[str]:
    """
    從 Cmp_Business 抓 Business_Item。
    """
    codes = []

    def add_code(value: Any) -> None:
        text = clean_text(value).upper()
        if not text:
            return

        found = re.findall(r"[A-Z]{1,2}\d{5,6}", text)
        for code in found:
            if code not in codes:
                codes.append(code)

    for row in rows:
        cmp_business = row.get("Cmp_Business")

        if isinstance(cmp_business, list):
            for item in cmp_business:
                if isinstance(item, dict):
                    add_code(item.get("Business_Item"))
                    add_code(item.get("Business_Item_Desc"))
                else:
                    add_code(item)

        elif isinstance(cmp_business, dict):
            add_code(cmp_business.get("Business_Item"))
            add_code(cmp_business.get("Business_Item_Desc"))

        # 保險：有些資料可能直接放在 row 裡
        add_code(row.get("Business_Item"))

    return codes


def business_codes_to_industry(codes: list[str]) -> tuple[str, dict[str, int]]:
    """
    用營業項目代碼第一個英文字母 A-J 轉產業分類。
    Z 或其他不是 A-J 的代碼不納入。
    """
    letters = []

    for code in codes:
        code = clean_text(code).upper()
        if not code:
            continue

        first_letter = code[0]

        if first_letter in LETTER_TO_INDUSTRY and first_letter not in letters:
            letters.append(first_letter)

    ordered_letters = [
        letter for letter in LETTER_TO_INDUSTRY.keys()
        if letter in letters
    ]

    industries = [
        LETTER_TO_INDUSTRY[letter]
        for letter in ordered_letters
    ]

    industry_text = ",".join(industries)

    one_hot = {
        industry_name: 1 if industry_name in industries else 0
        for industry_name in INDUSTRY_COLUMNS
    }

    return industry_text, one_hot


# ============================================================
# 整理輸出
# ============================================================

def build_final_record(
    candidate: dict[str, Any],
    company_data: dict[str, Any],
    gcis_company: dict[str, Any],
    business_codes: list[str],
) -> dict[str, Any]:
    """
    組成最後 28 欄資料。
    """
    content_record = company_104.company_content_to_record(company_data)

    profile_status = (
        "ok"
        if clean_text(content_record.get("公司簡介"))
        else "empty_profile"
    )

    industry_text, one_hot = business_codes_to_industry(business_codes)

    capital_from_gcis = parse_number(gcis_company.get("Capital_Stock_Amount"))
    capital_from_104 = content_record.get("104_capital_ntd")

    local_capital_ntd = capital_from_gcis
    if local_capital_ntd is None:
        local_capital_ntd = capital_from_104

    record = {
        "企業名稱": clean_text(gcis_company.get("Company_Name")) or clean_text(candidate.get("企業名稱")),
        "104_custNo": clean_text(candidate.get("104_custNo")),
        "104_matched_company_name": clean_text(candidate.get("104_matched_company_name")),
        "104_profile_status": profile_status,
        "104_official_company_name": content_record.get("104_official_company_name"),
        "公司簡介": content_record.get("公司簡介"),
        "福利制度": content_record.get("福利制度"),
        "104_welfare_tags": content_record.get("104_welfare_tags"),
        "104_legal_welfare_tags": content_record.get("104_legal_welfare_tags"),
        "104_has_snack_related_welfare": content_record.get("104_has_snack_related_welfare"),
        "104_welfare_snack_keywords": content_record.get("104_welfare_snack_keywords"),
        "104_address": content_record.get("104_address"),
        "104_address_city": content_record.get("104_address_city"),
        "104_address_district": content_record.get("104_address_district"),
        "104_addr_no_desc": content_record.get("104_addr_no_desc"),
        "104_website": content_record.get("104_website"),
        "local_employee_count": content_record.get("104_employee_count_number"),
        "local_capital_ntd": local_capital_ntd,
        "industry": industry_text,
    }

    for industry_name in INDUSTRY_COLUMNS:
        record[industry_name] = one_hot[industry_name]

    final_record = {}
    for column in ORIGINAL_COLUMNS:
        final_record[column] = record.get(column, pd.NA)

    return final_record


def save_final_files(
    records: list[dict[str, Any]],
    output_csv: str | Path = FINAL_CSV_FILE,
    output_xlsx: str | Path = FINAL_XLSX_FILE,
) -> pd.DataFrame:
    """
    儲存最後結果。
    欄位固定為 ORIGINAL_COLUMNS。
    """
    df = pd.DataFrame(records)

    for column in ORIGINAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[ORIGINAL_COLUMNS].copy()

    output_csv = Path(output_csv)
    output_xlsx = Path(output_xlsx)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="profiles")

        worksheet = writer.sheets["profiles"]
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter
            header = clean_text(column_cells[0].value)

            if header in ["公司簡介", "福利制度"]:
                worksheet.column_dimensions[column_letter].width = 45
            elif header == "104_address":
                worksheet.column_dimensions[column_letter].width = 32
            elif header == "industry":
                worksheet.column_dimensions[column_letter].width = 36
            else:
                worksheet.column_dimensions[column_letter].width = 18

            for cell in column_cells:
                cell.alignment = Alignment(vertical="top", wrap_text=False)

    return df


# ============================================================
# Pipeline
# ============================================================

def generate_random_profiles(
    target_valid_n: int = TARGET_VALID_N,
    max_candidates_to_check: int = MAX_CANDIDATES_TO_CHECK,
    output_csv: str | Path = FINAL_CSV_FILE,
    output_xlsx: str | Path = FINAL_XLSX_FILE,
    excluded_company_keys: set[str] | None = None,
    excluded_cust_nos: set[str] | None = None,
    save_every: int = 20,
) -> pd.DataFrame:
    session_104 = requests.Session()
    session_gcis = requests.Session()

    excluded_company_keys = excluded_company_keys or set()
    seen_cust_no = set(excluded_cust_nos or set())
    valid_records = []

    checked_candidates = 0
    round_no = 0

    while (
        len(valid_records) < target_valid_n
        and checked_candidates < max_candidates_to_check
    ):
        round_no += 1

        candidates = collect_104_candidates_once(
            session_104,
            seen_cust_no,
            excluded_company_keys=excluded_company_keys,
        )

        if not candidates:
            print(f"[Round {round_no}] no 104 candidates")
            continue

        for candidate in candidates:
            if len(valid_records) >= target_valid_n:
                break

            if checked_candidates >= max_candidates_to_check:
                break

            checked_candidates += 1

            company_name = clean_text(candidate.get("企業名稱"))
            cust_no = clean_text(candidate.get("104_custNo"))

            print("=" * 80)
            print(
                f"[Check {checked_candidates}] "
                f"valid={len(valid_records)}/{target_valid_n} | "
                f"104 company={company_name}"
            )

            if company_history_key(company_name) in excluded_company_keys:
                print("    skip: 近期已開發公司")
                continue

            # 1. 抓 104 profile
            company_data = company_104.fetch_company_profile(session_104, cust_no)
            content_record = company_104.company_content_to_record(company_data)

            official_104_name = clean_text(content_record.get("104_official_company_name"))
            matched_104_name = clean_text(candidate.get("104_matched_company_name"))

            target_names = [
                official_104_name,
                matched_104_name,
                company_name,
            ]

            # 2. 商工查公司
            gcis_company = search_gcis_company(session_gcis, target_names)

            if not gcis_company:
                print("    skip: 商工查不到可確認的公司")
                continue

            gcis_name = clean_text(gcis_company.get("Company_Name"))
            business_no = clean_text(gcis_company.get("Business_Accounting_NO"))

            if company_history_key(gcis_name) in excluded_company_keys:
                print(f"    skip: 商工匹配為近期已開發公司 {gcis_name}")
                continue

            print(f"    GCIS matched: {gcis_name} ({business_no})")

            # 3. 查所營事業資料
            business_rows = fetch_gcis_business_rows(session_gcis, business_no)
            business_codes = extract_business_item_codes(business_rows)

            if not business_codes:
                print("    skip: 商工沒有抓到所營事業代碼")
                continue

            industry_text, _ = business_codes_to_industry(business_codes)

            if not industry_text:
                print("    skip: 所營事業代碼沒有 A-J 可分類項目")
                continue

            # 4. 組最後資料
            final_record = build_final_record(
                candidate=candidate,
                company_data=company_data,
                gcis_company=gcis_company,
                business_codes=business_codes,
            )

            valid_records.append(final_record)
            excluded_company_keys.add(company_history_key(final_record.get("企業名稱")))

            print(f"    keep: industry={final_record['industry']}")

            # 每 20 筆先存一次，避免中途斷掉
            if save_every and len(valid_records) % save_every == 0:
                save_final_files(valid_records, output_csv=output_csv, output_xlsx=output_xlsx)
                print(f"    progress saved: {len(valid_records)} records")

    final_df = save_final_files(valid_records, output_csv=output_csv, output_xlsx=output_xlsx)

    print("=" * 80)
    print(f"Done. Checked candidates: {checked_candidates}")
    print(f"Done. Valid records: {len(final_df)}")
    print(f"Final CSV: {output_csv}")
    print(f"Final Excel: {output_xlsx}")

    if len(final_df) < target_valid_n:
        print(
            f"WARNING: 目前只有 {len(final_df)} 筆，低於目標 {target_valid_n} 筆。"
            f"可以把 max_candidates_to_check 從 {max_candidates_to_check} 調大後再跑一次。"
        )

    return final_df


# ============================================================
# 主程式
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly crawl active 104 companies and enrich with GCIS data.")
    parser.add_argument("--target-valid-n", type=int, default=TARGET_VALID_N, help="Target valid company count.")
    parser.add_argument(
        "--max-candidates-to-check",
        type=int,
        default=MAX_CANDIDATES_TO_CHECK,
        help="Maximum 104 candidate companies to inspect.",
    )
    parser.add_argument("--output-csv", default=FINAL_CSV_FILE, help="Output CSV path.")
    parser.add_argument("--output-xlsx", default=FINAL_XLSX_FILE, help="Output Excel path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_random_profiles(
        target_valid_n=args.target_valid_n,
        max_candidates_to_check=args.max_candidates_to_check,
        output_csv=args.output_csv,
        output_xlsx=args.output_xlsx,
    )


if __name__ == "__main__":
    main()
