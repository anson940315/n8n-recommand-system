# -*- coding: utf-8 -*-
"""Rank random 104 companies by similarity to existing customers.

This script is intentionally kept as a production-style scoring pipeline, not a
Colab notebook export. It reads:

* private/customer_profiles.xlsx
* private/random_104_gcis_filtered_profiles.xlsx

and writes ranking outputs under outputs/.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CUSTOMER_FILE = "private/customer_profiles.xlsx"
DEFAULT_POTENTIAL_FILE = "private/random_104_gcis_filtered_profiles.xlsx"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_HISTORY_FILE = "outputs/development_history.csv"
DEFAULT_TOP_LEADS_FILE = "outputs/n8n_top100_leads.csv"
DEFAULT_CRAWL_TARGET = 500
DEFAULT_TOP_N = 100
DEFAULT_RECENT_DAYS = 7
DEFAULT_MIN_CRAWL_SUCCESS_RATIO = 0.5

COMPANY_NAME_COL = "企業名稱"
CUST_NO_COL = "104_custNo"

PROFILE_COL = "公司簡介"
WELFARE_TEXT_COLS = [
    "福利制度",
    "104_welfare_tags",
    "104_legal_welfare_tags",
    "104_welfare_snack_keywords",
]
INDUSTRY_TEXT_COL = "industry"
LOCATION_COLS = ["104_address_city", "104_address_district"]
NUMERIC_COLS = ["local_employee_count", "local_capital_ntd"]
SNACK_SIGNAL_COL = "104_has_snack_related_welfare"

INDUSTRY_COLS = [
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

# Welfare is weighted highest because the product is an employee snack service.
COMPONENT_WEIGHTS = {
    "profile": 0.18,
    "welfare": 0.55,
    "industry": 0.10,
    "size": 0.07,
    "location": 0.04,
    "snack_signal": 0.06,
}

AGGREGATION_WEIGHTS = {
    "max": 0.45,
    "top3_mean": 0.35,
    "top10_mean": 0.20,
}

TEXT_NGRAM_RANGE = (2, 4)
TOP_MATCH_N = 5
HISTORY_COLUMNS = [
    "developed_at",
    "company_name",
    "company_key",
    "104_custNo",
    "rank",
    "similarity_score",
    "source",
    "status",
    "outcome",
    "outcome_at",
    "run_id",
]
PERMANENT_SUCCESS_KEYWORDS = (
    "won",
    "signed",
    "closed_won",
    "converted",
    "converted_customer",
    "customer",
    "success",
    "successful",
    "deal",
    "成交",
    "簽約",
    "签约",
    "成為客戶",
    "成为客户",
    "正式客戶",
    "正式客户",
    "成功開發",
    "成功开发",
    "開發成功",
    "开发成功",
    "已成交",
    "已簽約",
    "已签约",
    "已成為客戶",
    "已成为客户",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("臺", "台")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_company_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"（[^）]*）|\([^)]*\)", "", text)
    text = re.sub(r"[_＿]+", "", text)
    text = re.sub(r"[-－–—/／|｜].*$", "", text)
    text = re.sub(r"股份有限公司|有限公司|台灣分公司|臺灣分公司|分公司|股份", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def truthy(value: Any) -> bool:
    text = clean_text(value).lower()
    return text in {"true", "1", "yes", "y", "是", "有", "t"}


def is_permanent_success_status(*values: Any) -> bool:
    text = " ".join(clean_text(value).lower() for value in values if clean_text(value))
    return any(keyword.lower() in text for keyword in PERMANENT_SUCCESS_KEYWORDS)


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def run_id_from_now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def read_excel_first_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {path}")
    return pd.read_excel(path, sheet_name=0)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_excel_first_sheet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def load_development_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.read_csv(path, encoding="utf-8-sig")
    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA
    return history[HISTORY_COLUMNS].copy()


def recent_development_filters(
    history_file: Path,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> tuple[set[str], set[str], pd.DataFrame]:
    history = load_development_history(history_file)
    if history.empty:
        return set(), set(), history

    developed_at = pd.to_datetime(history["developed_at"], errors="coerce", utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=recent_days)
    permanent_success = history.apply(
        lambda row: is_permanent_success_status(
            row.get("status"),
            row.get("outcome"),
            row.get("source"),
        ),
        axis=1,
    )
    recent = history[(developed_at >= cutoff) | permanent_success].copy()

    company_keys = {
        clean_text(key)
        for key in recent.get("company_key", pd.Series(dtype=str))
        if clean_text(key)
    }
    for name_column in ["company_name", COMPANY_NAME_COL, "CompanyName"]:
        if name_column in recent.columns:
            company_keys.update(
                normalize_company_key(name)
                for name in recent[name_column]
                if normalize_company_key(name)
            )

    cust_nos = {
        clean_text(cust_no)
        for cust_no in recent.get(CUST_NO_COL, pd.Series(dtype=str))
        if clean_text(cust_no)
    }

    return company_keys, cust_nos, history


def latest_history_lookup(history: pd.DataFrame) -> dict[str, str]:
    latest: dict[str, tuple[pd.Timestamp, str]] = {}
    if history.empty:
        return {}

    developed_at = pd.to_datetime(history["developed_at"], errors="coerce", utc=True)
    for position, (_, row) in enumerate(history.iterrows()):
        timestamp = developed_at.iloc[position]
        if pd.isna(timestamp):
            continue

        display_value = clean_text(row.get("developed_at"))
        keys = [
            f"cust:{clean_text(row.get(CUST_NO_COL))}",
            f"company:{clean_text(row.get('company_key'))}",
            f"company:{normalize_company_key(row.get('company_name'))}",
        ]
        for key in keys:
            if key in {"cust:", "company:"}:
                continue
            existing = latest.get(key)
            if existing is None or timestamp > existing[0]:
                latest[key] = (timestamp, display_value)

    return {key: value for key, (_, value) in latest.items()}


def add_development_history_columns(
    ranking: pd.DataFrame,
    history_file: Path,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> pd.DataFrame:
    company_keys, cust_nos, history = recent_development_filters(history_file, recent_days)
    latest_lookup = latest_history_lookup(history)

    enriched = ranking.copy()
    last_developed_values: list[str | None] = []
    recently_developed_values: list[bool] = []

    for _, row in enriched.iterrows():
        company_key = normalize_company_key(row.get(COMPANY_NAME_COL))
        cust_no = clean_text(row.get(CUST_NO_COL))
        last_developed = (
            latest_lookup.get(f"cust:{cust_no}")
            or latest_lookup.get(f"company:{company_key}")
        )
        last_developed_values.append(last_developed)
        recently_developed_values.append(
            (company_key in company_keys) or (cust_no in cust_nos and bool(cust_no))
        )

    enriched["company_key"] = enriched[COMPANY_NAME_COL].map(normalize_company_key)
    enriched["last_developed_at"] = last_developed_values
    enriched["recently_developed"] = recently_developed_values
    return enriched


def append_development_history(
    leads: pd.DataFrame,
    history_file: Path,
    *,
    source: str = "n8n",
    status: str = "developed",
    outcome: str | None = None,
    outcome_at: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history = load_development_history(history_file)
    default_developed_at = now_iso()
    run_id = run_id or run_id_from_now()

    rows: list[dict[str, Any]] = []
    for _, row in leads.iterrows():
        row_developed_at = (
            clean_text(row.get("developed_at"))
            or clean_text(row.get("manual_outreach_at"))
            or clean_text(row.get("outcome_at"))
            or default_developed_at
        )
        row_status = clean_text(row.get("status")) or status
        row_outcome = clean_text(row.get("outcome")) or outcome
        row_outcome_at = (
            clean_text(row.get("outcome_at"))
            or clean_text(row.get("manual_outreach_at"))
            or outcome_at
        )
        row_source = clean_text(row.get("source")) or source
        rows.append(
            {
                "developed_at": row_developed_at,
                "company_name": clean_text(row.get(COMPANY_NAME_COL) or row.get("company_name")),
                "company_key": normalize_company_key(row.get(COMPANY_NAME_COL) or row.get("company_name")),
                CUST_NO_COL: clean_text(row.get(CUST_NO_COL)),
                "rank": row.get("rank"),
                "similarity_score": row.get("similarity_score"),
                "source": row_source,
                "status": row_status,
                "outcome": row_outcome,
                "outcome_at": row_outcome_at,
                "run_id": run_id,
            }
        )

    new_rows = pd.DataFrame(rows)
    updated = new_rows if history.empty else pd.concat([history, new_rows], ignore_index=True)
    updated = updated[HISTORY_COLUMNS]
    dedupe_columns = ["developed_at", "company_key", CUST_NO_COL, "source", "status", "outcome"]
    dedupe_keys = updated[dedupe_columns].fillna("").astype(str)
    updated = updated[~dedupe_keys.duplicated(keep="last")]
    updated.to_csv(history_file, index=False, encoding="utf-8-sig")
    return updated



def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} 缺少必要欄位: {missing}")


def fill_missing_columns(df: pd.DataFrame, columns: list[str], default: Any = "") -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = default
    return df


def combine_text_fields(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = []
    for column in columns:
        if column in df.columns:
            parts.append(df[column].map(clean_text))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.strip()


def active_industry_text(df: pd.DataFrame) -> pd.Series:
    texts: list[str] = []
    for _, row in df.iterrows():
        tokens = [clean_text(row.get(INDUSTRY_TEXT_COL))]
        for column in INDUSTRY_COLS:
            try:
                is_active = float(row.get(column, 0) or 0) > 0
            except (TypeError, ValueError):
                is_active = False
            if is_active:
                tokens.append(column)
        texts.append(" ".join(token for token in tokens if token))
    return pd.Series(texts, index=df.index)


def tfidf_similarity(
    customer_text: pd.Series,
    potential_text: pd.Series,
    *,
    min_df: int = 1,
) -> np.ndarray:
    all_text = pd.concat([customer_text, potential_text], ignore_index=True).fillna("")
    if all_text.str.len().sum() == 0:
        return np.zeros((len(potential_text), len(customer_text)))

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=TEXT_NGRAM_RANGE,
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(all_text)
    customer_matrix = matrix[: len(customer_text)]
    potential_matrix = matrix[len(customer_text) :]
    return cosine_similarity(potential_matrix, customer_matrix)


def industry_similarity(customers: pd.DataFrame, potentials: pd.DataFrame) -> np.ndarray:
    customers = fill_missing_columns(customers, INDUSTRY_COLS, 0)
    potentials = fill_missing_columns(potentials, INDUSTRY_COLS, 0)

    customer_dummy = customers[INDUSTRY_COLS].fillna(0).astype(float).to_numpy() > 0
    potential_dummy = potentials[INDUSTRY_COLS].fillna(0).astype(float).to_numpy() > 0
    intersection = potential_dummy.astype(int) @ customer_dummy.astype(int).T
    potential_counts = potential_dummy.sum(axis=1)[:, None]
    customer_counts = customer_dummy.sum(axis=1)[None, :]
    union = potential_counts + customer_counts - intersection
    dummy_sim = np.divide(
        intersection,
        union,
        out=np.zeros((len(potentials), len(customers)), dtype=float),
        where=union > 0,
    )

    text_sim = tfidf_similarity(active_industry_text(customers), active_industry_text(potentials))

    return 0.65 * dummy_sim + 0.35 * text_sim


def numeric_similarity(customers: pd.DataFrame, potentials: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(potentials), len(customers)), dtype=float)
    counts = np.zeros_like(result)

    for column in NUMERIC_COLS:
        if column not in customers.columns or column not in potentials.columns:
            continue

        c = pd.to_numeric(customers[column], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(potentials[column], errors="coerce").to_numpy(dtype=float)

        c_log = np.log1p(np.where(np.isfinite(c) & (c >= 0), c, np.nan))
        p_log = np.log1p(np.where(np.isfinite(p) & (p >= 0), p, np.nan))

        diff = np.abs(p_log[:, None] - c_log[None, :])
        valid = np.isfinite(diff)
        sim = np.exp(-diff / 1.5)
        sim[~valid] = 0

        result += sim
        counts += valid.astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = np.divide(result, counts, out=np.zeros_like(result), where=counts > 0)
    return averaged


def location_similarity(customers: pd.DataFrame, potentials: pd.DataFrame) -> np.ndarray:
    c_city = customers.get("104_address_city", pd.Series([""] * len(customers))).map(clean_text).to_numpy()
    p_city = potentials.get("104_address_city", pd.Series([""] * len(potentials))).map(clean_text).to_numpy()
    c_dist = customers.get("104_address_district", pd.Series([""] * len(customers))).map(clean_text).to_numpy()
    p_dist = potentials.get("104_address_district", pd.Series([""] * len(potentials))).map(clean_text).to_numpy()

    city_match = (p_city[:, None] != "") & (p_city[:, None] == c_city[None, :])
    district_match = (p_dist[:, None] != "") & (p_dist[:, None] == c_dist[None, :])
    return city_match.astype(float) * 0.7 + district_match.astype(float) * 0.3


def snack_signal_similarity(customers: pd.DataFrame, potentials: pd.DataFrame) -> np.ndarray:
    c = customers.get(SNACK_SIGNAL_COL, pd.Series([False] * len(customers))).map(truthy).to_numpy()
    p = potentials.get(SNACK_SIGNAL_COL, pd.Series([False] * len(potentials))).map(truthy).to_numpy()

    p_matrix = p[:, None]
    c_matrix = c[None, :]
    similarity = np.zeros((len(potentials), len(customers)), dtype=float)
    similarity[p_matrix & c_matrix] = 1.0
    similarity[p_matrix & ~c_matrix] = 0.45
    similarity[~p_matrix & ~c_matrix] = 0.25
    similarity[~p_matrix & c_matrix] = 0.10
    return similarity


def duplicate_mask(customers: pd.DataFrame, potentials: pd.DataFrame) -> np.ndarray:
    c_names = customers[COMPANY_NAME_COL].map(normalize_company_key).to_numpy()
    p_names = potentials[COMPANY_NAME_COL].map(normalize_company_key).to_numpy()
    c_cust = customers.get(CUST_NO_COL, pd.Series([""] * len(customers))).map(clean_text).to_numpy()
    p_cust = potentials.get(CUST_NO_COL, pd.Series([""] * len(potentials))).map(clean_text).to_numpy()

    same_name = (p_names[:, None] != "") & (p_names[:, None] == c_names[None, :])
    same_cust = (p_cust[:, None] != "") & (p_cust[:, None] == c_cust[None, :])
    return same_name | same_cust


def weighted_pairwise_similarity(component_matrices: dict[str, np.ndarray]) -> np.ndarray:
    shape = next(iter(component_matrices.values())).shape
    weighted = np.zeros(shape, dtype=float)
    total_weight = 0.0

    for name, weight in COMPONENT_WEIGHTS.items():
        matrix = component_matrices.get(name)
        if matrix is None:
            continue
        weighted += matrix * weight
        total_weight += weight

    if total_weight == 0:
        return weighted
    return weighted / total_weight


def aggregate_scores(matrix: np.ndarray) -> dict[str, np.ndarray]:
    sorted_scores = np.sort(matrix, axis=1)[:, ::-1]
    max_score = sorted_scores[:, 0] if sorted_scores.shape[1] else np.zeros(matrix.shape[0])
    top3_mean = sorted_scores[:, : min(3, sorted_scores.shape[1])].mean(axis=1)
    top10_mean = sorted_scores[:, : min(10, sorted_scores.shape[1])].mean(axis=1)
    final = (
        AGGREGATION_WEIGHTS["max"] * max_score
        + AGGREGATION_WEIGHTS["top3_mean"] * top3_mean
        + AGGREGATION_WEIGHTS["top10_mean"] * top10_mean
    )
    return {
        "max": max_score,
        "top3_mean": top3_mean,
        "top10_mean": top10_mean,
        "final": final,
    }


def feature_coverage(df: pd.DataFrame) -> pd.Series:
    checks = pd.DataFrame(index=df.index)
    checks["profile"] = combine_text_fields(df, [PROFILE_COL]).str.len() > 0
    checks["welfare"] = combine_text_fields(df, WELFARE_TEXT_COLS).str.len() > 0
    checks["industry"] = active_industry_text(fill_missing_columns(df, INDUSTRY_COLS, 0)).str.len() > 0
    checks["size"] = df[[column for column in NUMERIC_COLS if column in df.columns]].notna().any(axis=1)
    checks["location"] = combine_text_fields(df, LOCATION_COLS).str.len() > 0
    return checks.mean(axis=1)


def build_top_match_columns(
    potentials: pd.DataFrame,
    customers: pd.DataFrame,
    total_similarity: np.ndarray,
    n: int = TOP_MATCH_N,
) -> tuple[list[str], list[str], list[float]]:
    top_names: list[str] = []
    top_scores_text: list[str] = []
    top_scores: list[float] = []

    for i in range(len(potentials)):
        order = np.argsort(total_similarity[i])[::-1][:n]
        names = customers.iloc[order][COMPANY_NAME_COL].map(clean_text).tolist()
        scores = total_similarity[i, order]
        top_names.append(" | ".join(names))
        top_scores_text.append(" | ".join(f"{score:.4f}" for score in scores))
        top_scores.append(float(scores[0]) if len(scores) else 0.0)

    return top_names, top_scores_text, top_scores


def score_potential_customers(customers: pd.DataFrame, potentials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(customers, [COMPANY_NAME_COL], "customer_profiles")
    require_columns(potentials, [COMPANY_NAME_COL], "random profiles")

    customers = fill_missing_columns(customers, [PROFILE_COL, INDUSTRY_TEXT_COL, SNACK_SIGNAL_COL] + WELFARE_TEXT_COLS + LOCATION_COLS + NUMERIC_COLS + INDUSTRY_COLS)
    potentials = fill_missing_columns(potentials, [PROFILE_COL, INDUSTRY_TEXT_COL, SNACK_SIGNAL_COL] + WELFARE_TEXT_COLS + LOCATION_COLS + NUMERIC_COLS + INDUSTRY_COLS)

    profile_sim = tfidf_similarity(customers[PROFILE_COL].map(clean_text), potentials[PROFILE_COL].map(clean_text))
    welfare_sim = tfidf_similarity(
        combine_text_fields(customers, WELFARE_TEXT_COLS),
        combine_text_fields(potentials, WELFARE_TEXT_COLS),
    )
    industry_sim = industry_similarity(customers, potentials)
    size_sim = numeric_similarity(customers, potentials)
    loc_sim = location_similarity(customers, potentials)
    snack_sim = snack_signal_similarity(customers, potentials)

    component_matrices = {
        "profile": profile_sim,
        "welfare": welfare_sim,
        "industry": industry_sim,
        "size": size_sim,
        "location": loc_sim,
        "snack_signal": snack_sim,
    }

    duplicates = duplicate_mask(customers, potentials)
    for matrix in component_matrices.values():
        matrix[duplicates] = 0

    total_sim = weighted_pairwise_similarity(component_matrices)
    total_sim[duplicates] = 0

    final_scores = aggregate_scores(total_sim)
    component_scores = {
        name: aggregate_scores(matrix)["final"]
        for name, matrix in component_matrices.items()
    }

    top_names, top_scores_text, top_scores = build_top_match_columns(potentials, customers, total_sim)

    ranking = potentials.copy()
    ranking["similarity_score"] = final_scores["final"]
    ranking["max_pair_similarity"] = final_scores["max"]
    ranking["top3_mean_similarity"] = final_scores["top3_mean"]
    ranking["top10_mean_similarity"] = final_scores["top10_mean"]
    ranking["profile_similarity_component"] = component_scores["profile"]
    ranking["welfare_similarity_component"] = component_scores["welfare"]
    ranking["industry_similarity_component"] = component_scores["industry"]
    ranking["size_similarity_component"] = component_scores["size"]
    ranking["location_similarity_component"] = component_scores["location"]
    ranking["snack_signal_similarity_component"] = component_scores["snack_signal"]
    ranking["top_existing_customer_names"] = top_names
    ranking["top_existing_customer_scores"] = top_scores_text
    ranking["top_existing_customer_score"] = top_scores
    ranking["is_existing_customer_duplicate"] = duplicates.any(axis=1)
    ranking["feature_coverage"] = feature_coverage(potentials)

    ranking = ranking.sort_values(
        by=["similarity_score", "welfare_similarity_component", "industry_similarity_component"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["score_percentile"] = ranking["similarity_score"].rank(pct=True) * 100

    top_match_rows = []
    for i, potential_row in potentials.iterrows():
        order = np.argsort(total_sim[i])[::-1][:TOP_MATCH_N]
        for match_rank, customer_idx in enumerate(order, start=1):
            top_match_rows.append(
                {
                    "potential_company": potential_row[COMPANY_NAME_COL],
                    "match_rank": match_rank,
                    "existing_customer": customers.iloc[customer_idx][COMPANY_NAME_COL],
                    "total_similarity": total_sim[i, customer_idx],
                    "profile_similarity": profile_sim[i, customer_idx],
                    "welfare_similarity": welfare_sim[i, customer_idx],
                    "industry_similarity": industry_sim[i, customer_idx],
                    "size_similarity": size_sim[i, customer_idx],
                    "location_similarity": loc_sim[i, customer_idx],
                    "snack_signal_similarity": snack_sim[i, customer_idx],
                }
            )

    return ranking, pd.DataFrame(top_match_rows)


def write_outputs(
    ranking: pd.DataFrame,
    top_matches: pd.DataFrame,
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_csv = output_dir / "cosine_similarity_ranking.csv"
    ranking_xlsx = output_dir / "cosine_similarity_ranking.xlsx"
    matches_csv = output_dir / "cosine_similarity_top_matches.csv"
    ranking_snapshot_csv = (
        output_dir / f"cosine_similarity_ranking_{run_id}.csv"
        if run_id
        else ranking_csv
    )
    ranking_snapshot_xlsx = (
        output_dir / f"cosine_similarity_ranking_{run_id}.xlsx"
        if run_id
        else ranking_xlsx
    )

    ranking.to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    if ranking_snapshot_csv != ranking_csv:
        ranking.to_csv(ranking_snapshot_csv, index=False, encoding="utf-8-sig")
    top_matches.to_csv(matches_csv, index=False, encoding="utf-8-sig")

    def write_ranking_workbook(path: Path) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            ranking.to_excel(writer, index=False, sheet_name="ranking")
            top_matches.to_excel(writer, index=False, sheet_name="top_matches")

            for sheet in writer.sheets.values():
                sheet.freeze_panes = "A2"
                for column_cells in sheet.columns:
                    header = clean_text(column_cells[0].value)
                    width = 18
                    if header in {PROFILE_COL, "福利制度"}:
                        width = 45
                    elif header in {"top_existing_customer_names", "top_existing_customer_scores"}:
                        width = 36
                    elif "similarity" in header or "score" in header:
                        width = 16
                    sheet.column_dimensions[column_cells[0].column_letter].width = width

    write_ranking_workbook(ranking_xlsx)
    if ranking_snapshot_xlsx != ranking_xlsx:
        write_ranking_workbook(ranking_snapshot_xlsx)

    print(f"Wrote: {ranking_csv}")
    print(f"Wrote: {ranking_xlsx}")
    if ranking_snapshot_csv != ranking_csv:
        print(f"Wrote: {ranking_snapshot_csv}")
        print(f"Wrote: {ranking_snapshot_xlsx}")
    print(f"Wrote: {matches_csv}")

    return {
        "ranking_csv": ranking_csv,
        "ranking_xlsx": ranking_xlsx,
        "ranking_snapshot_csv": ranking_snapshot_csv,
        "ranking_snapshot_xlsx": ranking_snapshot_xlsx,
        "top_matches_csv": matches_csv,
    }

def write_top_leads(
    ranking: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    output_file: Path = Path(DEFAULT_TOP_LEADS_FILE),
) -> pd.DataFrame:
    duplicate_col = (
        ranking["is_existing_customer_duplicate"].astype(bool)
        if "is_existing_customer_duplicate" in ranking.columns
        else pd.Series(False, index=ranking.index)
    )
    recent_col = (
        ranking["recently_developed"].astype(bool)
        if "recently_developed" in ranking.columns
        else pd.Series(False, index=ranking.index)
    )
    eligible = ranking[
        (~duplicate_col)
        & (~recent_col)
    ].copy()
    top_leads = eligible.head(top_n).copy()

    if "outreach_rank" not in top_leads.columns:
        top_leads.insert(0, "outreach_rank", np.arange(1, len(top_leads) + 1))
    top_leads["n8n_should_email"] = True
    top_leads["manual_physical_letter"] = top_leads["outreach_rank"] <= 10

    output_file.parent.mkdir(parents=True, exist_ok=True)
    top_leads.to_csv(output_file, index=False, encoding="utf-8-sig")

    if output_file.suffix.lower() == ".csv":
        xlsx_file = output_file.with_suffix(".xlsx")
    else:
        xlsx_file = output_file
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        top_leads.to_excel(writer, index=False, sheet_name="top_leads")
        worksheet = writer.sheets["top_leads"]
        worksheet.freeze_panes = "A2"
        for column_cells in worksheet.columns:
            header = clean_text(column_cells[0].value)
            width = 18
            if header in {PROFILE_COL, "福利制度"}:
                width = 45
            elif header in {"top_existing_customer_names", "top_existing_customer_scores"}:
                width = 36
            elif "similarity" in header or "score" in header:
                width = 16
            worksheet.column_dimensions[column_cells[0].column_letter].width = width

    print(f"Wrote: {output_file}")
    print(f"Wrote: {xlsx_file}")
    return top_leads


def crawl_random_potentials(
    output_dir: Path,
    history_file: Path,
    *,
    target_leads: int = DEFAULT_CRAWL_TARGET,
    max_candidates: int | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
    run_id: str | None = None,
) -> Path:
    from random_104_pipeline import generate_random_profiles

    run_id = run_id or run_id_from_now()
    max_candidates = max_candidates or max(target_leads * 5, target_leads)
    excluded_company_keys, excluded_cust_nos, _ = recent_development_filters(history_file, recent_days)

    crawled_csv = output_dir / f"weekly_random_104_profiles_{run_id}.csv"
    crawled_xlsx = output_dir / f"weekly_random_104_profiles_{run_id}.xlsx"

    generate_random_profiles(
        target_valid_n=target_leads,
        max_candidates_to_check=max_candidates,
        output_csv=crawled_csv,
        output_xlsx=crawled_xlsx,
        excluded_company_keys=excluded_company_keys,
        excluded_cust_nos=excluded_cust_nos,
    )
    return crawled_xlsx


def validate_potential_count(
    potentials: pd.DataFrame,
    *,
    target_leads: int,
    min_success_ratio: float,
    potentials_path: Path,
) -> None:
    minimum_required = max(1, int(target_leads * min_success_ratio))
    actual_count = len(potentials)
    if actual_count < minimum_required:
        raise RuntimeError(
            "random crawl produced too few usable potential companies: "
            f"{actual_count} rows from {potentials_path}, "
            f"minimum required is {minimum_required} "
            f"({min_success_ratio:.0%} of target_leads={target_leads})."
        )


def run_weekly_pipeline(
    *,
    customers_path: Path = Path(DEFAULT_CUSTOMER_FILE),
    potentials_path: Path = Path(DEFAULT_POTENTIAL_FILE),
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR),
    history_file: Path = Path(DEFAULT_HISTORY_FILE),
    crawl_before_score: bool = False,
    target_leads: int = DEFAULT_CRAWL_TARGET,
    max_candidates: int | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
    top_n: int = DEFAULT_TOP_N,
    top_leads_file: Path = Path(DEFAULT_TOP_LEADS_FILE),
    mark_top_n_developed: bool = False,
    enrich_emails: bool = False,
    min_crawl_success_ratio: float = DEFAULT_MIN_CRAWL_SUCCESS_RATIO,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or run_id_from_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    if crawl_before_score:
        potentials_path = crawl_random_potentials(
            output_dir,
            history_file,
            target_leads=target_leads,
            max_candidates=max_candidates,
            recent_days=recent_days,
            run_id=run_id,
        )

    customers = read_table(customers_path)
    potentials = read_table(potentials_path)
    if crawl_before_score:
        validate_potential_count(
            potentials,
            target_leads=target_leads,
            min_success_ratio=min_crawl_success_ratio,
            potentials_path=potentials_path,
        )

    ranking, top_matches = score_potential_customers(customers, potentials)
    ranking = add_development_history_columns(ranking, history_file, recent_days)
    output_paths = write_outputs(ranking, top_matches, output_dir, run_id=run_id)
    top_leads = write_top_leads(ranking, top_n=top_n, output_file=top_leads_file)

    email_found_count = None
    if enrich_emails:
        from email_enrichment import enrich_lead_emails

        top_leads = enrich_lead_emails(top_leads)
        top_leads.to_csv(top_leads_file, index=False, encoding="utf-8-sig")
        top_leads.to_excel(top_leads_file.with_suffix(".xlsx"), index=False)
        email_found_count = int(top_leads["Email"].map(clean_text).astype(bool).sum())

    history_rows = None
    if mark_top_n_developed:
        history_rows = append_development_history(
            top_leads,
            history_file,
            source="cosine_similarity_analysis.py",
            status="queued",
            run_id=run_id,
        )

    summary = {
        "run_id": run_id,
        "crawl_before_score": crawl_before_score,
        "customers_path": str(customers_path),
        "potentials_path": str(potentials_path),
        "output_dir": str(output_dir),
        "history_file": str(history_file),
        "ranking_csv": str(output_paths["ranking_csv"]),
        "ranking_xlsx": str(output_paths["ranking_xlsx"]),
        "ranking_snapshot_csv": str(output_paths["ranking_snapshot_csv"]),
        "ranking_snapshot_xlsx": str(output_paths["ranking_snapshot_xlsx"]),
        "top_matches_csv": str(output_paths["top_matches_csv"]),
        "top_leads_csv": str(top_leads_file),
        "top_leads_xlsx": str(top_leads_file.with_suffix(".xlsx")),
        "potential_count": int(len(potentials)),
        "top_leads_count": int(len(top_leads)),
        "email_enrichment_enabled": bool(enrich_emails),
        "email_found_count": email_found_count,
        "recently_developed_in_ranking": int(ranking["recently_developed"].sum()) if "recently_developed" in ranking else 0,
        "score_min": float(ranking["similarity_score"].min()) if not ranking.empty else 0.0,
        "score_max": float(ranking["similarity_score"].max()) if not ranking.empty else 0.0,
        "history_rows": int(len(history_rows)) if history_rows is not None else None,
    }
    return summary


def print_summary(ranking: pd.DataFrame) -> None:
    print("\nScoring summary")
    print(f"Potential companies: {len(ranking)}")
    print(f"Score range: {ranking['similarity_score'].min():.4f} - {ranking['similarity_score'].max():.4f}")
    print(f"Mean score: {ranking['similarity_score'].mean():.4f}")
    print(f"Duplicate existing-customer rows flagged: {int(ranking['is_existing_customer_duplicate'].sum())}")
    print("\nTop 20 ranking")
    columns = [
        "rank",
        COMPANY_NAME_COL,
        "similarity_score",
        "welfare_similarity_component",
        "industry_similarity_component",
        "top_existing_customer_names",
    ]
    print(ranking[columns].head(20).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosine similarity scoring for 104 random company leads.")
    parser.add_argument("--customers", default=DEFAULT_CUSTOMER_FILE, help="Existing customer profile table.")
    parser.add_argument("--potentials", default=DEFAULT_POTENTIAL_FILE, help="Potential customer profile table.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--crawl-before-score",
        action="store_true",
        help="Crawl random active 104 companies before scoring.",
    )
    parser.add_argument(
        "--target-leads",
        type=int,
        default=DEFAULT_CRAWL_TARGET,
        help="Target valid potential companies when crawling.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum 104 candidates to inspect when crawling.",
    )
    parser.add_argument(
        "--min-crawl-success-ratio",
        type=float,
        default=DEFAULT_MIN_CRAWL_SUCCESS_RATIO,
        help=(
            "Fail the weekly crawl if usable potential companies are below "
            "target_leads * this ratio."
        ),
    )
    parser.add_argument(
        "--history-file",
        default=None,
        help="Development history CSV. Defaults to <output-dir>/development_history.csv.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=DEFAULT_RECENT_DAYS,
        help="Companies developed within this many days are excluded from outreach and crawling.",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top lead count for n8n.")
    parser.add_argument(
        "--top-leads-file",
        default=None,
        help="n8n top leads CSV. Defaults to <output-dir>/n8n_top100_leads.csv.",
    )
    parser.add_argument(
        "--mark-top-n-developed",
        action="store_true",
        help="Immediately append top N leads to development history as queued. Usually let n8n mark after emails succeed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    customer_path = Path(args.customers)
    potential_path = Path(args.potentials)
    output_dir = Path(args.output_dir)
    history_file = Path(args.history_file) if args.history_file else output_dir / "development_history.csv"
    top_leads_file = Path(args.top_leads_file) if args.top_leads_file else output_dir / "n8n_top100_leads.csv"

    summary = run_weekly_pipeline(
        customers_path=customer_path,
        potentials_path=potential_path,
        output_dir=output_dir,
        history_file=history_file,
        crawl_before_score=args.crawl_before_score,
        target_leads=args.target_leads,
        max_candidates=args.max_candidates,
        recent_days=args.recent_days,
        top_n=args.top_n,
        top_leads_file=top_leads_file,
        mark_top_n_developed=args.mark_top_n_developed,
        min_crawl_success_ratio=args.min_crawl_success_ratio,
    )

    ranking = pd.read_csv(output_dir / "cosine_similarity_ranking.csv", encoding="utf-8-sig")
    print_summary(ranking)
    print("\nPipeline summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
