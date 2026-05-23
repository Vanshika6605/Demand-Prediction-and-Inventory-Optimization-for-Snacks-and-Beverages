"""
Validates an uploaded FMCG sales/inventory CSV before it enters the pipeline.

ML feature matrices (e.g. augmented_ml_ready_dataset.csv) may have multiple rows
per (date, product_name) with different lag/rolling features. Only full-row
duplicates are rejected for those files; aggregated daily CSVs dedupe on
(date, product_name) with a warning.
"""

import pandas as pd
from typing import Tuple

from backend.upload_handler import ValidationError, parse_upload

REQUIRED_COLUMNS = {
    "date",
    "product_name",
    "category",
    "quantity_sold",
    "net_stock",
}

ML_FEATURE_INDICATOR_COLS = {
    "lag_1", "lag_7", "lag_14",
    "rolling_7_mean", "rolling_14_mean", "rolling_7_std",
    "day_of_week", "week_of_year",
}

DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]


def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Returns: ml_feature_matrix | aggregated_daily | raw_transactional
    """
    cols_lower = {c.lower() for c in df.columns}

    if ML_FEATURE_INDICATOR_COLS & cols_lower:
        return "ml_feature_matrix"

    if {"transaction_id", "order_id", "invoice_id"} & cols_lower:
        return "raw_transactional"

    return "aggregated_daily"


def _normalize_column_aliases(df: pd.DataFrame) -> None:
    """Map common upload column names to the canonical schema (in-place)."""
    if "inventory_stock" in df.columns and "net_stock" not in df.columns:
        df["net_stock"] = df["inventory_stock"]
    elif "net_stock" in df.columns and "inventory_stock" not in df.columns:
        df["inventory_stock"] = df["net_stock"]

    for price_col in ("avg_unit_price", "unit_price", "price"):
        if price_col in df.columns:
            if "avg_unit_price" not in df.columns:
                df["avg_unit_price"] = df[price_col]
            if "price" not in df.columns:
                df["price"] = df[price_col]
            break


def _derive_category_column(df: pd.DataFrame) -> None:
    """Rebuild category from one-hot category_* columns when needed."""
    if "category" in df.columns:
        return

    cat_cols = [c for c in df.columns if c.startswith("category_")]
    if not cat_cols:
        return

    def _row_category(row):
        for col in cat_cols:
            if row[col] == 1:
                return col.replace("category_", "", 1).replace("_", " ")
        return "Unknown"

    df["category"] = df.apply(_row_category, axis=1)


def _parse_dates(series: pd.Series) -> Tuple[pd.Series | None, str | None]:
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(series, format=fmt, errors="raise"), None
        except Exception:
            continue

    try:
        parsed = pd.to_datetime(series, errors="coerce")
        null_count = parsed.isna().sum()
        if null_count / len(series) > 0.05:
            return None, (
                f"Date column could not be parsed. "
                f"{null_count} values are unreadable. "
                f"Expected formats: YYYY-MM-DD or DD-MM-YYYY."
            )
        return parsed, None
    except Exception:
        return None, (
            "Date column could not be parsed. "
            "Expected format: YYYY-MM-DD (e.g. 2024-03-15)."
        )


def validate_upload(df: pd.DataFrame, filename: str = "upload") -> dict:
    """
    Validates df in-place (column renames, dedupe) and returns a result dict.
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {"filename": filename},
    }

    def add_error(etype, msg):
        result["errors"].append({"type": etype, "message": msg})
        result["valid"] = False

    if df is None or len(df) == 0:
        add_error("empty_file",
                  "Uploaded file is empty. Please upload a file with at least 30 days of data.")
        return result

    if len(df) < 30:
        result["warnings"].append(
            f"Only {len(df)} rows detected. For accurate forecasting, "
            "at least 30 days of data is recommended."
        )

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    _normalize_column_aliases(df)
    _derive_category_column(df)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        add_error(
            "missing_columns",
            f"Required columns not found: {', '.join(sorted(missing))}. "
            f"Your file has: {', '.join(df.columns[:10])}"
            f"{'...' if len(df.columns) > 10 else ''}.",
        )
        return result

    dataset_type = detect_dataset_type(df)
    result["info"]["dataset_type"] = dataset_type

    parsed_dates, date_error = _parse_dates(df["date"])
    if date_error:
        add_error("invalid_date", date_error)
        return result
    df["date"] = parsed_dates

    for col in ["quantity_sold", "net_stock"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        null_count = df[col].isna().sum()
        if null_count > 0:
            add_error(
                "non_numeric",
                f"Column '{col}' contains {null_count} non-numeric "
                f"or empty values. All sales and stock values must be numbers.",
            )

    if not result["valid"]:
        return result

    neg = (df["quantity_sold"] < 0).sum()
    if neg > 0:
        add_error(
            "negative_quantity",
            f"'quantity_sold' contains {neg} negative values. All quantities must be ≥ 0.",
        )
        return result

    for col in REQUIRED_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            pct = round(null_count / len(df) * 100, 1)
            if pct > 20:
                add_error(
                    "excessive_nulls",
                    f"Column '{col}' has {null_count} missing values ({pct}%). "
                    "More than 20% nulls in a required column is not acceptable.",
                )
            else:
                result["warnings"].append(
                    f"Column '{col}' has {null_count} missing values ({pct}%). "
                    "These will be forward-filled during preprocessing."
                )

    if not result["valid"]:
        return result

    if dataset_type == "ml_feature_matrix":
        true_full_duplicates = int(df.duplicated(keep="first").sum())
        if true_full_duplicates > 0:
            df.drop_duplicates(inplace=True)
            result["warnings"].append(
                f"{true_full_duplicates} fully identical rows were removed automatically. "
                "These were exact duplicates (every column identical)."
            )
        result["info"]["true_duplicates_removed"] = true_full_duplicates

        combo_count = df.groupby(["date", "product_name"]).ngroups
        total_rows = len(df)
        if combo_count < total_rows:
            result["warnings"].append(
                f"ML feature matrix detected ({total_rows:,} rows, "
                f"{combo_count:,} unique date×product combinations). "
                "Multiple rows per date+product are expected and accepted."
            )
    else:
        date_product_dupes = int(df.duplicated(subset=["date", "product_name"], keep="first").sum())
        if date_product_dupes > 0:
            df.drop_duplicates(subset=["date", "product_name"], keep="last", inplace=True)
            result["warnings"].append(
                f"{date_product_dupes} rows with duplicate (date, product_name) "
                "combinations were found. The last occurrence was kept."
            )
        result["info"]["true_duplicates_removed"] = date_product_dupes

    date_min = df["date"].min()
    date_max = df["date"].max()
    date_span_days = (date_max - date_min).days
    result["info"]["date_range"] = (str(date_min.date()), str(date_max.date()))

    if date_span_days < 14:
        result["warnings"].append(
            f"Date range is only {date_span_days} days "
            f"({date_min.date()} → {date_max.date()}). "
            "Lag features require at least 14 days of history."
        )

    result["info"].update({
        "row_count": len(df),
        "column_count": len(df.columns),
        "products": int(df["product_name"].nunique()),
        "categories": int(df["category"].nunique()) if "category" in df.columns else 0,
    })

    return result


def validate_upload_file(uploaded_file) -> tuple[pd.DataFrame | None, dict]:
    """Parse a Streamlit upload, validate, and return (dataframe, result)."""
    try:
        df = parse_upload(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        result = validate_upload(df, uploaded_file.name)
        if not result["valid"]:
            return None, result
        return df, result
    except ValidationError as exc:
        return None, {
            "valid": False,
            "errors": [{"type": "parse_error", "message": str(exc)}],
            "warnings": [],
            "info": {},
        }
    except Exception as exc:
        return None, {
            "valid": False,
            "errors": [{"type": "unexpected", "message": f"Validation failed: {exc}"}],
            "warnings": [],
            "info": {},
        }


def display_validation_result(result: dict, use_sidebar: bool = False):
    """Render validation errors, warnings, and success in Streamlit."""
    import streamlit as st

    ui = st.sidebar if use_sidebar else st

    if result["errors"]:
        ui.error("**Validation Errors — upload rejected:**")
        for err in result["errors"]:
            ui.error(f"- {err['message']}")

    for warning in result["warnings"]:
        ui.warning(warning)

    if result["valid"]:
        info = result["info"]
        ui.success(
            f"File validated successfully. "
            f"{info.get('row_count', 0):,} rows | "
            f"{info.get('products', 0)} products | "
            f"{info.get('categories', 0)} categories | "
            f"Date range: {info.get('date_range', ('?', '?'))[0]} → "
            f"{info.get('date_range', ('?', '?'))[1]}"
        )
        if info.get("dataset_type") == "ml_feature_matrix":
            ui.info(
                "ML feature matrix detected (lag/rolling columns present). "
                "Multiple rows per date+product are valid; preprocessing will preserve them."
            )
