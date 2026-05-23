ERROR_MESSAGES = {
    "missing_columns": lambda cols: f"Upload rejected: required columns missing — {', '.join(cols)}",
    "null_values": lambda cols: f"Data quality issue: null values found in {', '.join(cols)}. Please clean your file and re-upload.",
    "invalid_date": "Date column could not be parsed. Expected format: YYYY-MM-DD",
    "negative_quantity": "quantity_sold contains negative values. All sales quantities must be ≥ 0.",
    "duplicate_rows": lambda n: f"{n} duplicate (date, product_name) rows detected and removed automatically.",
    "empty_file": "Uploaded file is empty. Please upload a file with at least 30 days of data.",
    "model_not_trained": "No trained model found. Please upload a dataset and run the forecasting engine first.",
    "api_timeout": "Backend server did not respond. Ensure uvicorn is running on port 8000.",
}
