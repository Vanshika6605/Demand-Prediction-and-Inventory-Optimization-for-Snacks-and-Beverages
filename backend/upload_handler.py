import io
import pandas as pd

class ValidationError(Exception):
    pass

def parse_upload(uploaded_file) -> pd.DataFrame:
    """
    Parses a Streamlit UploadedFile (BytesIO) into a pandas DataFrame.
    """
    filename = uploaded_file.name
    content = uploaded_file.read()
    # Reset read pointer in case of double reading
    uploaded_file.seek(0)
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValidationError("Unsupported file format. Please upload a CSV or Excel file.")
    except Exception as e:
        if isinstance(e, ValidationError):
            raise e
        raise ValidationError(f"Failed to parse file: {str(e)}")
        
    return df
