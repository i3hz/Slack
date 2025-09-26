import os
import xarray as xr
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
SQLITE_URI = "sqlite:///argo_data.db"
DATA_DIR = "./argo_data/"
CHROMA_PERSIST_DIR = "./chroma_db"
# You will need an OpenAI API key set as an environment variable: OPENAI_API_KEY

def find_best_variable(ds, potential_names):
    """
    Finds the best variable from a list of potential names by checking for existence
    and ensuring it contains at least one valid, non-null data point.
    """
    for name in potential_names:
        if name in ds.variables:
            # Check if there's any valid (non-NaN) data in the variable.
            # .any() is crucial to see if at least one value exists.
            if pd.notna(ds[name].values).any():
                print(f"Found valid data in variable: '{name}'")
                return name
    print(f"Could not find a variable with valid data from list: {potential_names}")
    return None

class DataProcessor:
    """
    Handles the ETL process for ARGO NetCDF files, following the user's specified structure.
    """
    def __init__(self, sqlite_uri):
        self.sql_engine = create_engine(sqlite_uri)
        print(f"Successfully connected to SQLite database at {sqlite_uri}")

    def process_netcdf_file(self, file_path):
        """
        Processes a single NetCDF file using the user-provided logic.
        """
        print(f"\n--- Processing File: {os.path.basename(file_path)} ---")
        try:
            with xr.open_dataset(file_path) as ds:
                # --- 1. Intelligently Identify Correct Column Names ---
                print("Searching for best available data columns...")
                name_map = {
                    'juld': find_best_variable(ds, ['juld', 'JULD']),
                    'latitude': find_best_variable(ds, ['latitude', 'LATITUDE']),
                    'longitude': find_best_variable(ds, ['longitude', 'LONGITUDE']),
                    'platform_number': find_best_variable(ds, ['platform_number', 'PLATFORM_NUMBER']),
                    'pres': find_best_variable(ds, ['pres_adjusted', 'PRES_ADJUSTED', 'pres', 'PRES']),
                    'temp': find_best_variable(ds, ['temp_adjusted', 'TEMP_ADJUSTED', 'temp', 'TEMP']),
                    'psal': find_best_variable(ds, ['psal_adjusted', 'PSAL_ADJUSTED', 'psal', 'PSAL']), # Optional
                }

                # Check if mandatory columns were found
                mandatory_vars = ['juld', 'latitude', 'longitude', 'platform_number', 'pres', 'temp']
                if not all(name_map[var] for var in mandatory_vars):
                    print(f"[ERROR] Skipping file: Could not find all mandatory variables with valid data. Missing: {[k for k, v in name_map.items() if not v and k in mandatory_vars]}")
                    return None, None
                
                # --- 2. Extract ONLY the required columns (user's method) ---
                cols_to_extract = [v for k, v in name_map.items() if v is not None]
                print(f"Extracting columns: {cols_to_extract}")
                
                df = ds[cols_to_extract].to_dataframe().reset_index()

                # --- 3. Perform Data Cleaning (user's method) ---
                print("Cleaning DataFrame...")
                
                # Define a mapping from the source column names to our standard names
                rename_dict = {
                    name_map['juld']: 'timestamp',
                    name_map['latitude']: 'latitude',
                    name_map['longitude']: 'longitude',
                    name_map['platform_number']: 'float_id',
                    name_map['pres']: 'pressure',
                    name_map['temp']: 'temperature'
                }
                if name_map['psal']:
                    rename_dict[name_map['psal']] = 'salinity'

                df.rename(columns=rename_dict, inplace=True)
                
                # 1. Drop rows where the essential measurement (temp) is missing
                df.dropna(subset=['temperature'], inplace=True)

                # 2. Clean the float ID ('platform_number')
                if 'float_id' in df.columns and df['float_id'].dtype == 'object':
                     df['float_id'] = df['float_id'].str.decode('utf-8').str.strip()

                if df.empty:
                    print(f"[WARNING] Skipping file {file_path}. DataFrame is empty after cleaning.")
                    return None, None
                
                print(f"Successfully extracted {len(df)} valid records.")

                # --- 4. Create Metadata Summary for Vector Store ---
                float_id_val = df['float_id'].iloc[0]
                start_date = pd.to_datetime(df['timestamp'].min()).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(df['timestamp'].max()).strftime('%Y-%m-%d')
                lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
                lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
                
                measurements_desc = "temperature and pressure profiles."
                if 'salinity' in df.columns:
                    measurements_desc = "temperature, salinity, and pressure profiles."

                metadata_summary = (
                    f"ARGO float with ID {float_id_val}. "
                    f"It recorded data from {start_date} to {end_date}. "
                    f"Its geographical operational area is between latitude {lat_min:.2f} and {lat_max:.2f}, "
                    f"and longitude {lon_min:.2f} and {lon_max:.2f}. "
                    f"The float contains measurements for {measurements_desc}"
                )
                doc = Document(page_content=metadata_summary, metadata={"source": file_path, "float_id": str(float_id_val)})
                
                return df, doc

        except Exception as e:
            print(f"An unexpected error occurred while processing file {file_path}: {e}")
            return None, None

    def run_etl(self, data_dir):
        """Runs the full ETL pipeline on a directory of NetCDF files."""
        all_dataframes = []
        all_documents = []
        
        print(f"\nStarting ETL process on directory: {data_dir}")
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".nc"):
                file_path = os.path.join(data_dir, filename)
                df, doc = self.process_netcdf_file(file_path)
                if df is not None and doc is not None:
                    all_dataframes.append(df)
                    all_documents.append(doc)
        
        if not all_dataframes:
            print("\nNo valid data was processed from any files. Exiting.")
            return

        print("\nConcatenating all dataframes...")
        full_df = pd.concat(all_dataframes, ignore_index=True)
        
        # --- Load to SQLite ---
        print("Loading data into SQLite...")
        # Select columns to load to avoid pandas/SQLAlchemy index issues
        sql_columns = ['timestamp', 'latitude', 'longitude', 'float_id', 'pressure', 'temperature']
        if 'salinity' in full_df.columns:
            sql_columns.append('salinity')
        
        full_df[sql_columns].to_sql('measurements', self.sql_engine, if_exists='replace', index=False)
        print(f"Successfully loaded {len(full_df)} records into 'measurements' table.")

        # --- Load to Chroma Vector Store ---
        print("Creating and loading metadata into ChromaDB...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        vector_store.persist()
        print(f"Successfully created vector store with {len(all_documents)} documents.")


if __name__ == '__main__':
    print("Starting Data Processing...")
    processor = DataProcessor(SQLITE_URI)
    processor.run_etl(DATA_DIR)
    print("Data Processing Finished.")

