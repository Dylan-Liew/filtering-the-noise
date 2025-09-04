import json

from dotenv import dotenv_values
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

config = dotenv_values()
engine = create_engine(config["DB_URI"])

merged_df = pd.read_parquet("data/cleaned_data.parquet", engine="pyarrow")

def to_json_string(data):
    # Handle nulls or other non-list types
    if not isinstance(data, (list, np.ndarray)):
        return None # Or '[]' if you prefer
    return json.dumps(data.tolist() if isinstance(data, np.ndarray) else data)

# Apply the conversion to the columns that contain lists/arrays
merged_df['category'] = merged_df['category'].apply(to_json_string)
merged_df['images'] = merged_df['images'].apply(to_json_string)

try:
    print("Writing merged_df to the database...")
    merged_df.to_sql(
        name='data',
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=10000,
        method='multi',
    )
    print("Successfully wrote merged_df to the 'merged_data' table.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Dispose of the connection pool
    engine.dispose()
    print("\nDatabase connection closed.")
