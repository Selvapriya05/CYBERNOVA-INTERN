
from pathlib import Path
import pandas as pd

# Find the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

print("=" * 60)
print("E-COMMERCE BUSINESS INTELLIGENCE PLATFORM")
print("DATASET INSPECTION")
print("=" * 60)

# Find all CSV files
csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV files found in the dataset folder.")
else:
    print(f"\nTotal CSV files found: {len(csv_files)}")

    for file_path in csv_files:
        print("\n" + "-" * 60)
        print(f"FILE: {file_path.name}")

        try:
            df = pd.read_csv(file_path)

            print(f"Rows and columns: {df.shape}")
            print("\nColumn names:")
            print(df.columns.tolist())

            print("\nFirst 3 rows:")
            print(df.head(3).to_string(index=False))

            print("\nMissing values:")
            print(df.isnull().sum().to_string())

        except Exception as error:
            print(f"Could not read file: {error}")

print("\nDataset inspection completed.")