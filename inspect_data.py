import pandas as pd
import os

file_path = 'd:/programming/Antigravity/App/Dataset (1).xlsx'

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nInfo:\n")
    df.info()
except Exception as e:
    print(f"Error reading file: {e}")
