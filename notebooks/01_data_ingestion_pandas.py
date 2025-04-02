#!/usr/bin/env python
# coding: utf-8

# # Online Retail Data Ingestion Pipeline
# 
# This notebook implements the data ingestion pipeline for the Online Retail dataset.
# It performs the following steps:
# 1. Load the raw Excel data
# 2. Perform initial data quality checks
# 3. Convert to appropriate formats for big data processing
# 4. Save to the processed data zone in CSV and Parquet formats
# 5. Generate data profile report

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.parquet as pq

# Set paths
RAW_DATA_PATH = "/home/ubuntu/big_data_project/data/raw/online_retail.xlsx"
PROCESSED_CSV_PATH = "/home/ubuntu/big_data_project/data/processed/online_retail.csv"
PROCESSED_PARQUET_PATH = "/home/ubuntu/big_data_project/data/processed/online_retail.parquet"
PROFILE_PATH = "/home/ubuntu/big_data_project/data/processed/data_profile.html"

print("Starting Online Retail Data Ingestion Pipeline...")

# Function to load data from Excel
def load_raw_data(file_path):
    """
    Load raw data from Excel file
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to perform data quality checks
def perform_data_quality_checks(df):
    """
    Perform initial data quality checks
    """
    print("\n=== Data Quality Checks ===")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    # Basic statistics
    print("\nBasic statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    return df

# Function to transform data
def transform_data(df):
    """
    Transform pandas DataFrame
    """
    print("\n=== Transforming Data ===")
    
    # Convert InvoiceDate to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Extract date components
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['Minute'] = df['InvoiceDate'].dt.minute
    
    # Calculate total amount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Filter out records with negative quantities (likely returns/cancellations)
    df_sales = df[df['Quantity'] > 0]
    df_returns = df[df['Quantity'] < 0]
    
    print(f"Number of sales transactions: {len(df_sales)}")
    print(f"Number of return transactions: {len(df_returns)}")
    
    # Convert InvoiceNo to string if it's not already
    if not pd.api.types.is_string_dtype(df['InvoiceNo']):
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    
    # Handle missing CustomerID
    missing_customer_id = df['CustomerID'].isna().sum()
    print(f"Rows with missing CustomerID: {missing_customer_id} ({missing_customer_id/len(df)*100:.2f}%)")
    
    return df

# Function to save data in CSV format
def save_as_csv(df, output_path):
    """
    Save DataFrame as CSV
    """
    print(f"\nSaving data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Data saved as CSV successfully!")

# Function to save data in Parquet format
def save_as_parquet(df, output_path):
    """
    Save DataFrame as Parquet using PyArrow
    """
    print(f"\nSaving data to {output_path}...")
    
    # Convert problematic columns to string to avoid PyArrow type errors
    df_copy = df.copy()
    df_copy['StockCode'] = df_copy['StockCode'].astype(str)
    df_copy['InvoiceNo'] = df_copy['InvoiceNo'].astype(str)
    
    # Convert Period type to string for YearMonth column
    if 'YearMonth' in df_copy.columns:
        df_copy['YearMonth'] = df_copy['YearMonth'].astype(str)
    
    # Handle NaN values in CustomerID
    if 'CustomerID' in df_copy.columns:
        df_copy['CustomerID'] = df_copy['CustomerID'].fillna(0).astype(int)
    
    table = pa.Table.from_pandas(df_copy)
    pq.write_table(table, output_path)
    print("Data saved as Parquet successfully!")

# Function to generate data profile
def generate_data_profile(df, output_path):
    """
    Generate a data profile report
    """
    try:
        from pandas_profiling import ProfileReport
        print(f"\nGenerating data profile report to {output_path}...")
        profile = ProfileReport(df, title="Online Retail Dataset Profile", explorative=True)
        profile.to_file(output_path)
        print("Data profile report generated successfully!")
    except ImportError:
        print("pandas-profiling not installed. Skipping profile generation.")
        try:
            # Alternative using ydata-profiling if available
            from ydata_profiling import ProfileReport
            print(f"\nGenerating data profile report using ydata-profiling to {output_path}...")
            profile = ProfileReport(df, title="Online Retail Dataset Profile", explorative=True)
            profile.to_file(output_path)
            print("Data profile report generated successfully!")
        except ImportError:
            print("ydata-profiling not installed either. Skipping profile generation.")

# Function to visualize data distributions
def visualize_data(df):
    """
    Create basic visualizations of the data
    """
    print("\n=== Creating Data Visualizations ===")
    
    # Create directory for visualizations
    os.makedirs("/home/ubuntu/big_data_project/images", exist_ok=True)
    
    # Plot 1: Distribution of sales by country
    plt.figure(figsize=(12, 6))
    country_counts = df['Country'].value_counts().head(10)
    sns.barplot(x=country_counts.index, y=country_counts.values)
    plt.title('Top 10 Countries by Number of Transactions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/big_data_project/images/country_distribution.png')
    print("Created visualization: Country distribution")
    
    # Plot 2: Monthly sales trend
    plt.figure(figsize=(12, 6))
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('YearMonth')['TotalAmount'].sum()
    plt.plot(monthly_sales.index.astype(str), monthly_sales.values)
    plt.title('Monthly Sales Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/big_data_project/images/monthly_sales.png')
    print("Created visualization: Monthly sales trend")
    
    # Plot 3: Distribution of order quantities
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Quantity'].clip(0, 100), bins=50)
    plt.title('Distribution of Order Quantities (Clipped at 100)')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/big_data_project/images/quantity_distribution.png')
    print("Created visualization: Quantity distribution")

# Main execution function
def main():
    """
    Main execution function
    """
    # Load raw data
    df = load_raw_data(RAW_DATA_PATH)
    
    # Perform data quality checks
    df = perform_data_quality_checks(df)
    
    # Transform data
    df = transform_data(df)
    
    # Save as CSV
    save_as_csv(df, PROCESSED_CSV_PATH)
    
    # Save as Parquet
    save_as_parquet(df, PROCESSED_PARQUET_PATH)
    
    # Create visualizations
    visualize_data(df)
    
    # Try to generate data profile
    try:
        generate_data_profile(df, PROFILE_PATH)
    except Exception as e:
        print(f"Error generating data profile: {e}")
    
    # Show sample data
    print("\nSample data from processed dataset:")
    print(df.head())
    
    print("\nData ingestion pipeline completed successfully!")
    print(f"Processed data saved to: {PROCESSED_CSV_PATH} and {PROCESSED_PARQUET_PATH}")
    print(f"Visualizations saved to: ../images/")

# Execute the pipeline
if __name__ == "__main__":
    main()
