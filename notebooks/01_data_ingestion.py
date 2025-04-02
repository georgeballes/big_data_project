#!/usr/bin/env python
# coding: utf-8

# # Online Retail Data Ingestion Pipeline
# 
# This notebook implements the data ingestion pipeline for the Online Retail dataset.
# It performs the following steps:
# 1. Load the raw Excel data
# 2. Perform initial data quality checks
# 3. Convert to appropriate formats for big data processing
# 4. Save to the processed data zone in Parquet format
# 5. Create Delta tables for efficient querying

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofmonth, hour, minute
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
RAW_DATA_PATH = "../data/raw/online_retail.xlsx"
PROCESSED_DATA_PATH = "../data/processed/online_retail.parquet"
DELTA_DATA_PATH = "../data/curated/online_retail_delta"

# Create Spark session
spark = SparkSession.builder \
    .appName("Online Retail Data Ingestion") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.3.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

print("Spark session created successfully!")

# Function to load data from Excel
def load_raw_data(file_path):
    """
    Load raw data from Excel file
    """
    print(f"Loading data from {file_path}...")
    # For large Excel files, it's better to use pandas first
    # and then convert to Spark DataFrame
    pdf = pd.read_excel(file_path)
    print(f"Data loaded successfully with {pdf.shape[0]} rows and {pdf.shape[1]} columns.")
    return pdf

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
def transform_data(pdf):
    """
    Transform pandas DataFrame and convert to Spark DataFrame
    """
    print("\n=== Transforming Data ===")
    
    # Convert to Spark DataFrame
    df = spark.createDataFrame(pdf)
    
    # Convert InvoiceDate to timestamp
    df = df.withColumn("InvoiceDate", to_timestamp(col("InvoiceDate")))
    
    # Extract date components
    df = df.withColumn("Year", year(col("InvoiceDate"))) \
           .withColumn("Month", month(col("InvoiceDate"))) \
           .withColumn("Day", dayofmonth(col("InvoiceDate"))) \
           .withColumn("Hour", hour(col("InvoiceDate"))) \
           .withColumn("Minute", minute(col("InvoiceDate")))
    
    # Calculate total amount
    df = df.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
    
    # Filter out records with negative quantities (likely returns/cancellations)
    df_sales = df.filter(col("Quantity") > 0)
    df_returns = df.filter(col("Quantity") < 0)
    
    print(f"Number of sales transactions: {df_sales.count()}")
    print(f"Number of return transactions: {df_returns.count()}")
    
    return df

# Function to save data in Parquet format
def save_as_parquet(df, output_path):
    """
    Save Spark DataFrame as Parquet
    """
    print(f"\nSaving data to {output_path}...")
    df.write.mode("overwrite").parquet(output_path)
    print("Data saved as Parquet successfully!")

# Function to create Delta table
def create_delta_table(df, output_path):
    """
    Save Spark DataFrame as Delta table
    """
    print(f"\nCreating Delta table at {output_path}...")
    df.write.format("delta").mode("overwrite").save(output_path)
    print("Delta table created successfully!")

# Main execution function
def main():
    """
    Main execution function
    """
    print("Starting Online Retail Data Ingestion Pipeline...")
    
    # Load raw data
    pdf = load_raw_data(RAW_DATA_PATH)
    
    # Perform data quality checks
    pdf = perform_data_quality_checks(pdf)
    
    # Transform data
    df = transform_data(pdf)
    
    # Save as Parquet
    save_as_parquet(df, PROCESSED_DATA_PATH)
    
    # Create Delta table
    create_delta_table(df, DELTA_DATA_PATH)
    
    # Show sample data
    print("\nSample data from processed dataset:")
    df.show(5)
    
    print("\nData ingestion pipeline completed successfully!")

# Execute the pipeline
if __name__ == "__main__":
    main()
    spark.stop()
