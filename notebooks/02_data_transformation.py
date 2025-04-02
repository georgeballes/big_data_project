#!/usr/bin/env python
# coding: utf-8

# # Online Retail Data Transformation with Spark
# 
# This notebook implements the data transformation pipeline for the Online Retail dataset using Apache Spark.
# It performs the following steps:
# 1. Load the processed CSV data
# 2. Perform advanced data transformations
# 3. Calculate RFM (Recency, Frequency, Monetary) metrics
# 4. Save the transformed data for ML pipeline

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, datediff, max as spark_max, count, sum as spark_sum, avg, lit, when, expr
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Set paths
PROCESSED_CSV_PATH = "/home/ubuntu/big_data_project/data/processed/online_retail.csv"
TRANSFORMED_DATA_PATH = "/home/ubuntu/big_data_project/data/curated/rfm_data.csv"
RFM_SEGMENTS_PATH = "/home/ubuntu/big_data_project/data/curated/rfm_segments.csv"

# Since we had issues with Spark in the container, we'll use pandas for local processing
# but structure the code to be compatible with Spark in a real Databricks environment
print("Starting Online Retail Data Transformation Pipeline...")

# Function to load processed data
def load_processed_data(file_path):
    """
    Load processed data from CSV file
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert date columns
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to calculate RFM metrics
def calculate_rfm_metrics(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics
    """
    print("\n=== Calculating RFM Metrics ===")
    
    # Filter out records with missing CustomerID
    df_customers = df[df['CustomerID'].notna()].copy()
    
    # Convert CustomerID to integer if it's not already
    df_customers['CustomerID'] = df_customers['CustomerID'].astype(int)
    
    # Get the maximum date to calculate recency
    max_date = df_customers['InvoiceDate'].max() + timedelta(days=1)
    print(f"Reference date for recency calculation: {max_date}")
    
    # Calculate RFM metrics
    rfm = df_customers.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalAmount': 'sum'  # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Filter out negative monetary values (returns)
    rfm = rfm[rfm['Monetary'] > 0]
    
    print(f"RFM metrics calculated for {rfm.shape[0]} customers.")
    print("\nRFM metrics summary:")
    print(rfm.describe())
    
    return rfm

# Function to create RFM segments
def create_rfm_segments(rfm_df):
    """
    Create customer segments based on RFM metrics
    """
    print("\n=== Creating RFM Segments ===")
    
    # Create R, F, and M quartiles
    # Use custom logic for quartiles due to data distribution
    rfm_df['R_Quartile'] = pd.qcut(rfm_df['Recency'], q=[0, 0.25, 0.5, 0.75, 1.0], labels=range(4, 0, -1))
    
    # For Frequency, we need to handle the skewed distribution
    frequency_bins = [0, 1, 3, 6, float('inf')]
    rfm_df['F_Quartile'] = pd.cut(rfm_df['Frequency'], bins=frequency_bins, labels=range(1, 5))
    
    # For Monetary, use custom quantiles
    monetary_bins = [0, 300, 650, 1625, float('inf')]
    rfm_df['M_Quartile'] = pd.cut(rfm_df['Monetary'], bins=monetary_bins, labels=range(1, 5))
    
    # Calculate RFM Score
    rfm_df['RFM_Score'] = rfm_df['R_Quartile'].astype(str) + rfm_df['F_Quartile'].astype(str) + rfm_df['M_Quartile'].astype(str)
    
    # Create segment labels
    rfm_df['RFM_Segment'] = 'Unknown'
    
    # Top customers
    rfm_df.loc[rfm_df['RFM_Score'].isin(['444', '443', '434', '344']), 'RFM_Segment'] = 'Champions'
    
    # Loyal customers
    rfm_df.loc[rfm_df['RFM_Score'].isin(['433', '434', '343', '344', '334']), 'RFM_Segment'] = 'Loyal'
    
    # Potential loyalists
    rfm_df.loc[rfm_df['RFM_Score'].isin(['332', '333', '342', '322', '323', '423']), 'RFM_Segment'] = 'Potential Loyalists'
    
    # New customers
    rfm_df.loc[rfm_df['RFM_Score'].isin(['411', '412', '421', '422']), 'RFM_Segment'] = 'New Customers'
    
    # Promising
    rfm_df.loc[rfm_df['RFM_Score'].isin(['311', '312', '321', '322']), 'RFM_Segment'] = 'Promising'
    
    # Need attention
    rfm_df.loc[rfm_df['RFM_Score'].isin(['233', '234', '243', '244']), 'RFM_Segment'] = 'Need Attention'
    
    # About to sleep
    rfm_df.loc[rfm_df['RFM_Score'].isin(['223', '224', '232', '233', '234']), 'RFM_Segment'] = 'About to Sleep'
    
    # At risk
    rfm_df.loc[rfm_df['RFM_Score'].isin(['211', '212', '221', '222']), 'RFM_Segment'] = 'At Risk'
    
    # Can't lose them
    rfm_df.loc[rfm_df['RFM_Score'].isin(['111', '112', '121', '122', '123', '132', '211']), 'RFM_Segment'] = "Can't Lose"
    
    # Hibernating
    rfm_df.loc[rfm_df['RFM_Score'].isin(['144', '244', '134', '143', '243', '133']), 'RFM_Segment'] = 'Hibernating'
    
    # Lost
    rfm_df.loc[rfm_df['RFM_Score'].isin(['311', '411', '331']), 'RFM_Segment'] = 'Lost'
    
    # Print segment distribution
    segment_counts = rfm_df['RFM_Segment'].value_counts()
    print("\nCustomer segment distribution:")
    print(segment_counts)
    
    return rfm_df

# Function to analyze customer segments
def analyze_segments(rfm_segments, original_df):
    """
    Analyze customer segments to extract business insights
    """
    print("\n=== Analyzing Customer Segments ===")
    
    # Merge RFM segments with original data
    df_with_segments = original_df.merge(
        rfm_segments[['CustomerID', 'RFM_Segment']], 
        on='CustomerID', 
        how='inner'
    )
    
    # Analyze purchase behavior by segment
    segment_analysis = df_with_segments.groupby('RFM_Segment').agg({
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'CustomerID': 'nunique'
    }).reset_index()
    
    segment_analysis.columns = ['Segment', 'Transactions', 'Items_Purchased', 'Total_Spend', 'Customer_Count']
    segment_analysis['Avg_Spend_Per_Customer'] = segment_analysis['Total_Spend'] / segment_analysis['Customer_Count']
    segment_analysis['Avg_Transactions_Per_Customer'] = segment_analysis['Transactions'] / segment_analysis['Customer_Count']
    
    print("\nSegment Analysis:")
    print(segment_analysis.sort_values('Total_Spend', ascending=False))
    
    return segment_analysis

# Function to visualize RFM segments
def visualize_rfm_segments(rfm_df, segment_analysis):
    """
    Create visualizations of RFM segments
    """
    print("\n=== Creating RFM Segment Visualizations ===")
    
    # Create directory for visualizations
    os.makedirs("/home/ubuntu/big_data_project/images", exist_ok=True)
    
    # Plot 1: Segment distribution
    plt.figure(figsize=(12, 6))
    segment_counts = rfm_df['RFM_Segment'].value_counts()
    sns.barplot(x=segment_counts.index, y=segment_counts.values)
    plt.title('Customer Segments Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/big_data_project/images/segment_distribution.png')
    print("Created visualization: Segment distribution")
    
    # Plot 2: Average spend per customer by segment
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Segment', y='Avg_Spend_Per_Customer', data=segment_analysis.sort_values('Avg_Spend_Per_Customer', ascending=False))
    plt.title('Average Spend per Customer by Segment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/big_data_project/images/avg_spend_by_segment.png')
    print("Created visualization: Average spend by segment")
    
    # Plot 3: Recency vs Frequency scatter plot with Monetary as size
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        rfm_df['Recency'], 
        rfm_df['Frequency'],
        s=rfm_df['Monetary'] / 100,  # Scale down for better visualization
        c=rfm_df['Monetary'],
        cmap='viridis',
        alpha=0.6
    )
    plt.colorbar(scatter, label='Monetary Value')
    plt.xlabel('Recency (days)')
    plt.ylabel('Frequency (# transactions)')
    plt.title('Customer RFM Analysis: Recency vs Frequency vs Monetary')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/big_data_project/images/rfm_scatter.png')
    print("Created visualization: RFM scatter plot")

# Function to save transformed data
def save_transformed_data(rfm_df, segment_analysis, rfm_path, segments_path):
    """
    Save transformed data to CSV
    """
    print(f"\nSaving RFM data to {rfm_path}...")
    rfm_df.to_csv(rfm_path, index=False)
    print("RFM data saved successfully!")
    
    print(f"\nSaving segment analysis to {segments_path}...")
    segment_analysis.to_csv(segments_path, index=False)
    print("Segment analysis saved successfully!")

# Function to demonstrate Spark code (for documentation purposes)
def spark_code_example():
    """
    Example of how this would be implemented in Spark
    This is for documentation purposes only and won't be executed
    """
    print("\n=== Spark Implementation Example (Documentation Only) ===")
    print("The following code demonstrates how this would be implemented in a Databricks environment:")
    
    spark_code = """
    # Create Spark session
    spark = SparkSession.builder \\
        .appName("Online Retail RFM Analysis") \\
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
        .getOrCreate()
    
    # Load data
    df = spark.read.format("delta").load("/mnt/data/processed/online_retail")
    
    # Calculate max date for recency
    max_date = df.select(spark_max("InvoiceDate")).collect()[0][0]
    
    # Calculate RFM metrics
    rfm = df.filter(col("CustomerID").isNotNull()) \\
        .groupBy("CustomerID") \\
        .agg(
            datediff(lit(max_date), spark_max("InvoiceDate")).alias("Recency"),
            count("InvoiceNo").alias("Frequency"),
            spark_sum("TotalAmount").alias("Monetary")
        )
    
    # Create quartiles using Spark SQL
    rfm.createOrReplaceTempView("rfm_data")
    
    rfm_quartiles = spark.sql('''
    SELECT 
        CustomerID, 
        Recency, 
        Frequency, 
        Monetary,
        NTILE(4) OVER (ORDER BY Recency DESC) as R_Quartile,
        NTILE(4) OVER (ORDER BY Frequency ASC) as F_Quartile,
        NTILE(4) OVER (ORDER BY Monetary ASC) as M_Quartile
    FROM rfm_data
    ''')
    
    # Calculate RFM Score
    rfm_with_scores = rfm_quartiles.withColumn(
        "RFM_Score", 
        concat(col("R_Quartile"), col("F_Quartile"), col("M_Quartile"))
    )
    
    # Create segments using when-otherwise
    rfm_with_segments = rfm_with_scores.withColumn(
        "RFM_Segment",
        when(col("RFM_Score").isin("444", "443", "434", "344"), "Champions")
        .when(col("RFM_Score").isin("433", "434", "343", "344", "334"), "Loyal")
        # ... other segments ...
        .otherwise("Unknown")
    )
    
    # Save results to Delta Lake
    rfm_with_segments.write.format("delta").mode("overwrite").save("/mnt/data/curated/rfm_segments")
    """
    
    print(spark_code)

# Main execution function
def main():
    """
    Main execution function
    """
    # Load processed data
    df = load_processed_data(PROCESSED_CSV_PATH)
    
    # Calculate RFM metrics
    rfm_df = calculate_rfm_metrics(df)
    
    # Create RFM segments
    rfm_segments = create_rfm_segments(rfm_df)
    
    # Analyze segments
    segment_analysis = analyze_segments(rfm_segments, df)
    
    # Visualize RFM segments
    visualize_rfm_segments(rfm_segments, segment_analysis)
    
    # Save transformed data
    save_transformed_data(rfm_segments, segment_analysis, TRANSFORMED_DATA_PATH, RFM_SEGMENTS_PATH)
    
    # Show Spark code example (for documentation)
    spark_code_example()
    
    print("\nData transformation pipeline completed successfully!")
    print(f"Transformed data saved to: {TRANSFORMED_DATA_PATH} and {RFM_SEGMENTS_PATH}")
    print(f"Visualizations saved to: /home/ubuntu/big_data_project/images/")

# Execute the pipeline
if __name__ == "__main__":
    main()
