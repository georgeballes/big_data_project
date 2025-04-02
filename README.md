# Big Data Analytics Project with Databricks and Apache Spark

This repository contains a comprehensive Big Data Analytics project that showcases proficiency with Databricks, Apache Spark, and cloud data workflows. The project implements a customer segmentation solution for an online retail business using RFM (Recency, Frequency, Monetary) analysis and machine learning.

## Project Overview

The project demonstrates a complete data analytics workflow:

1. **Data Ingestion**: Loading and validating the Online Retail dataset
2. **Data Transformation**: Cleaning, transforming, and calculating RFM metrics
3. **Machine Learning**: Implementing K-means clustering with MLflow for customer segmentation
4. **Visualization**: Creating interactive dashboards to present insights
5. **Documentation**: Providing architecture diagrams and technical explanations

## Repository Structure

```
big_data_project/
├── data/
│   ├── raw/             # Raw data files
│   ├── processed/       # Cleaned and transformed data
│   └── curated/         # Business-ready datasets (RFM, segments)
├── notebooks/
│   ├── 01_data_ingestion.py       # Data ingestion pipeline
│   ├── 02_data_transformation.py  # RFM analysis and transformation
│   ├── 03_ml_pipeline.py          # ML pipeline with MLflow
│   └── 04_dashboard.py            # Dashboard and visualizations
├── models/              # Saved ML models and artifacts
├── images/              # Generated visualizations
├── docs/
│   ├── architecture_diagram.md    # Solution architecture diagram
│   ├── technical_blog.md          # Technical explanation and scaling
│   ├── dashboard.html             # Interactive HTML dashboard
│   ├── dataset_selection.md       # Dataset evaluation and selection
│   └── data_architecture.md       # Data architecture design
└── README.md            # Project overview and instructions
```

## Key Features

- **Scalable Architecture**: Designed to handle enterprise-level data volumes
- **MLflow Integration**: Experiment tracking, model versioning, and deployment
- **Interactive Dashboard**: Business-friendly visualizations and insights
- **Comprehensive Documentation**: Architecture diagrams and technical explanations

## Getting Started

### Prerequisites

- Python 3.8+
- Apache Spark 3.3.0+
- MLflow
- Pandas, NumPy, Matplotlib, Seaborn, Plotly
- Scikit-learn

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas pyarrow matplotlib seaborn scikit-learn mlflow delta-spark pyspark==3.3.0 plotly
   ```

### Running the Project

1. **Data Ingestion**:
   ```
   python notebooks/01_data_ingestion_pandas.py
   ```

2. **Data Transformation**:
   ```
   python notebooks/02_data_transformation.py
   ```

3. **ML Pipeline**:
   ```
   python notebooks/03_ml_pipeline.py
   ```

4. **Dashboard Creation**:
   ```
   python notebooks/04_dashboard.py
   ```

5. **View the Dashboard**:
   Open `docs/dashboard.html` in a web browser

## Results and Insights

The analysis identified three distinct customer segments:

1. **Champions (20.1%)**: Recent purchasers with high frequency and monetary value
2. **Regular Customers (56.4%)**: Customers with moderate recency, frequency, and monetary values
3. **Hibernating Customers (23.5%)**: Customers who haven't purchased in a long time

These segments provide the foundation for targeted marketing strategies:

- **For Champions**: Implement VIP loyalty programs and referral incentives
- **For Regular Customers**: Develop personalized recommendations and bundle offers
- **For Hibernating Customers**: Launch re-engagement campaigns with special discounts

## Scaling to Production

In a production environment, this solution would be deployed on Databricks with:

- Delta Lake for ACID transactions and time travel
- Auto-scaling clusters for cost optimization
- Streaming ingestion for real-time updates
- MLflow Model Registry for deployment management
- Monitoring for data quality and model drift

For more details, see the [Technical Blog](docs/technical_blog.md).

## Architecture

The solution follows a modern data lakehouse architecture:

![Architecture Diagram](docs/architecture_diagram.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Online Retail dataset
- Databricks and Apache Spark communities for their excellent documentation
