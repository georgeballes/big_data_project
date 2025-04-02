# Dataset Selection and Use Case Definition

## Selected Dataset: Online Retail Dataset

After researching various public datasets suitable for a Big Data Analytics project, I've selected the **Online Retail dataset** from the UCI Machine Learning Repository.

### Dataset Details:
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/352/online+retail
- **Size**: 22.6 MB
- **Number of Instances**: 541,909 transactions
- **Number of Features**: 6
- **Time Period**: Transactions between 01/12/2010 and 09/12/2011
- **Format**: Tabular data (CSV)

### Dataset Description:
This is a transactional dataset containing all transactions occurring for a UK-based and registered non-store online retail business. The company mainly sells unique all-occasion gifts, and many customers of the company are wholesalers.

### Dataset Attributes:
1. InvoiceNo: Invoice number (a unique identifier for each transaction)
2. StockCode: Product code
3. Description: Product name/description
4. Quantity: Quantity of items purchased
5. InvoiceDate: Date and time of the transaction
6. UnitPrice: Unit price of the product
7. CustomerID: Customer identifier
8. Country: Country where the customer resides

## Selected Use Case: Customer Segmentation using RFM Analysis

For this Big Data Analytics project, I'll implement a **Customer Segmentation** solution using the RFM (Recency, Frequency, Monetary) model. This is a real-world enterprise use case that demonstrates the power of big data analytics for business intelligence.

### Use Case Description:
Customer segmentation is a critical business intelligence task that helps companies understand their customer base and tailor marketing strategies accordingly. The RFM model is a proven approach that segments customers based on:

- **Recency**: How recently a customer made a purchase
- **Frequency**: How often a customer makes purchases
- **Monetary Value**: How much money a customer spends

### Business Value:
This use case provides significant business value by:
1. Identifying high-value customers for retention programs
2. Finding customers at risk of churn
3. Recognizing potential loyal customers for targeted marketing
4. Optimizing marketing spend by focusing on the most promising segments
5. Improving customer experience through personalized approaches

### Technical Implementation:
The implementation will showcase:
- Data ingestion into a cloud data lake
- Data cleaning and transformation using Apache Spark
- Advanced analytics using Spark MLlib for clustering
- MLflow for experiment tracking and model management
- Interactive visualizations for business insights

This use case aligns perfectly with the project requirements and will effectively demonstrate proficiency with Databricks, Apache Spark, and cloud data workflows in a realistic enterprise scenario.
