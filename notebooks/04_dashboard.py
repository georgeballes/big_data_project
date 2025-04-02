#!/usr/bin/env python
# coding: utf-8

# # Online Retail Dashboard and Visualizations
# 
# This notebook creates a comprehensive dashboard and visualizations for the Online Retail customer segmentation project.
# It performs the following steps:
# 1. Load the processed data, RFM data, and cluster analysis
# 2. Create interactive visualizations
# 3. Generate a dashboard HTML file
# 4. Provide business insights and recommendations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import joblib
from sklearn.decomposition import PCA
import base64
from IPython.display import HTML

# Set paths
PROCESSED_DATA_PATH = "/home/ubuntu/big_data_project/data/processed/online_retail.csv"
RFM_DATA_PATH = "/home/ubuntu/big_data_project/data/curated/rfm_data.csv"
CLUSTER_ANALYSIS_PATH = "/home/ubuntu/big_data_project/data/curated/cluster_analysis.csv"
MODEL_PATH = "/home/ubuntu/big_data_project/models/kmeans_model_3_clusters.pkl"
SCALER_PATH = "/home/ubuntu/big_data_project/models/scaler.pkl"
DASHBOARD_PATH = "/home/ubuntu/big_data_project/docs/dashboard.html"
IMAGES_DIR = "/home/ubuntu/big_data_project/images"

# Ensure directories exist
os.makedirs(os.path.dirname(DASHBOARD_PATH), exist_ok=True)

print("Starting Online Retail Dashboard Creation...")

# Function to load all required data
def load_all_data():
    """
    Load all required data for dashboard creation
    """
    print("Loading all required data...")
    
    # Load processed data
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    processed_df['InvoiceDate'] = pd.to_datetime(processed_df['InvoiceDate'])
    
    # Load RFM data
    rfm_df = pd.read_csv(RFM_DATA_PATH)
    
    # Load cluster analysis
    cluster_analysis = pd.read_csv(CLUSTER_ANALYSIS_PATH)
    
    # Load model and scaler
    kmeans_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Processed data loaded with {processed_df.shape[0]} rows and {processed_df.shape[1]} columns")
    print(f"RFM data loaded with {rfm_df.shape[0]} rows and {rfm_df.shape[1]} columns")
    print(f"Cluster analysis loaded with {cluster_analysis.shape[0]} rows")
    
    return processed_df, rfm_df, cluster_analysis, kmeans_model, scaler

# Function to create sales trend visualizations
def create_sales_trend_visualizations(processed_df):
    """
    Create sales trend visualizations
    """
    print("\n=== Creating Sales Trend Visualizations ===")
    
    # Aggregate data by date
    daily_sales = processed_df.groupby(processed_df['InvoiceDate'].dt.date).agg({
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalAmount': 'sum'
    }).reset_index()
    
    daily_sales.columns = ['Date', 'Orders', 'Items', 'Revenue']
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    
    # Monthly trend
    monthly_sales = processed_df.groupby(processed_df['InvoiceDate'].dt.to_period('M')).agg({
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'CustomerID': pd.Series.nunique
    }).reset_index()
    
    monthly_sales['InvoiceDate'] = monthly_sales['InvoiceDate'].astype(str)
    monthly_sales.columns = ['Month', 'Orders', 'Items', 'Revenue', 'Unique_Customers']
    
    # Create plotly figure for daily sales
    fig_daily = px.line(daily_sales, x='Date', y='Revenue', 
                        title='Daily Revenue Trend',
                        labels={'Revenue': 'Revenue ($)', 'Date': 'Date'})
    
    fig_daily.update_layout(
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        template='plotly_white'
    )
    
    # Create plotly figure for monthly sales
    fig_monthly = px.bar(monthly_sales, x='Month', y='Revenue',
                         title='Monthly Revenue Trend',
                         labels={'Revenue': 'Revenue ($)', 'Month': 'Month'})
    
    fig_monthly.update_layout(
        xaxis_title='Month',
        yaxis_title='Revenue ($)',
        template='plotly_white'
    )
    
    # Create plotly figure for monthly orders and customers
    fig_monthly_metrics = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_monthly_metrics.add_trace(
        go.Bar(x=monthly_sales['Month'], y=monthly_sales['Orders'], name='Orders'),
        secondary_y=False
    )
    
    fig_monthly_metrics.add_trace(
        go.Scatter(x=monthly_sales['Month'], y=monthly_sales['Unique_Customers'], 
                  name='Unique Customers', mode='lines+markers'),
        secondary_y=True
    )
    
    fig_monthly_metrics.update_layout(
        title='Monthly Orders and Unique Customers',
        template='plotly_white'
    )
    
    fig_monthly_metrics.update_xaxes(title_text='Month')
    fig_monthly_metrics.update_yaxes(title_text='Number of Orders', secondary_y=False)
    fig_monthly_metrics.update_yaxes(title_text='Number of Unique Customers', secondary_y=True)
    
    # Save figures
    pio.write_html(fig_daily, f"{IMAGES_DIR}/daily_revenue_trend.html")
    pio.write_html(fig_monthly, f"{IMAGES_DIR}/monthly_revenue_trend.html")
    pio.write_html(fig_monthly_metrics, f"{IMAGES_DIR}/monthly_orders_customers.html")
    
    print("Sales trend visualizations created and saved")
    
    return fig_daily, fig_monthly, fig_monthly_metrics, monthly_sales

# Function to create geographic visualizations
def create_geographic_visualizations(processed_df):
    """
    Create geographic visualizations
    """
    print("\n=== Creating Geographic Visualizations ===")
    
    # Aggregate data by country
    country_sales = processed_df.groupby('Country').agg({
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'CustomerID': pd.Series.nunique
    }).reset_index()
    
    country_sales.columns = ['Country', 'Orders', 'Items', 'Revenue', 'Customers']
    country_sales = country_sales.sort_values('Revenue', ascending=False)
    
    # Calculate average order value
    country_sales['Avg_Order_Value'] = country_sales['Revenue'] / country_sales['Orders']
    
    # Create plotly figure for country revenue
    fig_country_revenue = px.bar(country_sales.head(10), x='Country', y='Revenue',
                                title='Top 10 Countries by Revenue',
                                labels={'Revenue': 'Revenue ($)', 'Country': 'Country'})
    
    fig_country_revenue.update_layout(
        xaxis_title='Country',
        yaxis_title='Revenue ($)',
        template='plotly_white'
    )
    
    # Create plotly figure for country customers
    fig_country_customers = px.bar(country_sales.head(10), x='Country', y='Customers',
                                  title='Top 10 Countries by Number of Customers',
                                  labels={'Customers': 'Number of Customers', 'Country': 'Country'})
    
    fig_country_customers.update_layout(
        xaxis_title='Country',
        yaxis_title='Number of Customers',
        template='plotly_white'
    )
    
    # Create plotly figure for average order value by country
    fig_country_aov = px.bar(country_sales.head(10), x='Country', y='Avg_Order_Value',
                            title='Average Order Value by Country',
                            labels={'Avg_Order_Value': 'Average Order Value ($)', 'Country': 'Country'})
    
    fig_country_aov.update_layout(
        xaxis_title='Country',
        yaxis_title='Average Order Value ($)',
        template='plotly_white'
    )
    
    # Save figures
    pio.write_html(fig_country_revenue, f"{IMAGES_DIR}/country_revenue.html")
    pio.write_html(fig_country_customers, f"{IMAGES_DIR}/country_customers.html")
    pio.write_html(fig_country_aov, f"{IMAGES_DIR}/country_aov.html")
    
    print("Geographic visualizations created and saved")
    
    return fig_country_revenue, fig_country_customers, fig_country_aov, country_sales

# Function to create RFM and cluster visualizations
def create_rfm_cluster_visualizations(rfm_df, cluster_analysis):
    """
    Create RFM and cluster visualizations
    """
    print("\n=== Creating RFM and Cluster Visualizations ===")
    
    # Add cluster names based on characteristics
    cluster_analysis['Cluster_Name'] = [
        'Hibernating Customers',
        'Regular Customers',
        'Champions'
    ]
    
    # Create plotly figure for cluster characteristics
    fig_cluster_chars = go.Figure()
    
    # Normalize values for better visualization
    max_recency = cluster_analysis['Avg_Recency'].max()
    max_frequency = cluster_analysis['Avg_Frequency'].max()
    max_monetary = cluster_analysis['Avg_Monetary'].max()
    
    cluster_analysis['Norm_Recency'] = cluster_analysis['Avg_Recency'] / max_recency
    cluster_analysis['Norm_Frequency'] = cluster_analysis['Avg_Frequency'] / max_frequency
    cluster_analysis['Norm_Monetary'] = cluster_analysis['Avg_Monetary'] / max_monetary
    
    # Add traces for each metric
    for cluster, name in zip(cluster_analysis['Cluster'], cluster_analysis['Cluster_Name']):
        fig_cluster_chars.add_trace(go.Scatterpolar(
            r=[
                cluster_analysis.loc[cluster_analysis['Cluster'] == cluster, 'Norm_Recency'].values[0],
                cluster_analysis.loc[cluster_analysis['Cluster'] == cluster, 'Norm_Frequency'].values[0],
                cluster_analysis.loc[cluster_analysis['Cluster'] == cluster, 'Norm_Monetary'].values[0]
            ],
            theta=['Recency', 'Frequency', 'Monetary'],
            fill='toself',
            name=f"Cluster {cluster}: {name}"
        ))
    
    fig_cluster_chars.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Cluster Characteristics (Normalized)',
        template='plotly_white'
    )
    
    # Create plotly figure for cluster size
    fig_cluster_size = px.pie(cluster_analysis, values='Customer_Count', names='Cluster_Name',
                             title='Customer Segment Distribution')
    
    fig_cluster_size.update_layout(template='plotly_white')
    
    # Create plotly figure for RFM distribution
    fig_rfm_dist = make_subplots(rows=1, cols=3, 
                                subplot_titles=('Recency Distribution', 'Frequency Distribution', 'Monetary Distribution'))
    
    fig_rfm_dist.add_trace(
        go.Histogram(x=rfm_df['Recency'], name='Recency'),
        row=1, col=1
    )
    
    fig_rfm_dist.add_trace(
        go.Histogram(x=rfm_df['Frequency'], name='Frequency'),
        row=1, col=2
    )
    
    fig_rfm_dist.add_trace(
        go.Histogram(x=rfm_df['Monetary'], name='Monetary'),
        row=1, col=3
    )
    
    fig_rfm_dist.update_layout(
        title='RFM Metrics Distribution',
        template='plotly_white',
        showlegend=False
    )
    
    # Save figures
    pio.write_html(fig_cluster_chars, f"{IMAGES_DIR}/cluster_characteristics.html")
    pio.write_html(fig_cluster_size, f"{IMAGES_DIR}/cluster_size.html")
    pio.write_html(fig_rfm_dist, f"{IMAGES_DIR}/rfm_distribution.html")
    
    print("RFM and cluster visualizations created and saved")
    
    return fig_cluster_chars, fig_cluster_size, fig_rfm_dist, cluster_analysis

# Function to create business insights
def create_business_insights(processed_df, rfm_df, cluster_analysis):
    """
    Create business insights
    """
    print("\n=== Creating Business Insights ===")
    
    insights = []
    
    # Customer segmentation insights
    insights.append({
        'title': 'Customer Segmentation Overview',
        'content': f"""
        <p>Our analysis has identified <strong>{len(cluster_analysis)}</strong> distinct customer segments:</p>
        <ul>
            <li><strong>Champions ({cluster_analysis.loc[cluster_analysis['Cluster'] == 2, 'Cluster_Size_Percentage'].values[0]:.1f}% of customers)</strong>: 
                Recent purchasers with high frequency and monetary value. These are your best customers who bought recently, 
                buy often and spend the most. Focus on retention and loyalty programs for this segment.</li>
            
            <li><strong>Regular Customers ({cluster_analysis.loc[cluster_analysis['Cluster'] == 1, 'Cluster_Size_Percentage'].values[0]:.1f}% of customers)</strong>: 
                Customers with moderate recency, frequency, and monetary values. These customers are regular shoppers 
                but have potential to increase their spending. Target them with personalized offers and upselling.</li>
            
            <li><strong>Hibernating Customers ({cluster_analysis.loc[cluster_analysis['Cluster'] == 0, 'Cluster_Size_Percentage'].values[0]:.1f}% of customers)</strong>: 
                Customers who haven't purchased in a long time. These customers are at risk of churning or have already churned. 
                Re-engagement campaigns and win-back offers should be directed at this segment.</li>
        </ul>
        """
    })
    
    # Revenue insights
    total_revenue = processed_df['TotalAmount'].sum()
    avg_order_value = total_revenue / processed_df['InvoiceNo'].nunique()
    total_customers = processed_df['CustomerID'].nunique()
    customer_lifetime_value = total_revenue / total_customers
    
    insights.append({
        'title': 'Revenue and Customer Value Insights',
        'content': f"""
        <p>Key business metrics from our analysis:</p>
        <ul>
            <li><strong>Total Revenue</strong>: ${total_revenue:,.2f}</li>
            <li><strong>Average Order Value</strong>: ${avg_order_value:,.2f}</li>
            <li><strong>Customer Lifetime Value</strong>: ${customer_lifetime_value:,.2f}</li>
            <li><strong>Top Country by Revenue</strong>: {processed_df.groupby('Country')['TotalAmount'].sum().idxmax()}</li>
        </ul>
        <p>The Champions segment, while only {cluster_analysis.loc[cluster_analysis['Cluster'] == 2, 'Cluster_Size_Percentage'].values[0]:.1f}% of the customer base, 
        likely contributes a disproportionately high percentage of total revenue. This highlights the importance of retention strategies for high-value customers.</p>
        """
    })
    
    # Recommendations
    insights.append({
        'title': 'Strategic Recommendations',
        'content': f"""
        <p>Based on our customer segmentation analysis, we recommend the following strategies:</p>
        <ol>
            <li><strong>For Champions:</strong>
                <ul>
                    <li>Implement a VIP loyalty program with exclusive benefits</li>
                    <li>Provide early access to new products</li>
                    <li>Create a referral program to leverage their network</li>
                    <li>Gather feedback to improve products and services</li>
                </ul>
            </li>
            <li><strong>For Regular Customers:</strong>
                <ul>
                    <li>Develop personalized product recommendations based on purchase history</li>
                    <li>Implement targeted email campaigns with relevant offers</li>
                    <li>Create bundle offers to increase average order value</li>
                    <li>Encourage more frequent purchases with limited-time promotions</li>
                </ul>
            </li>
            <li><strong>For Hibernating Customers:</strong>
                <ul>
                    <li>Launch re-engagement campaigns with special "we miss you" discounts</li>
                    <li>Conduct surveys to understand reasons for inactivity</li>
                    <li>Showcase new products or improvements since their last purchase</li>
                    <li>Consider a win-back program with incentives for returning</li>
                </ul>
            </li>
        </ol>
        <p>Additionally, consider implementing the following general strategies:</p>
        <ul>
            <li>Develop a customer journey map for each segment to identify touchpoints for improvement</li>
            <li>Set up automated marketing workflows based on RFM segments</li>
            <li>Regularly update the segmentation model with new transaction data</li>
            <li>Monitor segment migration to measure the effectiveness of marketing strategies</li>
        </ul>
        """
    })
    
    print(f"Created {len(insights)} business insights")
    
    return insights

# Function to create HTML dashboard
def create_html_dashboard(figures, insights, cluster_analysis):
    """
    Create HTML dashboard
    """
    print("\n=== Creating HTML Dashboard ===")
    
    # Unpack figures
    fig_daily, fig_monthly, fig_monthly_metrics = figures['sales_trends']
    fig_country_revenue, fig_country_customers, fig_country_aov = figures['geographic']
    fig_cluster_chars, fig_cluster_size, fig_rfm_dist = figures['rfm_clusters']
    
    # Convert plotly figures to HTML
    daily_revenue_html = fig_daily.to_html(full_html=False, include_plotlyjs='cdn')
    monthly_revenue_html = fig_monthly.to_html(full_html=False, include_plotlyjs=False)
    monthly_metrics_html = fig_monthly_metrics.to_html(full_html=False, include_plotlyjs=False)
    
    country_revenue_html = fig_country_revenue.to_html(full_html=False, include_plotlyjs=False)
    country_customers_html = fig_country_customers.to_html(full_html=False, include_plotlyjs=False)
    country_aov_html = fig_country_aov.to_html(full_html=False, include_plotlyjs=False)
    
    cluster_chars_html = fig_cluster_chars.to_html(full_html=False, include_plotlyjs=False)
    cluster_size_html = fig_cluster_size.to_html(full_html=False, include_plotlyjs=False)
    rfm_dist_html = fig_rfm_dist.to_html(full_html=False, include_plotlyjs=False)
    
    # Create cluster table HTML
    cluster_table = cluster_analysis[['Cluster', 'Cluster_Name', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Customer_Count', 'Cluster_Size_Percentage']]
    cluster_table = cluster_table.sort_values('Cluster')
    
    # Format the table
    cluster_table['Avg_Recency'] = cluster_table['Avg_Recency'].round(1)
    cluster_table['Avg_Frequency'] = cluster_table['Avg_Frequency'].round(1)
    cluster_table['Avg_Monetary'] = cluster_table['Avg_Monetary'].round(2)
    cluster_table['Cluster_Size_Percentage'] = cluster_table['Cluster_Size_Percentage'].round(1)
    
    cluster_table_html = cluster_table.to_html(index=False, classes='table table-striped table-hover')
    
    # Create insights HTML
    insights_html = ""
    for insight in insights:
        insights_html += f"""
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="mb-0">{insight['title']}</h5>
            </div>
            <div class="card-body">
                {insight['content']}
            </div>
        </div>
        """
    
    # Create HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Online Retail Customer Segmentation Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding-top: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .nav-pills .nav-link.active {{ background-color: #6c757d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header class="mb-4">
                <div class="row align-items-center">
                    <div class="col">
                        <h1>Online Retail Customer Segmentation</h1>
                        <p class="lead">A Big Data Analytics Project with Databricks, Apache Spark, and MLflow</p>
                    </div>
                </div>
                <hr>
            </header>
            
            <div class="row">
                <div class="col-md-3">
                    <div class="list-group" id="dashboard-tabs" role="tablist">
                        <a class="list-group-item list-group-item-action active" id="overview-tab" data-bs-toggle="list" href="#overview" role="tab">Overview</a>
                        <a class="list-group-item list-group-item-action" id="sales-trends-tab" data-bs-toggle="list" href="#sales-trends" role="tab">Sales Trends</a>
                        <a class="list-group-item list-group-item-action" id="geographic-tab" data-bs-toggle="list" href="#geographic" role="tab">Geographic Analysis</a>
                        <a class="list-group-item list-group-item-action" id="customer-segments-tab" data-bs-toggle="list" href="#customer-segments" role="tab">Customer Segments</a>
                        <a class="list-group-item list-group-item-action" id="insights-tab" data-bs-toggle="list" href="#insights" role="tab">Business Insights</a>
                    </div>
                </div>
                
                <div class="col-md-9">
                    <div class="tab-content">
                        <!-- Overview Tab -->
                        <div class="tab-pane fade show active" id="overview" role="tabpanel">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Project Overview</h3>
                                </div>
                                <div class="card-body">
                                    <p>This dashboard presents the results of a Big Data Analytics project that demonstrates proficiency with Databricks, Apache Spark, and cloud data workflows. The project uses the Online Retail dataset to perform customer segmentation using RFM (Recency, Frequency, Monetary) analysis and machine learning.</p>
                                    
                                    <h4>Project Components:</h4>
                                    <ul>
                                        <li>Data ingestion from a public dataset into a data lake structure</li>
                                        <li>Data cleaning and transformation using Apache Spark</li>
                                        <li>RFM analysis for customer segmentation</li>
                                        <li>Machine learning pipeline with MLflow for clustering</li>
                                        <li>Interactive visualizations and business insights</li>
                                    </ul>
                                    
                                    <h4>Dataset Information:</h4>
                                    <p>The Online Retail dataset contains all transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based online retail company. The company mainly sells unique all-occasion gifts, and many customers are wholesalers.</p>
                                    
                                    <div class="row mt-4">
                                        <div class="col-md-6">
                                            <div class="card">
                                                <div class="card-body text-center">
                                                    <h5 class="card-title">Customer Segments</h5>
                                                    {cluster_size_html}
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="card">
                                                <div class="card-body">
                                                    <h5 class="card-title">Segment Characteristics</h5>
                                                    {cluster_table_html}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Sales Trends Tab -->
                        <div class="tab-pane fade" id="sales-trends" role="tabpanel">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Sales Trends</h3>
                                </div>
                                <div class="card-body">
                                    <div class="section">
                                        <h4>Daily Revenue Trend</h4>
                                        {daily_revenue_html}
                                    </div>
                                    
                                    <div class="section">
                                        <h4>Monthly Revenue Trend</h4>
                                        {monthly_revenue_html}
                                    </div>
                                    
                                    <div class="section">
                                        <h4>Monthly Orders and Customers</h4>
                                        {monthly_metrics_html}
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Geographic Analysis Tab -->
                        <div class="tab-pane fade" id="geographic" role="tabpanel">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Geographic Analysis</h3>
                                </div>
                                <div class="card-body">
                                    <div class="section">
                                        <h4>Top Countries by Revenue</h4>
                                        {country_revenue_html}
                                    </div>
                                    
                                    <div class="section">
                                        <h4>Top Countries by Number of Customers</h4>
                                        {country_customers_html}
                                    </div>
                                    
                                    <div class="section">
                                        <h4>Average Order Value by Country</h4>
                                        {country_aov_html}
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Customer Segments Tab -->
                        <div class="tab-pane fade" id="customer-segments" role="tabpanel">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Customer Segments</h3>
                                </div>
                                <div class="card-body">
                                    <div class="section">
                                        <h4>Cluster Characteristics</h4>
                                        {cluster_chars_html}
                                    </div>
                                    
                                    <div class="section">
                                        <h4>RFM Metrics Distribution</h4>
                                        {rfm_dist_html}
                                    </div>
                                    
                                    <div class="section">
                                        <h4>Segment Details</h4>
                                        <div class="table-responsive">
                                            {cluster_table_html}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Business Insights Tab -->
                        <div class="tab-pane fade" id="insights" role="tabpanel">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Business Insights and Recommendations</h3>
                                </div>
                                <div class="card-body">
                                    {insights_html}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <footer class="mt-5 mb-3 text-center text-muted">
                <p>Online Retail Customer Segmentation Dashboard | Created with Python, Plotly, and Bootstrap</p>
                <p>Data Source: UCI Machine Learning Repository</p>
            </footer>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(DASHBOARD_PATH, 'w') as f:
        f.write(html_content)
    
    print(f"HTML dashboard created and saved to {DASHBOARD_PATH}")
    
    return html_content

# Main execution function
def main():
    """
    Main execution function
    """
    # Load all required data
    processed_df, rfm_df, cluster_analysis, kmeans_model, scaler = load_all_data()
    
    # Create sales trend visualizations
    sales_trend_figures = create_sales_trend_visualizations(processed_df)
    
    # Create geographic visualizations
    geographic_figures = create_geographic_visualizations(processed_df)
    
    # Create RFM and cluster visualizations
    rfm_cluster_figures = create_rfm_cluster_visualizations(rfm_df, cluster_analysis)
    
    # Create business insights
    insights = create_business_insights(processed_df, rfm_df, cluster_analysis)
    
    # Organize figures
    figures = {
        'sales_trends': sales_trend_figures[:3],
        'geographic': geographic_figures[:3],
        'rfm_clusters': rfm_cluster_figures[:3]
    }
    
    # Create HTML dashboard
    dashboard_html = create_html_dashboard(figures, insights, cluster_analysis)
    
    print("\nDashboard creation completed successfully!")
    print(f"Dashboard saved to: {DASHBOARD_PATH}")
    print(f"Visualizations saved to: {IMAGES_DIR}")

# Execute the dashboard creation
if __name__ == "__main__":
    main()
