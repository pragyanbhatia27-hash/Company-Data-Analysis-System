import pandas as pd
import numpy as np

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("company_data_600plus.csv")

print("\nOriginal Dataset")
print(df)

# ==========================
# Basic Information
# ==========================

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ==========================
# New Calculated Columns
# ==========================

df["Profit"] = (
    df["Revenue_Generated"]
    - df["Salary"]
)

df["Revenue_Per_Hour"] = (
    df["Revenue_Generated"]
    / df["Working_Hours"]
)

df["Productivity_Score"] = (
    df["Projects_Completed"]
    * df["Customer_Rating"]
)

# ==========================
# Performance Level
# ==========================

conditions = [
    df["Revenue_Generated"] >= 200000,

    (df["Revenue_Generated"] >= 150000)
    &
    (df["Revenue_Generated"] < 200000),

    df["Revenue_Generated"] < 150000
]

categories = [
    "Excellent",
    "Good",
    "Needs Improvement"
]

df["Performance_Level"] = np.select(
    conditions,
    categories,
    default="Needs Improvement"
)

# ==========================
# Efficiency Score
# ==========================

df["Efficiency_Score"] = (
      df["Revenue_Per_Hour"] * 0.40
    + df["Customer_Rating"] * 20
    + df["Projects_Completed"] * 2
)

# ==========================
# Business Risk
# ==========================

df["Business_Risk"] = np.where(
    df["Efficiency_Score"] < 450,
    "High Risk",
    "Low Risk"
)

# ==========================
# Department Analysis
# ==========================

department_revenue = (
    df.groupby("Department")["Revenue_Generated"].sum()
)

print("\nDepartment Revenue")
print(department_revenue)

department_profit = (
    df.groupby("Department")["Profit"].sum()
)

print("\nDepartment Profit")
print(department_profit)

department_rating = (
    df.groupby("Department")["Customer_Rating"].mean()
)

print("\nAverage Customer Rating")
print(department_rating)

# ==========================
# Top & Bottom Employees
# ==========================

print("\nTop 10 Revenue Employees")
print(
    df.nlargest(
        10,
        "Revenue_Generated"
    )
)

print("\nBottom 10 Revenue Employees")
print(
    df.nsmallest(
        10,
        "Revenue_Generated"
    )
)

# ==========================
# Highest & Lowest Revenue Employee
# ==========================

print("\nHighest Revenue Employee")
print(
    df.loc[
        df["Revenue_Generated"].idxmax()
    ]
)

print("\nLowest Revenue Employee")
print(
    df.loc[
        df["Revenue_Generated"].idxmin()
    ]
)

# ==========================
# NumPy Statistics
# ==========================

print("\nAverage Revenue")
print(
    np.mean(
        df["Revenue_Generated"]
    )
)

print("\nMedian Revenue")
print(
    np.median(
        df["Revenue_Generated"]
    )
)

print("\nMaximum Revenue")
print(
    np.max(
        df["Revenue_Generated"]
    )
)

print("\nMinimum Revenue")
print(
    np.min(
        df["Revenue_Generated"]
    )
)

print("\nVariance")
print(
    np.var(
        df["Revenue_Generated"]
    )
)

print("\nStandard Deviation")
print(
    np.std(
        df["Revenue_Generated"]
    )
)

print("\n25th Percentile")
print(
    np.percentile(
        df["Revenue_Generated"],
        25
    )
)

print("\n50th Percentile")
print(
    np.percentile(
        df["Revenue_Generated"],
        50
    )
)

print("\n75th Percentile")
print(
    np.percentile(
        df["Revenue_Generated"],
        75
    )
)

# ==========================
# Advanced Pandas Functions
# ==========================

df["Revenue_Rank"] = (
    df["Revenue_Generated"]
    .rank(
        ascending=False,
        method="dense"
    )
)

df["Salary_Percentage"] = (
    df["Salary"]
    / df["Revenue_Generated"]
    * 100
)

df["Growth_Index"] = np.log(
    df["Revenue_Generated"]
)

df["Revenue_Quartile"] = pd.qcut(
    df["Revenue_Generated"],
    4,
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

df["Revenue_Category"] = np.where(
    df["Revenue_Generated"]
    >
    df["Revenue_Generated"].mean(),
    "Above Average",
    "Below Average"
)

df["Revenue_Percentile"] = (
    df["Revenue_Generated"]
    .rank(pct=True)
    * 100
)

# ==========================
# Department Statistics
# ==========================

department_statistics = (
    df.groupby("Department").agg({
        "Revenue_Generated": [
            "sum",
            "mean",
            "max",
            "min"
        ],
        "Salary": [
            "sum",
            "mean"
        ],
        "Projects_Completed": "mean",
        "Customer_Rating": "mean"
    })
)

print("\nDepartment Statistics")
print(department_statistics)

# ==========================
# Pivot Table
# ==========================

pivot_table = pd.pivot_table(
    df,
    values="Revenue_Generated",
    index="Department",
    columns="Revenue_Category",
    aggfunc=np.sum,
    fill_value=0
)

print("\nPivot Table")
print(pivot_table)

# ==========================
# Correlation Matrix
# ==========================

correlation_matrix = (
    df[
        [
            "Revenue_Generated",
            "Salary",
            "Working_Hours",
            "Projects_Completed",
            "Customer_Rating",
            "Profit",
            "Efficiency_Score"
        ]
    ]
    .corr()
)

print("\nCorrelation Matrix")
print(correlation_matrix)

# ==========================
# Employee Count
# ==========================

print("\nEmployees in Each Department")
print(
    df["Department"]
    .value_counts()
)

# ==========================
# Sort by Efficiency
# ==========================

sorted_employees = (
    df.sort_values(
        by="Efficiency_Score",
        ascending=False
    )
)

print("\nTop 10 Efficient Employees")
print(
    sorted_employees.head(10)
)

# ==========================
# Final Report
# ==========================

final_report = df[
    [
        "Employee_Name",
        "Department",
        "Revenue_Generated",
        "Salary",
        "Profit",
        "Revenue_Per_Hour",
        "Productivity_Score",
        "Efficiency_Score",
        "Revenue_Rank",
        "Revenue_Percentile",
        "Performance_Level",
        "Business_Risk"
    ]
]

final_report.to_csv(
    "Advanced_Company_Analysis_Report.csv",
    index=False
)

print("\nAdvanced Company Analysis Report Generated Successfully!")