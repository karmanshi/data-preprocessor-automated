import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_string_dtype

## Step 1:- Taking file from user and check type of file to read the data

file_path = input("Enter the file path: ")

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print("Error Occurred: ", str(e))


# Step 2 :- Data Profiling
print("Rows and Columns DataFrame Contains: \t", df.shape)
print("Missing Value that Dataframe contains:\t", df.isnull().sum().sum())
print("Duplicate Value that DataFrame contains:  \t", df.duplicated().sum())
print("DataType that DataFrame contains: \n", df.dtypes)
print()


# Step 3 :- Numerical Analysis
numeric_cols = df.select_dtypes(include="number").columns.tolist()
print("Numeric Columns that DataFrame Contain", numeric_cols)

column_name = input("Enter the Column Name for numerical analysis: ")
column_extract = column_name.split(",")
for col_name in column_extract:
    col_name = col_name.strip()
    print(f"Numerical Analysis for: {col_name}")
    print("\tMean: ", df[col_name].mean())
    print("\tMedian: ", df[col_name].median())
    print("\tMinimum: ", df[col_name].min())
    print("\tMaximum: ", df[col_name].max())
    print("\tStandard Deviation: ", df[col_name].std())
    print("\tSkewness: ", df[col_name].skew())
    print("\tQuartiles: \n", df[col_name].quantile([0.25, 0.5, 0.75]))
    print()


# Visualizations
from llm import llm_call

resp = llm_call(df.dtypes)
eligible_plots = list(resp.keys())
print("Eligible Plots: ", list(resp.keys()))

if "Boxplot" in eligible_plots:
    print("Preparing Boxplot")
    part = resp["Boxplot"]
    print("Part", part)
    x_axis_cols = part["x-axis"]
    y_axis_cols = part["y-axis"]
    title = part["title"]
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=x_axis_cols[0], y=y_axis_cols[0], palette="Set2")
    plt.title(title)
    plt.show()


if "Histogram" in eligible_plots:
    print("Preparing Histogram")
    part = resp["Histogram"]
    print("Part", part)
    x_axis_cols = part["x-axis"]
    y_axis_cols = part["y-axis"]
    title = part["title"]
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=x_axis_cols[0], kde=True, bins=15, color="skyblue")
    plt.title(title)
    plt.show()

if "Barchart" in eligible_plots:
    print("Preparing Barchart")
    part = resp["Barchart"]
    print("Part", part)
    x_axis_cols = part["x-axis"]
    y_axis_cols = part["y-axis"]
    title = part["title"]
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=df, x=x_axis_cols[0], y=y_axis_cols[0], errorbar=None, palette="Pastel1"
    )
    plt.title(title)
    plt.show()


if "Scatter Plot" in eligible_plots:
    print("Preparing Scatter Plot")
    part = resp["Scatter Plot"]
    print("Part", part)
    x_axis_cols = part["x-axis"]
    y_axis_cols = part["y-axis"]
    title = part["title"]
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df, x=x_axis_cols[0], y=y_axis_cols[0], color="purple", alpha=0.7
    )
    plt.title(title)
    plt.show()

if "Correlation Heatmap" in eligible_plots:
    print("Preparing Correlation Heatmap")
    part = resp["Correlation Heatmap"]
    print("Part", part)
    x_axis_cols = part["x-axis"]
    y_axis_cols = part["y-axis"]
    title = part["title"]
    plt.figure(figsize=(6, 4))

    numerical_df = df.select_dtypes(include=["number"])
    corr_matrix = numerical_df.corr()

    sns.heatmap(
        data=corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title(title)
    plt.show()


# Preprocessing
# Missing Values
for col_name, col_data in df.items():
    if df[col_name].isnull().sum()>0:
        if is_numeric_dtype(df[col_name]):
            df[col_name] = df[col_name].fillna(df[col_name].median())
        if is_string_dtype(df[col_name]):  
            df[col_name] = df[col_name].fillna(df[col_name].mode()[0]) 



# DataType
df = df.convert_dtypes()
print(df.dtypes)

numeric_col = df.select_dtypes(include = "number").columns.tolist()
find_outliers = input(    "Enter the method through which you want to detect outliers ['IQR', 'Z-Score']: "
)
outliers_columns = {}
for col in numeric_col:
    if find_outliers == "IQR":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3-Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (df[col] < lower) | (df[col] > upper)
    elif find_outliers == "Z-Score":
        z_threshold = 3
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or pd.isna(std):
            continue
        z_score = (df[col]-mean)/std
        outlier_mask = z_score.abs()>z_threshold
        lower = mean - z_threshold*std
        upper = mean + z_threshold*std

    else:
        print("Invalid Method")
        break

    # Count outliers
    outlier_count = outlier_mask.sum()
    if outlier_count>0:
        outliers_columns[col]={
            "mask": outlier_mask,
            "lower": lower,
            "upper": upper,
            "count": outlier_count
        }
        print(f"{col}: {outlier_count} outliers found")
    else:
            print(f"{col}: No outliers")


if outliers_columns:
    print("Column contain outliers: ")
    for col,info in outliers_columns.items():
        print(f"{col}->{info['count']}outliers")
    action = input("Enter Action ['Remove','Cap']")

    if action == "Remove":
        for col, info in outliers_columns.items():
            df = df.loc[~info["mask"]]
            print("Outliers removed successfully.")

    elif action == "Cap":
        for index, rows in df.iterrows():
            rows[col]=int(info['lower'])
        # for col, info in outliers_columns.items():
        #         df[col] = df[col].clip(
        #             lower=info["lower"],
        #             upper=info["upper"]
        #         )
        print("Outliers capped successfully.")


    else:
            print("Invalid action.")

else:
    print("No outliers found.")

df.to_csv("./files/output.csv",index= False)
