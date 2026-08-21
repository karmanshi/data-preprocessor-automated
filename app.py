import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_string_dtype
from llm import llm_call

def load_data():
    while True:
        file_path = input("\nEnter the file path (CSV): ")
        try:
            df = pd.read_csv(file_path)
            print("File loaded successfully!")
            return df
        except Exception as e:
            print(f"Error Occurred: {e}. Please try again.")

def profile_data(df):
    print("\n--- Data Profiling ---")
    print(f"Rows and Columns: {df.shape}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Values: {df.duplicated().sum()}")
    print("\nData Types:\n", df.dtypes)

def numerical_analysis(df):
    print("\n--- Numerical Analysis ---")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("No numeric columns found.")
        return
    
    print(f"Numeric Columns available: {numeric_cols}")
    column_name = input("Enter the Column Name(s) for analysis (comma separated): ")
    column_extract = [c.strip() for c in column_name.split(",")]
    
    for col_name in column_extract:
        if col_name in df.columns:
            print(f"\nAnalysis for: {col_name}")
            print(f"\tMean: {df[col_name].mean():.2f}")
            print(f"\tMedian: {df[col_name].median():.2f}")
            print(f"\tMinimum: {df[col_name].min():.2f}")
            print(f"\tMaximum: {df[col_name].max():.2f}")
            print(f"\tStd Dev: {df[col_name].std():.2f}")
            print(f"\tSkewness: {df[col_name].skew():.2f}")
            print(f"\tQuartiles:\n{df[col_name].quantile([0.25, 0.5, 0.75])}")
        else:
            print(f"Column '{col_name}' not found in DataFrame.")

def visualize_data(df):
    print("\n--- AI-Powered Visualizations ---")
    try:
        resp = llm_call(df.dtypes)
        eligible_plots = list(resp.keys())
        print("Eligible Plots suggested by AI: ", eligible_plots)
        
        if not eligible_plots:
            print("No suitable plots found for this dataset.")
            return

        for plot_type in eligible_plots:
            part = resp[plot_type]
            x_axis = part["x-axis"][0]
            y_axis = part["y-axis"][0] if "y-axis" in part and part["y-axis"] else None
            title = part["title"]
            
            plt.figure(figsize=(6, 4))
            if plot_type == "Boxplot":
                sns.boxplot(data=df, x=x_axis, y=y_axis, palette="Set2")
            elif plot_type == "Histogram":
                sns.histplot(data=df, x=x_axis, kde=True, bins=15, color="skyblue")
            elif plot_type == "Barchart":
                sns.barplot(data=df, x=x_axis, y=y_axis, errorbar=None, palette="Pastel1")
            elif plot_type == "Scatter Plot":
                sns.scatterplot(data=df, x=x_axis, y=y_axis, color="purple", alpha=0.7)
            elif plot_type == "Correlation Heatmap":
                numerical_df = df.select_dtypes(include=["number"])
                sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            
            plt.title(title)
            plt.show()
    except Exception as e:
        print(f"Visualization Error: {e}")

def handle_missing_values(df):
    print("\n--- Handling Missing Values ---")
    for col_name in df.columns:
        null_count = df[col_name].isnull().sum()
        if null_count > 0:
            print(f"Column '{col_name}' has {null_count} missing values.")
            if is_numeric_dtype(df[col_name]):
                df[col_name] = df[col_name].fillna(df[col_name].median())
                print(f"  -> Filled with Median")
            elif is_string_dtype(df[col_name]):
                df[col_name] = df[col_name].fillna(df[col_name].mode()[0])
                print(f"  -> Filled with Mode")
    return df

def handle_outliers(df):
    print("\n--- Outlier Detection & Handling ---")
    # Use standard float64 for numeric columns to avoid Int64 (nullable) dtype issues with clip
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns for outlier detection.")
        return df

    method = input("Enter method ['IQR', 'Z-Score']: ").strip()
    outliers_info = {}

    for col in numeric_cols:
        # Ensure column is float for calculations and clipping
        df[col] = df[col].astype(float)
        
        if method == "IQR":
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            mask = (df[col] < lower) | (df[col] > upper)
        elif method == "Z-Score":
            mean, std = df[col].mean(), df[col].std()
            if std == 0 or pd.isna(std): continue
            z_score = (df[col] - mean) / std
            mask = z_score.abs() > 3
            lower, upper = mean - 3*std, mean + 3*std
        else:
            print("Invalid method.")
            return df

        count = mask.sum()
        if count > 0:
            outliers_info[col] = {"mask": mask, "lower": lower, "upper": upper, "count": count}
            print(f"{col}: {count} outliers found")

    if outliers_info:
        action = input("Action ['Remove', 'Cap']: ").strip()
        if action == "Remove":
            for col, info in outliers_info.items():
                df = df.loc[~info["mask"]]
            print("Outliers removed.")
        elif action == "Cap":
            for col, info in outliers_info.items():
                df[col] = df[col].clip(lower=info["lower"], upper=info["upper"])
            print("Outliers capped.")
    else:
        print("No outliers found.")
    
    return df

def main():
    df = load_data()
    
    while True:
        print("\n==============================")
        print("  DATA PREPROCESSOR AUTOMATED ")
        print("==============================")
        print("[1] Data Profiling")
        print("[2] Numerical Analysis")
        print("[3] AI Visualizations")
        print("[4] Handle Missing Values")
        print("[5] Handle Outliers")
        print("[6] Export to CSV")
        print("[0] Exit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            profile_data(df)
        elif choice == '2':
            numerical_analysis(df)
        elif choice == '3':
            visualize_data(df)
        elif choice == '4':
            df = handle_missing_values(df)
        elif choice == '5':
            df = handle_outliers(df)
        elif choice == '6':
            output_path = "./files/output.csv"
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        elif choice == '0':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
