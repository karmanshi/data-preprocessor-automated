import os
import requests
import json
from decouple import config


def llm_call(df_info):
    name_of_charts = [
        # "Histogram",
        # "Boxplot",
        # "Barchart",
        # "Scatter Plot",
        "Correlation Heatmap",
    ]
    output_structure = {
        "chart_name": {
            "x-axis": ["column names"],
            "y-axis": ["column names"],
            "title" : "Give Title for this visualization",
            "reason": "Why these columns were selected, a breif summary of 100 words",
        }
    }
    prompt = f"""
    You are an experience Data Analyst having superb skills in Data Visualization.

    Column Data:
    {df_info}

    Necessary Charts to Visualize: {",".join(name_of_charts)}

    You have to decide from above column data that which columns should be used to create the charts mentioned.
    
    The output should be in following json format:
    {output_structure}

    Rules:
    - If No columns met the criteria the chart should not be in the response.
    - Make sure to only send the json in the response, nothing else, No other string or not even an extra comma
    - Make sure to choose the columns based on the analysis of the data type give, like for histogram, we can use something like Price Each
    """

    url = "https://ollama.com/api/generate"  # Adjust to your exact endpoint
    api_key = config("CLOUD_OLLAMA_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {"model": "gemma4:31b-cloud", "prompt": prompt, "stream": False}

    response = requests.post(url, json=payload, headers=headers)
    return json.loads(response.json()["response"].replace("`", "").replace("json", ""))
