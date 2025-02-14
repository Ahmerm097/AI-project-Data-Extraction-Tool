import google.generativeai as genai
import pandas as pd
import json
import warnings
from secret_key import google_api_key

# Suppress gRPC warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure the GenAI client
genai.configure(api_key=google_api_key)


def extract_financial_data(text):
    prompt = get_prompt_financial() + text
    model = genai.GenerativeModel('gemini-pro')

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0)
        )
        content = response.text

        # Debug: Print raw response
        print("Raw API Response:\n", content, "\n" + "-" * 50)

        # Clean response (remove Markdown backticks)
        content = content.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        data = json.loads(content)
        return pd.DataFrame(data.items(), columns=["Measure", "Value"])

    except json.JSONDecodeError:
        print("JSON Parsing Failed. Returning empty DataFrame.")
        return empty_dataframe()
    except Exception as e:
        print(f"Error: {e}")
        return empty_dataframe()


def get_prompt_financial():
    return '''Extract the following details from the news article below. 
    If any information is missing, leave the value empty. Do NOT invent data.

    Required Fields:
    - Company Name
    - Stock Symbol
    - Revenue
    - Net Income
    - EPS (Earnings Per Share)

    Return the data as strictly valid JSON in this format:
    {
        "Company Name": "Company Name",
        "Stock Symbol": "SYM",
        "Revenue": "X.XX billion/million",
        "Net Income": "Y.YY billion/million",
        "EPS": "Z.ZZ $"
    }

    News Article:
    ============
    '''


def empty_dataframe():
    return pd.DataFrame({
        "Measure": ["Company Name", "Stock Symbol", "Revenue", "Net Income", "EPS"],
        "Value": ["", "", "", "", ""]
    })


if __name__ == '__main__':
    text = '''
    Tesla's Earning news in text format: Tesla's earning this quarter blew all the estimates.
    They reported 4.5 billion $ profit against a revenue of 30 billion $. Their earnings per share was 2.3 $
    '''
    df = extract_financial_data(text)
    print("\nFinal Output:")
    print(df.to_string(index=False))