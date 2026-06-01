import os
import re
import argparse
import pandas as pd
from openai import OpenAI
from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.talk_to_data.prompt_templates import SQL_TRANSLATION_PROMPT
from src.talk_to_data.query_runner import execute_query

logger = setup_logger("nl_to_sql")
config = get_config()

def generate_fallback_sql(user_query: str) -> str:
    """
    A smart rule-based regex fallback to translate common credit risk questions to SQL,
    ensuring the script is runnable even without live API credentials.
    """
    query_lower = user_query.lower()
    
    if "average income" in query_lower or "avg income" in query_lower:
        return "SELECT AVG(AMT_INCOME_TOTAL) AS average_income FROM applications;"
        
    elif "default rate" in query_lower or "default percentage" in query_lower:
        if "children" in query_lower:
            return "SELECT CNT_CHILDREN, COUNT(*) as total_apps, ROUND(AVG(TARGET) * 100, 2) as default_rate_percent FROM applications GROUP BY CNT_CHILDREN ORDER BY CNT_CHILDREN ASC;"
        return "SELECT COUNT(*) as total_apps, ROUND(AVG(TARGET) * 100, 2) as default_rate_percent FROM applications;"
        
    elif "top 5" in query_lower or "highest credit" in query_lower:
        return "SELECT SK_ID_CURR, AMT_INCOME_TOTAL, AMT_CREDIT, TARGET FROM applications ORDER BY AMT_CREDIT DESC LIMIT 5;"
        
    elif "repaid" in query_lower or "defaulted" in query_lower:
        if "count" in query_lower or "how many" in query_lower:
            return "SELECT TARGET, COUNT(*) as total FROM applications GROUP BY TARGET;"
            
    # Default fallback
    return "SELECT SK_ID_CURR, AMT_INCOME_TOTAL, AMT_CREDIT, TARGET FROM applications LIMIT 10;"

def translate_nl_to_sql(user_query: str) -> str:
    """
    Translates Natural Language queries to SQL using the Groq API (llama3-8b-8192),
    falling back to local pattern-matching if no key is configured.
    """
    if not config.groq_api_key:
        logger.warning("GROQ_API_KEY environment variable not configured. Applying smart local fallback engine...")
        sql = generate_fallback_sql(user_query)
        logger.info(f"Synthesized Local SQL Fallback: {sql}")
        return sql

    logger.info("Connecting to Groq API for SQL synthesis...")
    try:
        client = OpenAI(
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        prompt = SQL_TRANSLATION_PROMPT.format(user_query=user_query)
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a professional credit risk database architect. Return ONLY the valid SQL statement inside a single code block starting with ```sql and ending with ```."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        raw_reply = response.choices[0].message.content.strip()
        
        # Extract SQL using regex
        sql_match = re.search(r"```sql\n(.*?)\n```", raw_reply, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1).strip()
        else:
            sql = raw_reply.replace("```", "").strip()
            
        logger.info(f"Synthesized Groq SQL: {sql}")
        return sql
        
    except Exception as e:
        logger.error(f"Failed to communicate with Groq LLM: {str(e)}. Using fallback SQL parser.")
        return generate_fallback_sql(user_query)

def synthesize_business_insight(user_query: str, sql_query: str, results_df: pd.DataFrame) -> str:
    """
    Leverages Groq LLM to synthesize a natural, readable business insight
    based on the raw SQL results.
    """
    if not config.groq_api_key:
        return "Local Insight: Database query executed successfully. Check the table below for statistics."
        
    if results_df.empty:
        return "No records were returned matching your search query."

    logger.info("Synthesizing business insight via Groq...")
    # Format database rows for the prompt
    data_summary = results_df.head(15).to_string(index=False)
    
    prompt = f"""You are a professional credit risk analyst.
Based on the client question, the SQL query executed, and the resulting data, write a concise, executive-level business insight summary.
Make it highly readable, interpret the percentage default rates or financial averages, and keep it under 3 sentences.

Client Question: "{user_query}"
SQL Query Executed: {sql_query}
Returned Data Table:
{data_summary}

Provide a short, direct insight to the client:"""

    try:
        client = OpenAI(
            api_key=config.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a professional credit analyst writing executive insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error synthesizing insight: {str(e)}")
        return "Database query successfully executed. Inspect the records in the table below."

def ask_data(query: str):
    logger.info(f"User Query: '{query}'")
    
    # 1. Translate NL to SQL
    sql_query = translate_nl_to_sql(query)
    
    # 2. Run SQL query on SQLite/DuckDB database
    try:
        results = execute_query(sql_query)
        print("\n--- SQL Query Results ---")
        print(results.to_string(index=False))
        print("-------------------------")
        
        # 3. Synthesize insight
        insight = synthesize_business_insight(query, sql_query, results)
        print("\n--- Business Insight ---")
        print(insight)
        print("-------------------------\n")
    except Exception as e:
        logger.error(f"SQL execution error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talk to Data - NL to SQL Engine")
    parser.add_argument("--query", type=str, default="Show me the top 5 highest credit amounts", help="Natural Language database query")
    
    args = parser.parse_args()
    ask_data(args.query)
