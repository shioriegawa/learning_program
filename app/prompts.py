SYSTEM_PROMPT = """You are a SQL query generator for a sales analytics system. You must generate ONLY safe SELECT queries for the 'sales' table.

Table Schema:
- sales table columns: date, category, units, unit_price, region, sales_channel, customer_segment, revenue

Rules:
1. Generate ONLY SELECT statements (WITH...SELECT is also allowed)
2. Use ONLY the 'sales' table
3. Return valid JSON with this exact structure: {"sql": "your_query_here", "chart_suggestion": "chart_type_suggestion"}
4. Chart suggestions: "bar", "line", "pie", "scatter", "table"
5. Include appropriate aggregations (SUM, COUNT, AVG, etc.) when analyzing metrics
6. Use proper date formatting and grouping for time-based queries
7. Always limit results to reasonable numbers (use LIMIT when appropriate)

Your response must be valid JSON only - no explanations or additional text."""

FEW_SHOT_EXAMPLES = [
    {
        "user_query": "月毎×カテゴリ別の売上を見せて",
        "response": {
            "sql": "SELECT DATE_TRUNC('month', date::DATE) as month, category, SUM(revenue) as total_revenue FROM sales GROUP BY DATE_TRUNC('month', date::DATE), category ORDER BY month, total_revenue DESC LIMIT 50",
            "chart_suggestion": "bar"
        }
    },
    {
        "user_query": "チャネル別売上を教えて",
        "response": {
            "sql": "SELECT sales_channel, SUM(revenue) as total_revenue, COUNT(*) as transaction_count FROM sales GROUP BY sales_channel ORDER BY total_revenue DESC",
            "chart_suggestion": "pie"
        }
    },
    {
        "user_query": "地域別売上合計",
        "response": {
            "sql": "SELECT region, SUM(revenue) as total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC",
            "chart_suggestion": "bar"
        }
    },
    {
        "user_query": "What are the top selling products by category?",
        "response": {
            "sql": "SELECT category, SUM(units) as total_units_sold, SUM(revenue) as total_revenue FROM sales GROUP BY category ORDER BY total_revenue DESC LIMIT 10",
            "chart_suggestion": "bar"
        }
    },
    {
        "user_query": "Show me sales trends over time",
        "response": {
            "sql": "SELECT DATE_TRUNC('month', date::DATE) as month, SUM(revenue) as monthly_revenue FROM sales GROUP BY DATE_TRUNC('month', date::DATE) ORDER BY month",
            "chart_suggestion": "line"
        }
    }
]

def get_runtime_prompt(user_query: str) -> str:
    """Generate the complete prompt for LLM"""
    few_shot_text = "\n\nExamples:\n"
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_text += f"\nExample {i}:\n"
        few_shot_text += f"User: {example['user_query']}\n"
        few_shot_text += f"Response: {example['response']}\n"
    
    return f"""{SYSTEM_PROMPT}{few_shot_text}

Now generate a SQL query for this user request:
User: {user_query}

Response (JSON only):"""

def get_preset_queries():
    """Get preset queries for quick access"""
    return {
        "月毎×カテゴリ": {
            "sql": "SELECT DATE_TRUNC('month', date::DATE) as month, category, SUM(revenue) as total_revenue FROM sales GROUP BY DATE_TRUNC('month', date::DATE), category ORDER BY month, total_revenue DESC LIMIT 50",
            "chart_suggestion": "bar"
        },
        "チャネル別売上": {
            "sql": "SELECT sales_channel, SUM(revenue) as total_revenue, COUNT(*) as transaction_count FROM sales GROUP BY sales_channel ORDER BY total_revenue DESC",
            "chart_suggestion": "pie"
        },
        "地域別売上合計": {
            "sql": "SELECT region, SUM(revenue) as total_revenue FROM sales GROUP BY region ORDER BY total_revenue DESC",
            "chart_suggestion": "bar"
        }
    }