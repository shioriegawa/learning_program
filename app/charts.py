import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

def auto_chart(df: pd.DataFrame, chart_suggestion: str = "table") -> Optional[go.Figure]:
    """
    Generate appropriate Plotly chart based on data and suggestion
    """
    if df.empty:
        return None
    
    # Determine chart type
    chart_type = chart_suggestion.lower()
    
    try:
        # Get column info
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Auto-detect datetime columns that might be stored as objects
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains date-like strings
                sample_value = str(df[col].iloc[0]) if len(df) > 0 else ""
                if any(keyword in sample_value.lower() for keyword in ['month', 'date', 'year']) or \
                   any(char in sample_value for char in ['-', '/']):
                    datetime_cols.append(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)
        
        # Generate chart based on suggestion and data structure
        if chart_type == "bar":
            return create_bar_chart(df, numeric_cols, categorical_cols, datetime_cols)
        elif chart_type == "line":
            return create_line_chart(df, numeric_cols, categorical_cols, datetime_cols)
        elif chart_type == "pie":
            return create_pie_chart(df, numeric_cols, categorical_cols)
        elif chart_type == "scatter":
            return create_scatter_chart(df, numeric_cols, categorical_cols)
        else:
            # Default to bar chart if data is suitable
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                return create_bar_chart(df, numeric_cols, categorical_cols, datetime_cols)
            return None
            
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None

def create_bar_chart(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, datetime_cols: list) -> go.Figure:
    """Create bar chart"""
    if not numeric_cols:
        return None
    
    # Choose x and y axes
    if datetime_cols:
        x_col = datetime_cols[0]
    elif categorical_cols:
        x_col = categorical_cols[0]
    else:
        x_col = df.columns[0]
    
    y_col = numeric_cols[0]
    
    # Check if we have multiple categories to group by
    color_col = None
    if len(categorical_cols) >= 2:
        color_col = categorical_cols[1] if categorical_cols[0] == x_col else categorical_cols[0]
    elif len(categorical_cols) == 1 and categorical_cols[0] != x_col:
        color_col = categorical_cols[0]
    
    fig = px.bar(
        df.head(50),  # Limit to 50 rows for performance
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{y_col} by {x_col}",
        height=500
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=bool(color_col)
    )
    
    return fig

def create_line_chart(df: pd.DataFrame, numeric_cols: list, categorical_cols: list, datetime_cols: list) -> go.Figure:
    """Create line chart (ideal for time series)"""
    if not numeric_cols:
        return None
    
    # Prefer datetime columns for x-axis
    if datetime_cols:
        x_col = datetime_cols[0]
    elif categorical_cols:
        x_col = categorical_cols[0]
    else:
        x_col = df.columns[0]
    
    y_col = numeric_cols[0]
    
    # Check for grouping column
    color_col = None
    if len(categorical_cols) >= 1 and categorical_cols[0] != x_col:
        color_col = categorical_cols[0]
    elif len(categorical_cols) >= 2:
        color_col = categorical_cols[1]
    
    fig = px.line(
        df.head(100),  # Limit for performance
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{y_col} trend over {x_col}",
        height=500
    )
    
    return fig

def create_pie_chart(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> go.Figure:
    """Create pie chart"""
    if not numeric_cols or not categorical_cols:
        return None
    
    # Use first categorical for names and first numeric for values
    names_col = categorical_cols[0]
    values_col = numeric_cols[0]
    
    # Take top 10 categories for readability
    df_top = df.nlargest(10, values_col)
    
    fig = px.pie(
        df_top,
        values=values_col,
        names=names_col,
        title=f"{values_col} distribution by {names_col}",
        height=500
    )
    
    return fig

def create_scatter_chart(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> go.Figure:
    """Create scatter plot"""
    if len(numeric_cols) < 2:
        return None
    
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    # Color by categorical if available
    color_col = categorical_cols[0] if categorical_cols else None
    
    fig = px.scatter(
        df.head(200),  # Limit for performance
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{y_col} vs {x_col}",
        height=500
    )
    
    return fig

def create_fallback_chart(df: pd.DataFrame) -> go.Figure:
    """Create a simple fallback chart for category revenue"""
    if df.empty:
        return None
    
    fig = px.bar(
        df,
        x='category',
        y='total_revenue',
        title='Category Revenue (Fallback)',
        height=400
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig