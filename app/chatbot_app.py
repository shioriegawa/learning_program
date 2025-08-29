import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

from duck import get_db_manager, execute_sql, get_fallback_data
from llm import generate_sql_from_query
from sql_guard import validate_and_sanitize_sql
from charts import auto_chart, create_fallback_chart
from prompts import get_preset_queries

def main():
    st.set_page_config(
        page_title="Sales Analytics Chatbot",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Sales Analytics Chatbot")
    st.markdown("Ask questions about your sales data in natural language!")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Data Overview")
        
        # Initialize database and show info
        try:
            db_manager = get_db_manager()
            if not st.session_state.db_initialized:
                table_info = db_manager.get_table_info()
                if table_info:
                    st.session_state.db_initialized = True
                    st.success("✅ Data loaded successfully!")
                    
                    st.subheader("Table Info")
                    st.write(f"**Rows:** {table_info['row_count']}")
                    
                    if 'columns' in table_info:
                        st.write("**Columns:**")
                        for col in table_info['columns']:
                            st.write(f"- {col['column_name']} ({col['data_type']})")
                    
                    if 'sample_data' in table_info and not table_info['sample_data'].empty:
                        st.write("**Sample Data:**")
                        st.dataframe(table_info['sample_data'].head(3), use_container_width=True)
                else:
                    st.error("❌ Data not loaded. Check if sample_sales.csv exists.")
            else:
                st.success("✅ Data loaded successfully!")
                
        except Exception as e:
            st.error(f"❌ Database error: {str(e)}")
        
        # Preset queries
        st.header("🚀 Quick Queries")
        preset_queries = get_preset_queries()
        
        for label, query_info in preset_queries.items():
            if st.button(label, key=f"preset_{label}", use_container_width=True):
                execute_preset_query(query_info, label)
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.expander(f"Query {i+1}: {chat['question'][:50]}...", expanded=(i == len(st.session_state.chat_history) - 1)):
            st.write(f"**Question:** {chat['question']}")
            
            if chat.get('sql'):
                st.code(chat['sql'], language='sql')
            
            if chat.get('error'):
                st.error(f"Error: {chat['error']}")
            
            if chat.get('data') is not None and not chat['data'].empty:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Data Results:**")
                    st.dataframe(chat['data'], use_container_width=True)
                
                with col2:
                    if chat.get('chart'):
                        st.write("**Visualization:**")
                        st.plotly_chart(chat['chart'], use_container_width=True)
    
    # Input form
    with st.form("query_form", clear_on_submit=True):
        user_query = st.text_input(
            "Ask a question about your sales data:",
            placeholder="e.g., Show me monthly sales by category",
            help="Try: '月毎×カテゴリ別の売上を見せて' or 'What are the top selling regions?'"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("🔍 Analyze", type="primary")
        
        if submit_button and user_query.strip():
            process_query(user_query.strip())

def execute_preset_query(query_info: dict, label: str):
    """Execute a preset query"""
    try:
        # Validate SQL
        is_valid, cleaned_sql, error_msg = validate_and_sanitize_sql(query_info['sql'])
        
        if not is_valid:
            st.error(f"❌ SQL Validation Error: {error_msg}")
            return
        
        # Execute query
        df = execute_sql(cleaned_sql)
        
        if df.empty:
            st.warning("⚠️ Query returned no results")
            return
        
        # Create visualization
        chart = auto_chart(df, query_info.get('chart_suggestion', 'table'))
        
        # Add to chat history
        st.session_state.chat_history.append({
            'question': f"[Preset] {label}",
            'sql': cleaned_sql,
            'data': df,
            'chart': chart,
            'error': None,
            'timestamp': datetime.now()
        })
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error executing preset query: {str(e)}")

def process_query(user_query: str):
    """Process user query through the full pipeline"""
    with st.spinner('🤖 Generating SQL query...'):
        # Generate SQL using LLM
        llm_result = generate_sql_from_query(user_query)
        
        if not llm_result['success']:
            # LLM failed, use fallback
            st.warning(f"⚠️ LLM Error: {llm_result['error']}. Using fallback query.")
            fallback_query(user_query)
            return
        
        generated_sql = llm_result['sql']
        chart_suggestion = llm_result.get('chart_suggestion', 'table')
    
    with st.spinner('🔐 Validating SQL...'):
        # Validate SQL
        is_valid, cleaned_sql, error_msg = validate_and_sanitize_sql(generated_sql)
        
        if not is_valid:
            st.error(f"❌ SQL Validation Error: {error_msg}")
            fallback_query(user_query)
            return
    
    with st.spinner('🗄️ Executing query...'):
        try:
            # Execute query
            df = execute_sql(cleaned_sql)
            
            if df.empty:
                st.warning("⚠️ Query returned no results. Using fallback.")
                fallback_query(user_query)
                return
            
        except Exception as e:
            st.error(f"❌ Query Execution Error: {str(e)}")
            fallback_query(user_query)
            return
    
    with st.spinner('📊 Creating visualization...'):
        # Create chart
        chart = auto_chart(df, chart_suggestion)
    
    # Add to chat history
    st.session_state.chat_history.append({
        'question': user_query,
        'sql': cleaned_sql,
        'data': df,
        'chart': chart,
        'error': None,
        'timestamp': datetime.now()
    })
    
    st.success("✅ Query processed successfully!")
    st.rerun()

def fallback_query(user_query: str):
    """Execute fallback query when main pipeline fails"""
    try:
        st.info("🔄 Executing fallback query: Category revenue summary")
        
        df = get_fallback_data()
        if not df.empty:
            chart = create_fallback_chart(df)
            
            st.session_state.chat_history.append({
                'question': user_query,
                'sql': "SELECT category, SUM(revenue) as total_revenue FROM sales GROUP BY category ORDER BY total_revenue DESC",
                'data': df,
                'chart': chart,
                'error': "Used fallback query due to processing error",
                'timestamp': datetime.now()
            })
            
            st.rerun()
        else:
            st.error("❌ Fallback query also failed")
            
    except Exception as e:
        st.error(f"❌ Fallback query error: {str(e)}")

if __name__ == "__main__":
    main()