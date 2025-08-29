import duckdb
import pandas as pd
import os

class DuckDBManager:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = duckdb.connect(db_path)
        
    def load_csv_to_sales_table(self, csv_path: str) -> bool:
        """Load CSV data into sales table with revenue补完"""
        try:
            # First, read CSV into temporary table
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE temp_sales AS
                SELECT * FROM read_csv_auto('{csv_path}')
            """)
            
            # Create sales table with revenue补完 using COALESCE
            self.conn.execute("""
                CREATE OR REPLACE TABLE sales AS
                SELECT 
                    date,
                    category,
                    units,
                    unit_price,
                    region,
                    sales_channel,
                    customer_segment,
                    COALESCE(revenue, units * unit_price) as revenue
                FROM temp_sales
            """)
            
            # Drop temp table
            self.conn.execute("DROP TABLE temp_sales")
            
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return pandas DataFrame"""
        try:
            result = self.conn.execute(sql).df()
            return result
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()
    
    def get_table_info(self) -> dict:
        """Get information about the sales table"""
        try:
            # Get column info
            columns_info = self.conn.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'sales'
            """).df()
            
            # Get row count
            row_count = self.conn.execute("SELECT COUNT(*) as count FROM sales").df().iloc[0]['count']
            
            # Get sample data
            sample_data = self.conn.execute("SELECT * FROM sales LIMIT 5").df()
            
            return {
                'columns': columns_info.to_dict('records'),
                'row_count': row_count,
                'sample_data': sample_data
            }
            
        except Exception as e:
            print(f"Error getting table info: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Global instance
db_manager = None

def get_db_manager():
    """Get or create DuckDB manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DuckDBManager()
        # Load CSV if it exists
        csv_path = "/workspaces/learning_program/data/sample_sales.csv"
        if os.path.exists(csv_path):
            db_manager.load_csv_to_sales_table(csv_path)
    return db_manager

def execute_sql(sql: str) -> pd.DataFrame:
    """Execute SQL query"""
    return get_db_manager().execute_query(sql)

def get_fallback_data() -> pd.DataFrame:
    """Get fallback data when main query fails"""
    fallback_sql = """
    SELECT 
        category,
        SUM(revenue) as total_revenue
    FROM sales 
    GROUP BY category 
    ORDER BY total_revenue DESC
    """
    return execute_sql(fallback_sql)