import re
from typing import Tuple

class SQLGuard:
    """SQL validation to ensure only safe SELECT queries on sales table"""
    
    ALLOWED_PATTERNS = [
        r'^\s*(WITH\s+.+\s+)?SELECT\s+',  # Must start with optional WITH + SELECT
    ]
    
    FORBIDDEN_PATTERNS = [
        r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE)\b',
        r'\b(GRANT|REVOKE|COMMIT|ROLLBACK)\b',
        r'--',  # Comments
        r'/\*',  # Multi-line comments
        r';.*\S',  # Multiple statements (anything after semicolon that's not whitespace)
    ]
    
    ALLOWED_TABLES = ['sales']
    
    @staticmethod
    def validate_sql(sql: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety
        Returns (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"
        
        sql_upper = sql.upper()
        
        # Check if starts with allowed patterns
        starts_with_allowed = any(
            re.match(pattern, sql_upper, re.IGNORECASE)
            for pattern in SQLGuard.ALLOWED_PATTERNS
        )
        
        if not starts_with_allowed:
            return False, "Query must start with SELECT or WITH...SELECT"
        
        # Check for forbidden patterns
        for pattern in SQLGuard.FORBIDDEN_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return False, f"Forbidden SQL pattern detected: {pattern}"
        
        # Check table access - only allow 'sales' table
        # Extract table names using regex
        table_pattern = r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)'
        matches = re.findall(table_pattern, sql_upper, re.IGNORECASE)
        
        used_tables = set()
        for match in matches:
            # match is a tuple, get the non-empty group
            table = match[0] if match[0] else match[1]
            if table:
                used_tables.add(table.lower())
        
        # Allow empty (no tables found) or only 'sales' table
        if used_tables and not used_tables.issubset({'sales'}):
            return False, f"Only 'sales' table is allowed. Found: {', '.join(used_tables)}"
        
        return True, ""
    
    @staticmethod
    def sanitize_sql(sql: str) -> str:
        """Clean and sanitize SQL query"""
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Ensure single statement (remove everything after first complete statement)
        if ';' in sql:
            sql = sql.split(';')[0].strip()
        
        return sql

def validate_and_sanitize_sql(sql: str) -> Tuple[bool, str, str]:
    """
    Validate and sanitize SQL query
    Returns (is_valid, cleaned_sql, error_message)
    """
    try:
        # First sanitize
        cleaned_sql = SQLGuard.sanitize_sql(sql)
        
        # Then validate
        is_valid, error_msg = SQLGuard.validate_sql(cleaned_sql)
        
        return is_valid, cleaned_sql if is_valid else "", error_msg
        
    except Exception as e:
        return False, "", f"SQL validation error: {str(e)}"