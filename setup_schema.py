import os
import argparse
import pandas as pd
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
import vanna
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat
from urllib.parse import quote_plus

# Custom Vanna class that combines ChromaDB vector store with OpenAI
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self.engine = None
        self.run_sql_is_set = False
    
    def connect_to_database(self, engine):
        """Connect to the database using SQLAlchemy engine."""
        self.engine = engine
        
        # Define the run_sql method that executes queries
        def run_sql(sql):
            return pd.read_sql_query(sql, self.engine)
        
        # Set the run_sql function
        self.run_sql = run_sql
        self.run_sql_is_set = True
        
    def generate_explanation(self, sql):
        """Generate an explanation of the SQL query."""
        message_log = [
            self.system_message("You are a helpful SQL assistant. Explain the following SQL query in plain English without technical jargon."),
            self.user_message(f"Explain this SQL query: {sql}")
        ]
        return self.submit_prompt(message_log)

def get_all_ddl(engine):
    """Get all table definitions from the database."""
    inspector = inspect(engine)
    schema_name = 'public'  # Default PostgreSQL schema
    
    ddl_statements = []
    
    for table_name in inspector.get_table_names(schema=schema_name):
        columns = inspector.get_columns(table_name, schema=schema_name)
        
        column_defs = []
        for column in columns:
            column_def = f"{column['name']} {column['type']}"
            if not column.get('nullable', True):
                column_def += " NOT NULL"
            column_defs.append(column_def)
        
        ddl = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(column_defs) + "\n);"
        ddl_statements.append(ddl)
    
    return ddl_statements

def extract_sql_queries(file_path):
    """Extract SQL queries from a SQL file."""
    if not os.path.exists(file_path):
        return False, []
        
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content by SQL comments that indicate new queries
    query_blocks = re.split(r'--\s*=+|--\s*\w+', content)
    
    # Process each block to extract valid SQL queries
    queries = []
    for block in query_blocks:
        # Remove SQL comments
        block = re.sub(r'--.*?$', '', block, flags=re.MULTILINE)
        
        # Split on semicolons to get individual queries
        individual_queries = [q.strip() for q in re.split(r';\s*', block) if q.strip()]
        
        # Add the queries that are not empty and look like valid SQL
        for query in individual_queries:
            # Basic check: must contain SELECT, INSERT, UPDATE, DELETE, etc.
            if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\b', query, re.IGNORECASE):
                queries.append(query + ';')  # Add the semicolon back
    
    return True, queries

def generate_question_for_query(vn, sql):
    """Generate a question for a SQL query."""
    return vn.generate_question(sql)

def main():
    parser = argparse.ArgumentParser(description='Train Vanna on database schema and SQL queries')
    parser.add_argument('--skip-train', action='store_true', help='Skip training and only print DDL')
    parser.add_argument('--skip-sql-train', action='store_true', help='Skip training on SQL queries')
    parser.add_argument('--sql-file', default='queries.sql', help='Path to SQL file with queries for training')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    # Get database connection
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    # Look for DB_DATABASE first, then fallback to DB_NAME
    db_name = os.getenv("DB_DATABASE") or os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD", "")
    
    if not all([db_name, db_user]):
        print("Error: Database credentials not found in environment")
        print("Please set DB_USER and either DB_DATABASE or DB_NAME in the .env file")
        return
    
    # URL encode the username, password, and database name
    encoded_user = quote_plus(db_user)
    encoded_password = quote_plus(db_password)
    encoded_db_name = quote_plus(db_name)
    
    connection_string = f"postgresql://{encoded_user}:{encoded_password}@{db_host}:{db_port}/{encoded_db_name}"
    
    try:
        engine = create_engine(connection_string)
        print(f"Successfully connected to database: {db_name}")
        
        # Get DDL statements
        ddl_statements = get_all_ddl(engine)
        
        if args.skip_train:
            print("\nDatabase schema:\n")
            for ddl in ddl_statements:
                print(ddl)
                print()
                
            # Extract SQL queries from the file if not skipping SQL training
            if not args.skip_sql_train:
                print("\nSQL queries for training:\n")
                file_exists, sql_queries = extract_sql_queries(args.sql_file)
                
                if not file_exists:
                    print(f"INFO: Queries file '{args.sql_file}' was not found. Skipping SQL training step.")
                elif not sql_queries:
                    print(f"INFO: No valid SQL queries found in '{args.sql_file}'.")
                else:
                    for i, query in enumerate(sql_queries, 1):
                        print(f"Query {i}:")
                        print(query)
                        print()
            return
        
        # Define the path to store the ChromaDB data
        chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize Vanna with ChromaDB and OpenAI
        vn = MyVanna(
            config={
                "api_key": openai_api_key,
                "model": "gpt-4",
                "path": chroma_path
            }
        )
        
        # Connect Vanna to the database
        vn.connect_to_database(engine=engine)
        
        # Train Vanna on the schema
        print("\nTraining Vanna on database schema...")
        for ddl in ddl_statements:
            print(f"Adding DDL: {ddl[:60]}...")
            vn.train(ddl=ddl)  # Use ddl parameter for DDL statements
        
        # Train Vanna on SQL queries if not skipped
        if not args.skip_sql_train:
            print("\nTraining Vanna on SQL queries...")
            
            # Extract SQL queries from the file
            file_exists, sql_queries = extract_sql_queries(args.sql_file)
            
            if not file_exists:
                print(f"INFO: Queries file '{args.sql_file}' was not found. Skipping SQL training step.")
            elif not sql_queries:
                print(f"INFO: No valid SQL queries found in '{args.sql_file}'. Skipping SQL training step.")
            else:
                for i, query in enumerate(sql_queries, 1):
                    try:
                        # Generate a question for this SQL query
                        question = generate_question_for_query(vn, query)
                        
                        # Train Vanna on the question and SQL query
                        print(f"Training on Query {i}: {question[:60]}...")
                        vn.train(question=question, sql=query)
                        
                    except Exception as e:
                        print(f"Error training on query {i}: {e}")
        
        print("\nSuccessfully trained Vanna on database schema and SQL queries!")
        print("You can now use the main app to query your database in natural language.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 