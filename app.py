import os
import streamlit as st
import pandas as pd
import vanna
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set your OPENAI_API_KEY in the .env file")
    st.stop()

# Database connection
def get_db_connection():
    # Get database connection info from environment variables
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    # Look for DB_DATABASE first, then fallback to DB_NAME
    db_name = os.getenv("DB_DATABASE") or os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD", "")
    
    if not all([db_name, db_user]):
        st.error("Please set DB_USER and either DB_DATABASE or DB_NAME in the .env file")
        st.stop()
    
    # URL encode the username, password, and database name
    encoded_user = quote_plus(db_user)
    encoded_password = quote_plus(db_password)
    encoded_db_name = quote_plus(db_name)
    
    connection_string = f"postgresql://{encoded_user}:{encoded_password}@{db_host}:{db_port}/{encoded_db_name}"
    return create_engine(connection_string)

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

# Initialize Vanna with ChromaDB
def init_vanna():
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
    
    # Connect to database
    try:
        engine = get_db_connection()
        vn.connect_to_database(engine=engine)
        st.success("Connected to database successfully!")
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.stop()
    
    return vn

# Streamlit UI
st.title("Talk to Your Postgres Database")
st.write("Ask questions about your data in natural language")

# Initialize Vanna if not already in session state
if "vanna" not in st.session_state:
    st.session_state.vanna = init_vanna()

# User input
user_question = st.text_input("Ask a question about your data:")

if user_question:
    vn = st.session_state.vanna
    
    with st.spinner("Generating SQL..."):
        # Generate SQL from the natural language question
        sql = vn.generate_sql(user_question)
        
        if sql:
            st.subheader("Generated SQL:")
            st.code(sql, language="sql")
            
            # Run the query and display results
            with st.spinner("Running query..."):
                try:
                    df = vn.run_sql(sql)
                    st.subheader("Query Results:")
                    st.dataframe(df)
                    
                    # Generate natural language explanation
                    with st.spinner("Generating explanation..."):
                        explanation = vn.generate_explanation(sql)
                        st.subheader("Explanation:")
                        st.write(explanation)
                except Exception as e:
                    st.error(f"Error executing query: {e}")
        else:
            st.error("Failed to generate SQL from your question. Please try rephrasing.") 