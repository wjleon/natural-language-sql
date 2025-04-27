# Talk to Postgres DB

A natural language interface for PostgreSQL databases using Vanna.ai and OpenAI. This application allows you to query your database using plain English instead of SQL.

## Features

- Query your PostgreSQL database using natural language
- Generate SQL from your questions automatically
- Get visual results and natural language explanations
- Uses ChromaDB for vector storage and retrieval
- Requires your own OpenAI API key (GPT-4 model)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables by copying the template:
   ```
   cp env.example .env
   ```
4. Edit `.env` and add your OpenAI API key and database credentials

## Importing Sample Data (World Database)

If you want to use the sample World database:

1. Create a new database:
   ```
   createdb world_data
   ```

2. Import the SQL file:
   ```
   psql -d world -f dbsamples-0.1/world/world.sql
   ```

3. Update your `.env` file to use this database:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=world
   DB_USER=your_username
   DB_PASSWORD=your_password
   ```

## Training on Your Schema

Before using the app, it's recommended to train Vanna on your database schema:

```
python setup_schema.py
```

This will extract table definitions from your database and train Vanna to understand your specific schema, improving the quality of generated SQL.

To view your database schema without training:

```
python setup_schema.py --skip-train
```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```
2. Enter your natural language query in the text input field
3. The app will generate SQL, run the query, and show results with explanations

## Requirements

- Python 3.8+
- PostgreSQL database
- OpenAI API key (GPT-4 model recommended)
- Required Python packages (see requirements.txt):
  - streamlit
  - vanna
  - python-dotenv
  - sqlalchemy
  - psycopg2-binary
  - openai
  - chromadb

## Example Queries

- "How many countries aare there in the world?"
- "How many cities aare there in the world?"
- "What is the country with the highest number of cities?"

## Technical Details

The application uses:
- Vanna.ai for natural language to SQL conversion
- ChromaDB as a vector store for embeddings
- Streamlit for the web interface
- SQLAlchemy for database connections
- OpenAI GPT-4 for language processing 