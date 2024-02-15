# Import modules
from functools import wraps
from dotenv import load_dotenv
import pandas as pd
import psycopg2
import os

# Read database secrets fro DATABASE_URL
# Include sslmode='require' to enforce SSL connection
# Replace 'hostname' and 'username' with actual values
load_dotenv()
hostname = os.getenv('hostname')
username = os.getenv('username')
database = os.getenv('database')
password = os.getenv('password')

# Use variables to connect to the database instead of hardcoding the values
db_url = f"postgresql://{username}:{password}@{hostname}/{database}"
# db_url = os.environ["DATABASE_URL"]
print(f"Using database: {db_url}")


# Database connection decorator
def database_connection(func):

  @wraps(func)
  def with_connection(*args, **kwargs):
    conn = None  # Initialize conn to None
    try:
      conn = psycopg2.connect(db_url)  # Connect to the database
      cur = conn.cursor()
      try:
        result = func(cur, *args, **kwargs)
      finally:
        conn.commit()
        cur.close()
      return result
    except (Exception, psycopg2.Error) as error:
      print(f"Error in {func.__name__}: ", error)
    finally:
      if conn:
        conn.close()

  return with_connection


# Kiel dataframe column names when inserting data
info_keys = [
    "acid_temperature",
    "leakrate",
    "p_no_acid",
    "p_gases",
    "reference_refill",
    "total_CO2",
    "vm1_after_transfer",
    "initial_intensity",
    "reference_intensity",
    "reference_bellow_position",
    "line",
    "standard",
]


# Create postresql database table users
@database_connection
def create_kiel_table(cur):
  # Include other columns from info_keys with appropriate data types
  cur.execute("""
  CREATE TABLE IF NOT EXISTS kiel_parameters (
      id SERIAL PRIMARY KEY,
      time TIMESTAMP UNIQUE NOT NULL,
      "line" INTEGER NOT NULL,
      "standard" TEXT,
      "acid_temperature" FLOAT,
      "leakrate" FLOAT,
      "p_no_acid" FLOAT,
      "p_gases" FLOAT,
      "reference_refill" FLOAT,
      "total_CO2" FLOAT,
      "vm1_after_transfer" FLOAT,
      "initial_intensity" FLOAT,
      "reference_intensity" FLOAT,
      "reference_bellow_position" FLOAT
      -- Add any additional columns as needed
  );
  """)
  print("Kiel Parameters table created successfully")


# Extend the existing functions to include an insert function for df_kiel_par
@database_connection
def insert_kiel_data(cur, df: pd.DataFrame):
    
  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
  
  # This will convert NaN values to None, which psycopg2 will interpret as      NULL
  df = df.applymap(lambda x: None if pd.isna(x) else x)

  # Rename Line and Identifier 1 columns to match the table columns
  df.rename(columns={"Line": "line", "Identifier 1": "standard"}, inplace=True)

  # Convert the TimeCode column to a string with a format PostgreSQL understands
  df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
  # df_kiel_par['time'] = df_kiel_par['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
  df.reset_index(inplace=True)
  df.rename(columns={"Time Code": "time"}, inplace=True)

  # Prepare the column names and placeholders for the SQL statement
  columns = ", ".join(["\"{}\"".format(key) for key in info_keys])
  placeholders = ", ".join(["%s"] * len(info_keys))

  # Generate the SQL command and insert each row
  for index, row in df.iterrows():
    # Check if the TimeCode already exists in the database
    cur.execute("SELECT EXISTS(SELECT 1 FROM kiel_parameters WHERE time = %s)",
                (row['time'], ))
    exists = cur.fetchone()[0]

    if not exists:
      values = tuple(row[key] for key in info_keys)
      try:
        sql = f"INSERT INTO kiel_parameters (time, {columns}) VALUES (%s, {placeholders})"
        cur.execute(sql, (row['time'], ) + values)
      except psycopg2.Error as error:
        print(f"Failed to insert data: {values}")
        print(f"Error: {error}")
        if "integer out of range" in str(error):
          print("One of the integer columns has a value out of range.")
          # Log the exact value and column, or perform additional checks.
        continue  # Skip this row and continue with the next

  print("Data inserted successfully, duplicates ignored based on timestamp.")


# Read postresql database table users
@database_connection
def read_table(cur):
  cur.execute("SELECT * FROM kiel_parameters")
  rows = cur.fetchall()
  print("Table Kiel Parameters:")
  for row in rows:
    print(row)


# Fetch single user from database
# @database_connection
# def read_all_data(cur, name):
#   cur.execute("SELECT * FROM )
#   row = cur.fetchone()
#   return row
