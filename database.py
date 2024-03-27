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
    "instrument"
]

# Create postresql database kiel table
@database_connection
def create_kiel_table(cur):
  # Include other columns from info_keys with appropriate data types
  cur.execute("""
  CREATE TABLE IF NOT EXISTS kiel_parameters (
      id SERIAL PRIMARY KEY,
      time TIMESTAMP NOT NULL,
      instrument TEXT NOT NULL,
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
      "reference_bellow_position" FLOAT, 
      -- Add any additional columns as needed
      UNIQUE (time, instrument) -- Ensure the time&instrument are unique
  );
  """)
  print("Kiel Parameters table created successfully")

# Dictionary to map the pandas dataframe columns to the standard_parameters table entity names
standard_column_rename_dict = {
    'Line': 'line',
    'Weight': 'weight',
    'instrument': 'instrument',
    'Identifier 1': 'standard',
    '1  Cycle Int  Samp  44': 'init_intensity_44_sample',
    '1  Cycle Int  Ref  44': 'init_intensity_44_reference',
    'd 45CO2/44CO2  Std Dev': 'std_d45_44_co2',
    'd 46CO2/44CO2  Std Dev': 'std_d46_44_co2',
    'd 47CO2/44CO2  Std Dev': 'std_d47_44_co2',
    'd 48CO2/44CO2  Std Dev': 'std_d48_44_co2',
    'd 49CO2/44CO2  Std Dev': 'std_d49_44_co2'
}

# Create postresql database standards table
@database_connection
def create_standard_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS standard_parameters (
        id SERIAL PRIMARY KEY,
        time TIMESTAMP NOT NULL,
        instrument TEXT NOT NULL,
        line INTEGER NOT NULL,
        standard TEXT NOT NULL,
        weight FLOAT,
        init_intensity_44_sample FLOAT,
        init_intensity_44_reference FLOAT,
        std_d45_44_co2 FLOAT,
        std_d46_44_co2 FLOAT,
        std_d47_44_co2 FLOAT,
        std_d48_44_co2 FLOAT,
        std_d49_44_co2 FLOAT,
        UNIQUE (time, instrument) -- Ensure the time&instrument are unique
    );
    """)
    print("Standard Parameters table created successfully")

# Dictionary to map the pandas dataframe columns to the intensity_ratio_fit_pars table entity names
intensity_ratio_fit_pars_column_rename_dict = {
    'identifier': 'standard',
    'instrument': 'instrument',
    'isref': 'isref',
    'intensity_tatio': 'intensity_ratio',
    'init_intensity_44': 'init_intensity_44',
    'slope': 'slope',
    'intercept': 'intercept',
    'r2': 'r2'
}
# Create postresql database standard ratios fits table
@database_connection
def create_intensity_ratio_fit_pars_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS intensity_ratio_fit_pars (
        id SERIAL PRIMARY KEY,
        time TIMESTAMP NOT NULL,
        instrument TEXT NOT NULL,
        standard TEXT NOT NULL,
        isref TEXT NOT NULL,
        intensity_ratio TEXT,
        init_intensity_44 FLOAT,
        slope FLOAT,
        intercept FLOAT,
        r2 FLOAT,
        UNIQUE (time, instrument) -- Ensure the time&instrument are unique
    );
    """)
    print("Intensity ratio fit parameters table created successfully")

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
  for _, row in df.iterrows():
    # Check if the timestamp already exists in the database
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

  print("Kiel data inserted successfully, duplicates ignored based on timestamp.")

# Function to insert data into the standard_parameters table
@database_connection
def insert_standard_parameters_data(cur, df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    df = df.applymap(lambda x: None if pd.isna(x) else x)
    df.rename(columns=standard_column_rename_dict, inplace=True)

    df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    df.reset_index(inplace=True)
    df.rename(columns={"Time Code": "time"}, inplace=True)

    columns = ", ".join(["\"{}\"".format(key) for key in standard_column_rename_dict.values()])
    placeholders = ", ".join(["%s"] * len(standard_column_rename_dict))

    for _, row in df.iterrows():
        # Check if the timestamp already exists in the database
        cur.execute("SELECT EXISTS(SELECT 1 FROM standard_parameters WHERE time = %s)",
                    (row['time'], ))
        exists = cur.fetchone()[0]

        if not exists:
            values = tuple(row[key] for key in standard_column_rename_dict.values())
            try:
                sql = f"INSERT INTO standard_parameters (time, {columns}) VALUES (%s, {placeholders})"
                cur.execute(sql, (row['time'], ) + values)
            except psycopg2.Error as error:
                print(f"Failed to insert data: {values}")
                print(f"Error: {error}")
                if "integer out of range" in str(error):
                    print("One of the integer columns has a value out of range.")
                continue

    print("Standards data inserted successfully, duplicates ignored based on timestamp.")

# Function that insert intensity ratio fit pars into the database
@database_connection
def insert_intensity_ratio_fit_pars(cur, df: pd.DataFrame):
    # Add code to insert intensity ratio fit pars into the database
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    df = df.applymap(lambda x: None if pd.isna(x) else x)

    # Add 1 minute jitter to timestamps for the same identifier ensuring uniqueness
    df.sort_values(by=['standard', 'time'], inplace=True)
    df['time'] = df.groupby('standard')['time'].transform(
        lambda x: pd.to_datetime(x.iloc[0]) + pd.to_timedelta(range(len(x)), unit='T'))

    # Convert 'time' column to a string with a format PostgreSQL understands
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    columns = ", ".join(['"{}"'.format(col) for col in intensity_ratio_fit_pars_column_rename_dict.values()])
    placeholders = ", ".join(["%s"] * len(intensity_ratio_fit_pars_column_rename_dict))

    # Generate the SQL command to insert each row
    # ON CONFLICT (time) DO NOTHING will ignore duplicates based on the timestamp
    sql = f"INSERT INTO intensity_ratio_fit_pars ({columns}) VALUES ({placeholders})"

    # Rename the columns to match the table columns
    df.rename(columns=intensity_ratio_fit_pars_column_rename_dict, inplace=True)

    # Iterate each row in the DataFrame for insertion
    for _, row in df.iterrows():
        # Check if the timestamp already exists in the database
        cur.execute("SELECT EXISTS(SELECT 1 FROM intensity_ratio_fit_pars WHERE time = %s)",
                    (row['time'], ))
        exists = cur.fetchone()[0]
        
        if not exists:
            values = tuple(row[col] for col in intensity_ratio_fit_pars_column_rename_dict.values())

            try:
                sql = f"INSERT INTO intensity_ratio_fit_pars (time, {columns}) VALUES (%s, {placeholders})"
                cur.execute(sql, (row['time'],) + values)
            except psycopg2.Error as error:
                print(f"Failed to insert data: {values}")
                print(f"Error: {error}")
                continue

    print("Intensity ratio fit data inserted successfully, duplicates ignored based on timestamp.")

# Fetch kiel data as pandas DataFrame
@database_connection
def fetch_kiel_data_as_dataframe(cur, instrument_name):
    query = "SELECT * FROM kiel_parameters WHERE instrument = %(instrument_name)s"
    with cur.connection.cursor() as cursor:
        cursor.execute(query, {"instrument_name": instrument_name})
        colnames = [desc[0] for desc in cursor.description]
        records = cursor.fetchall()
    df = pd.DataFrame(records, columns=colnames)
    return df

# Fetch standard data as pandas DataFrame
@database_connection
def fetch_standard_data_as_dataframe(cur, instrument_name):
    query = "SELECT * FROM standard_parameters WHERE instrument = %(instrument_name)s"
    with cur.connection.cursor() as cursor:
        cursor.execute(query, {"instrument_name": instrument_name})
        colnames = [desc[0] for desc in cursor.description]
        records = cursor.fetchall()
    df = pd.DataFrame(records, columns=colnames)
    return df

# Fetch intensity ratio fit data as pandas DataFrame
@database_connection
def fetch_intensity_ratio_fit_pars_as_dataframe(cur, instrument_name):
    query = "SELECT * FROM intensity_ratio_fit_pars WHERE instrument = %(instrument_name)s"
    with cur.connection.cursor() as cursor:
        cursor.execute(query, {"instrument_name": instrument_name})
        colnames = [desc[0] for desc in cursor.description]
        records = cursor.fetchall()
    df = pd.DataFrame(records, columns=colnames)
    return df