# kiel.py
import re
import pandas as pd
import plotly.graph_objects as go
from plotly import subplots
# Import the standard_marker_color() function from standards.py
from standards import standard_marker_color


def get_kiel_data(df: pd.DataFrame):
    """
    Extract specific information from a DataFrame column using regular expressions and create a new DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    info_keys (list): A list of keys for the information to be extracted.
    key_re_dict (dict): A dictionary of regular expressions for each key.

    Returns:
    pandas.DataFrame: A new DataFrame with the extracted information.
    """

    # Ensure we have a pd.Dataframe object
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

    col_rename_dict = {
      "Acid": "acid_temperature",
      "LeakRate": "leakrate",
      "P no Acid": "p_no_acid",
      "P gases": "p_gases",
      "RefRe": "reference_refill",
      "Total CO2": "total_CO2",
      "VM1 aftr Trfr.": "vm1_after_transfer",
      "Init int": "initial_intensity",
      "Bellow Pos": "bellow_position",
      "RefI": "reference_intensity",
      "RefPos": "reference_bellow_position"
    }

    info_keys = [
        "Acid", "LeakRate", "P no Acid", "P gases", "RefRe",
        "Total CO2", "VM1 aftr Trfr.", "Init int", "Bellow Pos", "RefI", "RefPos"
    ]
    key_re_dict = {
        "Acid": "\s?:\s+([\d.]+)",
        "LeakRate": "\s?:\s+([\d.]+)",
        "P no Acid": "\s?:\s+([\d.]+)",
        "P gases": "\s?:\s+([\d.]+)",
        "Total CO2": "\s?:\s+([\d.]+)",
        "Init int": "\s?:\s+([\d.]+)",
        "VM1 aftr Trfr.": "\s?:\s+([\d.]+)",
        "Bellow Pos": "\s?:\s+([\d.]+)",
        "RefRe": "\s?:\s+R\s+mBar\s+([\d.]+)",
        "RefI": "\s?:\s+mBar\s+r\s+([\d.]+)\s+pos\s+r\s+[\d.]+"
    }

    extracted_values = {key: [] for key in info_keys}
    # List for Kiel Line and Time Code
    time_codes = []  # list for storing Time Code values

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        row_data = {}
        for key, value in key_re_dict.items():
            pattern = f'{key}{value}'
            match = re.search(pattern, row["Information"])
            if match:
                if key != "RefI":
                    row_data[key] = match.group(1)
                else:
                    tmp = match[0].split(" ")
                    row_data["RefI"] = tmp[3]
                    row_data["RefPos"] = tmp[7]
            else:
                row_data[key] = None

        for key in info_keys:
            extracted_values[key].append(row_data.get(key, None))

        # Append Time Code for each row
        time_codes.append(row["Time Code"])

    df_kiel_par_tmp = pd.DataFrame(extracted_values)
    # Add 'Time Code' & 'Line' columns from df
    df_kiel_par_tmp["time"] = time_codes

    # Convert all columns of DataFrame to numeric, except for Time Code
    numeric_cols = df_kiel_par_tmp.columns.drop('time')
    df_kiel_par_tmp[numeric_cols] = df_kiel_par_tmp[numeric_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce'))

    # rename cols
    df_kiel_par_tmp.rename(columns=col_rename_dict, inplace=True)

    # Joing df_kiel_par and df
    # Join df_kiel_par and df_std_cp DataFrames ingoring index
    df_kiel_par = df.join(df_kiel_par_tmp.set_index('time'), on='Time Code', how='inner')
    
    # Drop Background column from df_kiel_par
    if "Background" in df_kiel_par.columns:
        df_kiel_par.drop(columns=["Background"], inplace=True)
    else:
        pass

    # Set 'Time Code' as index
    df_kiel_par.set_index('Time Code', inplace=True)

    return df_kiel_par


def generate_kiel_plots(df: pd.DataFrame) -> go.Figure:
  """
    Generate plots for a DataFrame containing kiel data.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    fig (plotly.graph_objects.Figure): The figure containing the plots.
    """
  
  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
  
  # Set subplot titles
  cols = ['acid_temperature', 'leakrate', 'p_no_acid', 'p_gases',
        'reference_refill', 'total_CO2', 'vm1_after_transfer',
        'initial_intensity', 'reference_intensity',
        'reference_bellow_position']

  # Custom dictionary of text per def_kiel_par column
  kiel_par_dict = {
          "acid_temperature": "Acid temperature [°C]",
          "leakrate": "Leak Rate [mbar/min]",
          "p_no_acid": "P no Acid [mbar]",
          "p_gases": "P gases [mbar]",
          "total_CO2": "Total CO2 [mbar]",
          "vm1_after_transfer": "VM1 aftr CO2 Transfer. [mbar]",
          "initial_intensity": "Initial Intensity [mV]",
          # "bellow_position": "Bellow Compression [%]",
          "reference_refill": "Reference Refill [mbar]",
          "reference_intensity": "Ref Bellow Pressure [mbar]",
          "reference_bellow_position": "Ref Bellow Compression [%]",
          }

  # define subplot height and length 
  length = len(cols)
  subplot_height = 300

  # Create a subplot with a 4x3 grid
  fig = subplots.make_subplots(rows=length//2 + length%2, 
                      cols=2, 
                      subplot_titles=[kiel_par_dict[col] for col in cols],
                      vertical_spacing=0.1,)

  # Set the height of the figure
  fig.update_layout(height=(length//2 + length%2)*subplot_height, 
                    showlegend=False,
                    autosize=True)

  # Get unique identifiers in the filtered data
  identifiers = df["Identifier 1"].unique()

  # Iterate through the columns in df_kiel_par and create separate plots
  for i, column in enumerate(cols):
      # Iterate through the identifiers and create a scatter plot for each
      for identifier in identifiers:
          # print(f"Line: {line}, Column: {column}, Identifier: {identifier}")
          identifier_data = df[df["Identifier 1"] == identifier]

          # Get the color and marker for the identifier
          color, marker = standard_marker_color(identifier)

          # print(f"Identifier: {identifier}, Color: {color}, Marker: {marker}")

          # Create a scatter plot trace with custom colors and hover template
          fig.add_trace(
                  go.Scatter(x=identifier_data.index, 
                              y=identifier_data[column], 
                              mode='markers', 
                              name=f"{identifier}",  # Set the name of the data series,
                              marker=dict(color=color, symbol=marker),  # Use the color and marker corresponding to the identifier
                              # marker=dict(color='green', symbol='circle'),
                              customdata=identifier_data[["Identifier 1", "Line"]].assign(TimeCode=identifier_data.index).values,  # Add Time Code to customdata
                              hovertemplate=
                                  f"<b>{column}</b>: %{{y:.1f}}<br>" +
                                  "<b>Line</b>: %{customdata[1]}<br>" +
                                  "<b>Datetime</b>: %{customdata[2]}<br>" +
                                  "<b>Standard</b>: %{customdata[0]}<br>" +
                                  "<extra></extra>"
                      ),
              row = i//2 + 1,
              col = i%2 + 1
          )

  # Show the subplot
  return fig