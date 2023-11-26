# kiel.py
import re
import pandas as pd
import plotly.graph_objects as go
from plotly import subplots


def get_kiel_data(df: pd.DataFrame) -> pd.DataFrame:
  """
    Extract specific information from a DataFrame column using regular expressions and create a new DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    info_keys (list): A list of keys for the information to be extracted.
    key_re_dict (dict): A dictionary of regular expressions for each key.

    Returns:
    pandas.DataFrame: A new DataFrame with the extracted information.
    """

  info_keys = [
      "Acid", "LeakRate", "x drops", "P no Acid", "P gases", "RefRe",
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

  for row in df["Information"]:
    row_data = {}
    for key, value in key_re_dict.items():
      pattern = f'{key}{value}'
      match = re.search(pattern, row)
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

  df_kiel_par = pd.DataFrame(extracted_values)
  df_kiel_par["Line"] = df["Line"]

  # convert all columns of DataFrame
  df_kiel_par = df_kiel_par.apply(pd.to_numeric)

  # Ensure that dataframe to be returned is a dataframe
  assert isinstance(df_kiel_par,
                    pd.DataFrame), "df_kiel_par must be a DataFrame"

  return df_kiel_par


def generate_kiel_plots(df: pd.DataFrame) -> go.Figure:
  """
    Generate plots for a DataFrame containing kiel data.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    fig (plotly.graph_objects.Figure): The figure containing the plots.
    """
  # Set subplot titles
  cols = [
      'Acid', 'LeakRate', 'P no Acid', 'P gases', 'RefRe', 'Total CO2',
      'VM1 aftr Trfr.', 'Init int', 'Bellow Pos', 'RefI', 'RefPos', 'x drops'
  ]

  # Custom dictionary of text per def_kiel_par column
  kiel_par_dict = {
      "Acid": "Acid temperature [°C]",
      "LeakRate": "Leak Rate [mbar/min]",
      "P no Acid": "P no Acid [mbar]",
      "P gases": "P gases [mbar]",
      "Total CO2": "Total CO2 [mbar]",
      "VM1 aftr Trfr.": "VM1 aftr CO2 Transfer. [mbar]",
      "Init int": "Init Intensity [mV]",
      "Bellow Pos": "Bellow Compression [%]",
      "RefRe": "RefRe",
      "RefI": "Ref Bellow Pressure [mbar]",
      "RefPos": "Ref Bellow Compression [%]",
      "x drops": "x drops [count]"
  }

  # define subplot height and length
  length = len(cols)
  subplot_height = 200

  # Create a subplot with a 4x3 grid
  fig = subplots.make_subplots(
      rows=length // 2 + length % 2,
      cols=2,
      subplot_titles=[kiel_par_dict[col] for col in cols],
      vertical_spacing=0.08,
  )

  # Set the height of the figure
  fig.update_layout(height=(length // 2 + length % 2) * subplot_height,
                    showlegend=False,
                    autosize=True)

  # Initialize variables for row and column
  # row, col = 1, 1

  # Get unique values in the "Line" column
  lines = df["Line"].unique()

  # Define colors for each line
  colors = ['blue', 'red']  # Add more colors if there are more than two lines

  # Iterate through the columns in df_kiel_par and create separate plots
  for i, column in enumerate(cols):
    for j, line in enumerate(lines):
      filtered_data = df[df["Line"] == line]

      # Create a scatter plot trace with custom colors and hover template
      fig.add_trace(
          go.Scatter(
              x=filtered_data[filtered_data["Line"] == line].index,
              y=filtered_data[filtered_data["Line"] == line][column],
              mode='markers',
              name=f"Line {line}",  # Set the name of the data series,
              marker=dict(color=colors[
                  j % len(colors)]),  # Use the color corresponding to the line
              hovertemplate=f"<b>{column}</b>: %{{y:.1f}}<br>" +
              "<b>Line</b>: " + str(line) + "<br>" +
              "<extra></extra>",  # Hide the extra hover information
          ),
          row=i // 2 + 1,
          col=i % 2 + 1)

  # Show the subplot
  return fig
