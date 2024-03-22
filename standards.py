# stantards.py
import pandas as pd
from plotly import subplots
import plotly.graph_objects as go
import plotly.express as px


def numeric_columns_plots(df: pd.DataFrame):
  """
  Generates plots for numeric columns
  Parameters:
    df: Pandas DataFrame
  Returns:
    figs: List of plotly figures for all standard columns vs
    standard weight (mg)
  """

  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
  
  # Numeric columns to plot
  numeric_columns = ['1  Cycle Int  Samp  44', '1  Cycle Int  Ref  44', 
                    '1  Cycle Int  Samp  45', '1  Cycle Int  Ref  45', 
                    'd 45CO2/44CO2  Std Dev', 'd 46CO2/44CO2  Std Dev', 
                    'd 47CO2/44CO2  Std Dev', 'd 48CO2/44CO2  Std Dev', 
                    'd 49CO2/44CO2  Std Dev', 'd 13C/12C  Std Dev', 
                    'd 18O/16O  Std Dev']

  # Define the height of each subplot (in pixels)
  subplot_height = 300
  length = len(numeric_columns)

  # Create a subplot for each column in the DataFrame
  fig = subplots.make_subplots(rows=length//2 + length%2, cols=2, vertical_spacing=0.08, horizontal_spacing=0.15)

  # Set the height of the figure
  fig.update_layout(height=(length//2 + length%2)*subplot_height, showlegend=False)

  # Get unique identifiers in the filtered data
  identifiers = df["Identifier 1"].unique()

  # Iterate through each column and create a scatter plot
  for i, column in enumerate(numeric_columns):
      for identifier in identifiers:
          identifier_data = df[df["Identifier 1"] == identifier]

          # Get the color and marker for the identifier
          color, marker = standard_marker_color(identifier)

          fig.add_trace(
              go.Scatter(x=identifier_data.index, 
                      y=identifier_data[column], 
                      mode='markers', 
                      name=f"{identifier}",  # Set the name of the data series,
                      marker=dict(color=color, symbol=marker),  # Use the color and marker corresponding to the identifier
                      customdata=identifier_data[["Identifier 1", "Line", "Weight"]].values,  # Add Time Code & Identifier to customdata
                      hovertemplate=
                          f"<b>{column}</b>: %{{y:.1f}}<br>" +
                          "<b>Standard</b>: %{customdata[0]}<br>" +
                          "<b>Line</b>: %{customdata[1]}<br>" +
                          "<b>Weight (mg)</b>: %{customdata[2]}<br>" +
                          "<extra></extra>"              
                      ),
              row = i//2 + 1,
              col = i%2 + 1
          )

      # Set the title of the y-axis for each subplot
      fig.update_yaxes(title_text=column, row=i//2 + 1, col=i%2 + 1)
      fig.update_xaxes(title_text="", row=i//2 + 1, col=i%2 + 1)

  return fig


def isotope_std_plots(df: pd.DataFrame):
  """
  Generates plots for numeric columns
  Parameters:
  df: Pandas DataFrame
  Returns:
  figs: List of plotly figures for all standard columns vs
  datetime (Time Code)
  """

  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

  # Define the group columns to plot
  group_columns = [
      'Weight', '1  Cycle Int  Samp  44',
      'd 45CO2/44CO2  Std Dev',
      'd 46CO2/44CO2  Std Dev',
      'd 47CO2/44CO2  Std Dev',
      'd 48CO2/44CO2  Std Dev',
      'd 49CO2/44CO2  Std Dev', 
      'd 13C/12C  Std Dev',
      'd 18O/16O  Std Dev'
  ]

  # Create a new subplots figure
  # Create a subplot for each column in the DataFrame
  # Define the height of each subplot (in pixels)
  subplot_height = 400
  length = len(group_columns)
  fig_stds = subplots.make_subplots(rows=length,
                                    cols=1,
                                    horizontal_spacing=0.15)

  # Set the height of the figure
  fig_stds.update_layout(height=(length) * subplot_height,
                         showlegend=False,
                         legend=dict())

  # Get standards list from Identifier 1
  stds = df["Identifier 1"].unique()

  # Create a color mapping based on unique identifiers
  unique_identifiers = df['Identifier 1'].unique()
  colors = px.colors.qualitative.Plotly  # Using Plotly's qualitative colors
  # Ensure we have enough colors for identifiers by looping through them if necessary
  color_map = {
      identifier: colors[i % len(colors)]
      for i, identifier in enumerate(unique_identifiers)
  }

  # Iterate through each group column
  for i, group_column in enumerate(group_columns):
    # Iterate through each identifier1 group
    for identifier1_key in stds:
      # Filter data for the current group_column and identifier1_key
      filtered_data = df[(df["Identifier 1"] == identifier1_key)]

      # Use the color map to get the consistent color for the identifier
      identifier_color = color_map[identifier1_key]

      # Create a scatter plot trace with markers and customdata
      customdata = list(
          zip(identifier1_key, filtered_data["Weight"],
              filtered_data["1  Cycle Int  Samp  44"], filtered_data["Line"]))

      # Add the trace to the subplots figure
      fig_stds.add_trace(
          go.Scatter(
              x=filtered_data.index,
              y=filtered_data[group_column],
              name=identifier1_key,
              mode="markers",  # Set mode to 'markers' to remove lines
              marker=dict(
                  size=10, opacity=0.8,
                  color=identifier_color),  # Customize marker appearance
              customdata=customdata,  # Add customdata for hover information
              hovertemplate="<b>Init. intensity (mV): %{customdata[2]}</b><br>"
              + "<b>Weight (mg): %{customdata[1]:.1f}</b><br>" +
              "<b>Line: %{customdata[3]}</b><br>"  #+
              # "<extra></extra>",  # Hide the extra hover information
          ),
          row=i + 1,
          col=1  # i%2 + 1
      )

    # Update y-axis title
    fig_stds.update_yaxes(title_text=group_column, row=i + 1, col=1)

  return fig_stds

# Function for Identifier coloring
def standard_marker_color(identifier):
    # Define colors, markers, and identifiers
    colors = ['green', 'violet', 'violet', 
              'blue', 'red', 'lightblue',
              'orange', 'green', 'green',
              'red', 'red', 'blue',
              'blue', 'lightblue', 'orange', "green",
              'green', 'violet', 'violet', 'orange',
              'violet', 'green', 'red',
              'blue', 'green', 'lightblue'
              ]

    markers = ['triangle-up', 'circle', 'triangle-down',
           'circle', 'triangle-up', 'circle',
           'circle', 'circle', 'triangle-down',
           'square', 'triangle-down', 'square',
           'triangle-down', 'circle', 'triangle-up', 'square',
           'triangle-up', 'triangle-up', 'square', 'cross',
           'circle-open', 'triangle-down-open', 'square-open',
           'square-open', 'triangle-up-open', 'circle'
           ]
    
    identifiers = ['Carrara', 'CHALK', 'CHALK_new aliqu', 
                   'Equ Gas 25C', 'Fast Haga', 'Heated gas', 
                   'IAEA C1', 'IAEA C2', 'IAEA C2_new ali', 
                   'ISO A', 'Isolab A_new al', 'ISO B', 
                   'ISO B_new aliq', 'Merck', 'NBS18', 'NBS19', 
                   'Riedel', 'Speleo 2-8E', 'Speleo 9-25G', 'UN_CM12', 
                   'CHALK_2', '_IAEA C2_2', '_Isolab A 2', 
                   'ISO B_2', '_Riedel 2', 'MERCK'
                   ] 

    # Print the identifier that the function is trying to find
    # print(f"Looking for identifier: {identifier}")

    # Create dictionaries that map each identifier to a color and a marker
    color_dict = {identifier: colors[i % len(colors)] for i, identifier in enumerate(identifiers)}
    marker_dict = {identifier: markers[i % len(markers)] for i, identifier in enumerate(identifiers)}

    # Return the color and marker corresponding to the given identifier
    # Return 'black' and 'circle' if the identifier is not found
    return color_dict.get(identifier, 'black'), marker_dict.get(identifier, 'circle')