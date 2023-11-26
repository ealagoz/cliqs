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
  # numeric columns
  numeric_columns = df.select_dtypes(include=["float64", "Int64"]).columns

  # Define the height of each subplot (in pixels)
  subplot_height = 300
  length = len(numeric_columns[5:30])

  # Create a subplot for each column in the DataFrame
  fig = subplots.make_subplots(rows=length // 2 + length % 2,
                               cols=2,
                               vertical_spacing=0.03,
                               horizontal_spacing=0.15)

  # Set the height of the figure
  fig.update_layout(height=(length // 2 + length % 2) * subplot_height,
                    showlegend=False)

  # Get unique values in the "Line" column
  lines = df["Line"].unique()

  # Define colors for each line
  colors = ['blue', 'red']  # Add more colors if there are more than two lines

  # Iterate through each column and create a scatter plot
  for i, column in enumerate(numeric_columns[5:30]):
    for j, line in enumerate(lines):
      fig.add_trace(
          go.Scatter(
              x=df[df["Line"] == line]["Weight"],
              y=df[df["Line"] == line][column],
              mode='markers',
              name=f"Line {line}",  # Set the name of the data series,
              marker=dict(color=colors[
                  j % len(colors)]),  # Use the color corresponding to the line
              hovertemplate=f"<b>{column}</b>: %{{y:.1f}}<br>" +
              "<b>Weight (mg)</b>: %{x:.1f}<br>" + "<b>Line</b>: " +
              str(line) + "<br>" +
              "<extra></extra>",  # Hide the extra hover information
          ),
          row=i // 2 + 1,
          col=i % 2 + 1)

    # Set the title of the y-axis for each subplot
    fig.update_yaxes(title_text=column, row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(title_text="Weight (mg)", row=i // 2 + 1, col=i % 2 + 1)

  return fig


def create_scatter_plots(df: pd.DataFrame):
  """
  Generates plots for numeric columns
  Parameters:
  df: Pandas DataFrame
  Returns:
  figs: List of plotly figures for all standard columns vs
  datetime (Time Code)
  """
  group_columns = [
      'Weight', '1  Cycle Int  Samp  44', 'd 45CO2/44CO2  Mean',
      'd 45CO2/44CO2  Std Dev', 'd 46CO2/44CO2  Mean',
      'd 46CO2/44CO2  Std Dev', 'd 47CO2/44CO2  Mean',
      'd 47CO2/44CO2  Std Dev', 'd 48CO2/44CO2  Mean',
      'd 48CO2/44CO2  Std Dev', 'd 49CO2/44CO2  Mean',
      'd 49CO2/44CO2  Std Dev', 'd 13C/12C  Mean', 'd 13C/12C  Std Dev',
      'd 18O/16O  Mean', 'd 18O/16O  Std Dev'
  ]

  figures = {}

  for group_column in group_columns:
    fig_ = go.Figure()

    for identifier1_key in df["Identifier 1"].unique():
      filtered_data = df[(df["Identifier 1"] == identifier1_key)]
      customdata = list(
          zip(filtered_data["Weight"], filtered_data["1  Cycle Int  Samp  44"],
              filtered_data["Line"]))
      trace = go.Scatter(
          x=filtered_data["Time Code"],
          y=filtered_data[group_column],
          name=identifier1_key,
          mode="markers",
          marker=dict(size=10, opacity=0.8),
          customdata=customdata,
          hovertemplate="<b>Init. intensity (mV): %{customdata[1]}</b><br>" +
          "<b>Weight (mg): %{customdata[0]:.1f}</b><br>" +
          "<b>Line: %{customdata[2]}</b><br>" +
          "<extra></extra>",  # Hide the extra hover information
      )

      fig_.add_trace(trace)

    fig_.update_layout(title=f"<b>{group_column}</b>",
                       xaxis_title="",
                       yaxis_title=group_column,
                       showlegend=True)

    figures[group_column] = fig_

  return figures


def isotope_std_plots(df: pd.DataFrame):
  """
  Generates plots for numeric columns
  Parameters:
  df: Pandas DataFrame
  Returns:
  figs: List of plotly figures for all standard columns vs
  datetime (Time Code)
  """
  # Define the group columns to plot
  group_columns = [
      'Weight', '1  Cycle Int  Samp  44', 'd 45CO2/44CO2  Mean',
      'd 45CO2/44CO2  Std Dev', 'd 46CO2/44CO2  Mean',
      'd 46CO2/44CO2  Std Dev', 'd 47CO2/44CO2  Mean',
      'd 47CO2/44CO2  Std Dev', 'd 48CO2/44CO2  Mean',
      'd 48CO2/44CO2  Std Dev', 'd 49CO2/44CO2  Mean',
      'd 49CO2/44CO2  Std Dev', 'd 13C/12C  Mean', 'd 13C/12C  Std Dev',
      'd 18O/16O  Mean', 'd 18O/16O  Std Dev'
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
              x=filtered_data["Time Code"],
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
