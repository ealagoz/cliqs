# raw_intensity
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots
from scipy import stats


def group_standard_intensities(df: pd.DataFrame):
  """
  Groups the raw intensities by standard.

  Parameters
  ----------
  df_intensity : pd.DataFrame
    The raw intensities dataframe.
  Returns
  -------
  grouped_data : pd.DataFrame
    The raw intensities grouped by standard.
  """
  group_columns = ["Identifier 1", "Is Ref _", "Time Code"]
  # Group the data by the group_columns
  grouped_data = df.groupby(group_columns,
                            as_index=False).agg(lambda x: list(x))
  # Rename the columns
  grouped_data.columns = [
      "_".join(col).strip() if isinstance(col, tuple) else col
      for col in grouped_data.columns.values
  ]

  # Ensure that dataframe to be returned is a dataframe
  assert isinstance(grouped_data,
                    pd.DataFrame), "df_kiel_par must be a DataFrame"

  return grouped_data


def raw_standard_intensity_stats_plots(df: pd.DataFrame):
  """
  Generates plots for the raw standard intensity stats.
    Parameters
    ----------
    grouped_data : pd.DataFrame
    The sorted raw intensities dataframe.
    Returns
    -------
    figs : dict
    A dictionary of plotly figures
    intensity_stats: pd.Dataframe.
    """
  # Get the list of intensity columns
  intensity_cols = [col for col in df.columns if col.startswith("rIntensity")]

  # Empty list to store intensity stats
  intensity_data = []

  # Iterate through intensity columns
  for intensity_column in intensity_cols:
    # Iterate through the rows of the DataFrame
    for identifier, is_ref, time_code, values in zip(df['Identifier 1'],
                                                     df['Is Ref _'],
                                                     df["Time Code"],
                                                     df[intensity_column]):
      # Calculate mean and standard deviation for values per identifier
      mean_intensity = np.mean(values)
      std_dev_intensity = np.std(values)

      # Use Reference and Sample for Is Ref values in the dictionary
      is_ref_labels = {0: "Sample", 1: "Reference"}
      is_ref = is_ref_labels[is_ref]

      # Append the results to the intensity_data list
      intensity_data.append({
          "Identifier": identifier,
          "Is Ref": is_ref,
          "Datetime": time_code,
          "Intensity": intensity_column,
          "Mean": mean_intensity,
          "Std_Dev": std_dev_intensity
      })
  # Create DataFrame outside of the loop
  intensity_stats = pd.DataFrame(intensity_data,
                                 columns=[
                                     "Identifier", "Is Ref", "Datetime",
                                     "Intensity", "Mean", "Std_Dev"
                                 ])

  # Plot mean and standard deviation per intensity column
  fig = px.scatter(intensity_stats,
                   x="Identifier",
                   y="Mean",
                   color="Intensity",
                   size="Std_Dev",
                   hover_data=["Is Ref", "Datetime", "Std_Dev"],
                   title="Intensity mean and standard deviation",
                   size_max=15)

  # Update layout
  fig.update_layout(xaxis_title="",
                    yaxis_title="Intensity [mV]",
                    legend_title="Intensity",
                    showlegend=True)

  # Update hovertemplate
  fig.update_traces(hovertemplate=("<b></b>%{x}<br>" +
                                   "<b>Type</b>: %{customdata[0]}<br>" +
                                   "<b>Mean</b>: %{y:.2f}<br>" +
                                   "<b>Std</b>: %{customdata[2]:.2f}<br>" +
                                   "<b>Datetime</b>: %{customdata[1]}<br>"))

  # Ensure that dataframe to be returned is a dataframe
  assert isinstance(intensity_stats,
                    pd.DataFrame), "df_kiel_par must be a DataFrame"

  return intensity_stats, fig


def raw_standard_intensity_plots(df: pd.DataFrame):
  """
  Generates scatter plots of raw intensity values for each Identifier.
  Parameters
  ----------
  df : pd.DataFrame
  A dataframe containing raw intensity values for each Identifier.
  Returns
  -------
  fig : plotly.graph_objects.Figure
  A plotly figure containing scatter plots of raw intensity values for each Identifier.
    """
  # Get unique identifiers
  unique_identifiers = df['Identifier 1'].unique()

  # Create a subplot for each unique identifier
  fig = subplots.make_subplots(rows=len(unique_identifiers),
                                cols=1,
                                subplot_titles=unique_identifiers,
                              )
  
  # Create a separate figure for each unique identifier
  for identifier in unique_identifiers:
    # Filter DataFrame for the current identifier
    subset_df = df[df['Identifier 1'] == identifier]

    # Create an empty figure
    # fig = go.Figure()

    # Iterate through the rows of the subset DataFrame
    for is_ref, values, time_code in zip(subset_df['Is Ref _'],
                                         subset_df['rIntensity 44'],
                                         subset_df['Time Code']):
      # Create a line plot for each row
      legend_group = f'Is Ref _ {is_ref}'
      legend_name = 'Reference' if is_ref == 1 else 'Sample'
      # Create a line plot for each row
      fig.add_trace(
          go.Scatter(
              x=list(range(len(values))),
              y=values,
              mode='lines+markers',
              name=f'{identifier} - {legend_name}',
              legendgroup=legend_group,
              text=[f'Value: {v:.2f}' for v in values],
              hoverinfo='text',
              hovertemplate=f"<b>Identifier</b>: {identifier}<br>" +
              f"<b>Measurement type</b>: {legend_name}<br>" +
              "<b>Intensity [mV]</b>: %{y:.2f}<br>" +
              "<b>Cycle</b>: %{x}<br>" + f"<b>Datetime</b>: {time_code}<br>" +
              "<extra></extra>",  # Hide the extra hover information
          ),
          row=unique_identifiers.tolist().index(identifier) + 1,
          col=1)

  # Customize the layout of the plot
  fig.update_layout(height=400*len(unique_identifiers),
                    title='Raw intensity m44 per standard',
                    xaxis_title='Cycle',
                    yaxis_title='Raw intensity m44',
                    showlegend=False)
  
  return fig


def raw_standard_intensity_ratio_plots(df: pd.DataFrame):
  """
  Generates plots for the raw intensity ratio stats.
  Parameters
  ----------
  grouped_data : pd.DataFrame
  A sorted raw intensities dataframe.
  Returns
  -------
  figs : dict
  A dictionary of plotly figures
  df_raw_int_ratio: pd.Dataframe.
  """
  # Intensity columns except for rIntensity 44
  intensity_cols = [
      col for col in df.columns
      if col.startswith("rIntensity") and col != "rIntensity 44"
  ]

  # Generate empty dataframe to store results
  df_raw_int_ratio = df.copy()

  # Generate ratio_columns
  ratio_columns = [
      f"{intensity_column}_ratio" for intensity_column in intensity_cols
  ]

  # Calculate intensity ratios
  for intensity_column, ratio_column in zip(intensity_cols, ratio_columns):

    # Update the existing column with new values
    df_raw_int_ratio[ratio_column] = df_raw_int_ratio.apply(
        lambda row:
        [a / b for a, b in zip(row[intensity_column], row['rIntensity 44'])],
        axis=1)

  # Reshape the DataFrame for plotting
  df_raw_int_ratio = df_raw_int_ratio.melt(
      id_vars=['Identifier 1', 'Is Ref _', 'Time Code'],
      value_vars=[f"{col}_ratio" for col in intensity_cols],
      var_name='Intensity',
      value_name='Ratio')

  # Flatten the lists in the 'Ratio' column
  df_raw_int_ratio = df_raw_int_ratio.explode('Ratio')

  # Convert 'Time Code' to datetime format
  df_raw_int_ratio["Time Code"] = pd.to_datetime(df_raw_int_ratio["Time Code"])

  # Map 'Is Ref' values to 'Sample' and 'Reference'
  df_raw_int_ratio['Is Ref _'] = df_raw_int_ratio['Is Ref _'].map({
      0: "Reference",
      1: "Sample"
  })

  # Introduce jitter: add random time delta in the range of seconds
  # using pandas.Timedelta and numpy.random.uniform
  # Replace 'Time Code' with whatever unit of time you want the jitter
  jitter_seconds = 600  # Set the maximum number of seconds for jitter
  df_raw_int_ratio['Jittered Time Code'] = df_raw_int_ratio.apply(
      lambda x: x['Time Code'] + pd.Timedelta(seconds=np.random.uniform(
          -jitter_seconds, jitter_seconds)),
      axis=1)

  # Create the scatter plot
  fig = px.scatter(
      df_raw_int_ratio,
      x='Jittered Time Code',
      y='Ratio',
      color='Identifier 1',
      hover_data=['Identifier 1', 'Is Ref _', 'Intensity', 'Ratio'],
      title="Intensity ratios per Identifier",
      opacity=0.5,  # Set opacity to 0.5 (50%) to see overlapping points
      size_max=15)

  fig.update_layout(xaxis_title=" ",
                    yaxis_title="Intensity Ratio",
                    legend_title="Standard",
                    showlegend=True)

  # Update hovertemplate
  fig.update_traces(hovertemplate=("<b></b>%{customdata[0]}<br>" +
                                   "<b>Is Ref</b>: %{customdata[1]}<br>" +
                                   "<b>Intensity</b>: %{customdata[2]}<br>" +
                                   "<b>Ratio</b>: %{y}<br>" +
                                   "<extra></extra>"))

  # Ensure that dataframe to be returned is a dataframe
  assert isinstance(df_raw_int_ratio,
                    pd.DataFrame), "df_kiel_par must be a DataFrame"

  return df_raw_int_ratio, fig


def raw_standard_intensity_ratio_fit_plots(df: pd.DataFrame):
  """
  Generates scatter plots of raw intensity ratios and fitted intensity ratios.
  Parameters
  ----------
  df : pd.DataFrame
  A dataframe containing raw intensity ratios and fitted intensity ratios.
  Returns
  -------
  fig : plotly.graph_objects.Figure
  A plotly figure object.
  """

  # Create a 2x1 subplot layout
  fig = subplots.make_subplots(rows=2, cols=1)

  # Placeholder to store slope and r-squared values
  identifier_stats = []

  # Create a color mapping based on unique identifiers
  unique_identifiers = df['Identifier 1'].unique()
  colors = px.colors.qualitative.Plotly  # Using Plotly's qualitative colors
  # Ensure we have enough colors for identifiers by looping through them if necessary
  color_map = {
      identifier: colors[i % len(colors)]
      for i, identifier in enumerate(unique_identifiers)
  }

  for identifier in df['Identifier 1'].unique():
    for is_ref in df['Is Ref _'].unique():
      # Filter the dataframe for each identifier and Is Ref
      id_ref_group = df[(df['Identifier 1'] == identifier)
                        & (df['Is Ref _'] == is_ref)]

      # Use the color map to get the consistent color for the identifier
      identifier_color = color_map[identifier]

      # Extract x and y values for regression
      x = id_ref_group.index.values
      y = id_ref_group['Ratio'].astype(
          float)  # Ensure that ratios are floats for regression
      # Perform linear regression

      slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

      # Rˆ2 value
      r_squared = r_value**2

      # Store identifier, is_ref, slope, and r-squared in the list
      identifier_stats.append([identifier, is_ref, slope, r_squared])

      # Add scatterplot points for slope
      fig.add_trace(go.Scatter(
          x=[identifier],
          y=[slope],
          marker=dict(color=identifier_color),
          customdata=[[identifier, is_ref, slope]],
          hovertemplate=("<b>Identifier</b>: %{customdata[0]}<br>" +
                         "<b>Is Ref</b>: %{customdata[1]}<br>" +
                         "<b>Slope</b>: %{customdata[2]:.6f}<br>" +
                         "<extra></extra>")),
                    row=1,
                    col=1)
      # Add scatterplot points for Rˆ2
      fig.add_trace(go.Scatter(
          x=[identifier],
          y=[r_squared],
          marker=dict(color=identifier_color),
          customdata=[[identifier, is_ref, r_squared]],
          hovertemplate=("<b>Identifier</b>: %{customdata[0]}<br>" +
                         "<b>Is Ref</b>: %{customdata[1]}<br>" +
                         r"<b>$R^{2}$</b>: %{customdata[2]:.6f}<br>" +
                         "<extra></extra>")),
                    row=2,
                    col=1)

  # Update xaxis and yaxis properties if needed
  fig.update_xaxes(title_text="", type='category', row=1, col=1)
  fig.update_yaxes(title_text="Slope", row=1, col=1)
  fig.update_xaxes(title_text="", type='category', row=2, col=1)
  fig.update_yaxes(title_text=r"$R^{2}$", row=2, col=1)

  # Update layout if necessary, e.g., adding titles, adjusting height, etc.
  fig.update_layout(
      height=1200,
      title_text=
      r"Intensity cycles linear regression slope and $R^{2}$ per Standard",
      showlegend=False)

  return fig
