# raw_intensity
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

  group_columns = ["Identifier 1", "Is Ref _", "Time Code"]
  # Group the data by the group_columns
  grouped_data = df.groupby(group_columns,
                            as_index=False).agg(lambda x: list(x))
  # Rename the columns
  grouped_data.columns = [
      "_".join(col).strip() if isinstance(col, tuple) else col
      for col in grouped_data.columns.values
  ]

  # Transform group_data to df_raw_int_ratio
  # Intensity columns except for rIntensity 44
  intensity_cols = [col for col in grouped_data.columns if col.startswith("rIntensity") and col != "rIntensity 44"]

  # Generate empty dataframe to store results
  df_raw_int_ratio = grouped_data.copy()

  # Generate ratio_columns
  ratio_columns = [f"{intensity_column}_ratio" for intensity_column in intensity_cols]

  # Calculate intensity ratios
  for intensity_column, ratio_column in zip(intensity_cols, ratio_columns):
  
      # Update the existing column with new values
      df_raw_int_ratio[ratio_column] = df_raw_int_ratio.apply(lambda row: [a / b for a, b in zip(row[intensity_column], row['rIntensity 44'])], axis=1)

  # Reshape the DataFrame for plotting
  df_raw_int_ratio = df_raw_int_ratio.melt(id_vars=['Identifier 1', 'Is Ref _', 'Time Code'],
                            value_vars= [f"{col}_ratio" for col in intensity_cols],
                            var_name='Intensity', value_name='Ratio')

  # Flatten the lists in the 'Ratio' column
  df_raw_int_ratio = df_raw_int_ratio.explode('Ratio')

  # Map 'Is Ref' values to 'Sample' and 'Reference'
  df_raw_int_ratio['Is Ref _'] = df_raw_int_ratio['Is Ref _'].map({0: "Reference", 1: "Sample"})

  # Convert column dtypes to appropriate types
  df_raw_int_ratio["Time Code"] = pd.to_datetime(df_raw_int_ratio["Time Code"])
  df_raw_int_ratio["Intensity"] = df_raw_int_ratio["Intensity"].astype("string")
  df_raw_int_ratio["Ratio"] = df_raw_int_ratio["Ratio"].astype("float")
  # Ensure that 'Identifier' and 'Is Ref_' are categorical
  df_raw_int_ratio['Identifier 1'] = df_raw_int_ratio['Identifier 1'].astype('category')
  df_raw_int_ratio['Is Ref _'] = df_raw_int_ratio['Is Ref _'].astype('category')
  df_raw_int_ratio['Time Code'] = df_raw_int_ratio['Time Code'].astype('datetime64[ns]')

  # Create new columns in df_raw_int_ratios storing first intensity values/cycles
  intensity_columns = [f'rIntensity {i}' for i in range(44, 50)]

  for col in intensity_columns:
      new_col_name = f'start_{col}'
      grouped_data[new_col_name] = grouped_data[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
      df_raw_int_ratio[new_col_name] = grouped_data[new_col_name]

  df_raw_int_ratio_new = df_raw_int_ratio.copy()

  # Generate new indices from 0 to 40 for all repeated indices in the DataFrame
  df_raw_int_ratio_new.index = pd.Series(df_raw_int_ratio_new.index).groupby(df_raw_int_ratio_new.index).cumcount()

  # Ensure that dataframe to be returned is a dataframe
  assert isinstance(df_raw_int_ratio_new,
                    pd.DataFrame), "df_kiel_par must be a DataFrame"

  return grouped_data, df_raw_int_ratio_new


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
  
  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

  # Get unique identifiers
  unique_identifiers = df['Identifier 1'].unique()

  # Get a list of all intensity related columns you'd use for plotting
  intensity_columns = [col for col in df.columns if col.startswith('rIntensity')]

  # Calculate the total number of subplots
  total_subplots = len(unique_identifiers) * len(intensity_columns)

  # Create an empty figure with the correct number of subplots
  fig = make_subplots(rows=total_subplots, cols=1)

  # Counter for the current subplot
  subplot_counter = 1

  # Iterate through intensities
  for intensity in intensity_columns:
        # Create a separate subplot for each unique identifier
        for identifier in unique_identifiers:
            # Filter DataFrame for the current identifier and intensity
            subset_df = df[(df['Identifier 1'] == identifier)]

            # Iterate through the rows of the subset DataFrame
            for is_ref, values, time_code in zip(subset_df['Is Ref _'], subset_df[intensity], subset_df['Time Code']):
                # Create a line plot for each row
                legend_group = f'Is Ref _ {is_ref}'
                legend_name = 'Reference' if is_ref == 1 else 'Sample'

                # Get the color and marker for the identifier
                line_color = 'black' if is_ref == 1 else '#FFAA33' # Set line color based on Is Ref
                color, marker = standard_marker_color(identifier)

                # Create a line plot for each row
                fig.add_trace(go.Scatter(x=list(range(len(values))), y=values,
                                            mode='lines+markers',
                                            name=f'{identifier} - {legend_name}',
                                            legendgroup=legend_group,
                                            text=[f'Value: {v:.2f}' for v in values],
                                            hoverinfo='text',
                                            marker=dict(color=color, symbol=marker), # Use the color and marker
                                            line=dict(color=line_color), # Set line color
                                            hovertemplate=f"<b>Identifier</b>: {identifier}<br>" +
                                                        f"<b>Measurement type</b>: {legend_name}<br>" +
                                                        "<b>Intensity [mV]</b>: %{y:.2f}<br>" +
                                                        "<b>Cycle</b>: %{x}<br>" +
                                                        f"<b>Datetime</b>: {time_code}<br>" +
                                                        "<extra></extra>",  # Hide the extra hover information
                                            ), row=subplot_counter, col=1)

            # Customize the layout of the subplot
            fig.update_xaxes(title_text='Cycle', row=subplot_counter, col=1)
            fig.update_yaxes(title_text=f'{intensity} - {identifier} [mV]', row=subplot_counter, col=1)
            fig.update_layout(showlegend=False)
        
            # Increment the subplot counter
            subplot_counter += 1

  fig.update_layout(
        # autosize=False,
        height=500 * total_subplots,
        title="Raw Intensity and Standard",
        legend_title="Standard",
        showlegend=False,
    )
    # Return the figure
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
  
  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

  # Get unique intensity values
  unique_intensities = df['Intensity'].unique()

  # Determine the number of rows required for subplots based on the number of unique intensities
  subplot_rows = len(unique_intensities)

  # Create a subplot figure with one column and a row for each unique intensity
  fig = make_subplots(rows=subplot_rows, cols=1, shared_xaxes=False, vertical_spacing=0.075)

  # Row counter for adding traces to the correct subplot
  row = 1

  # Generate a plot for each unique intensity
  for idx, intensity in enumerate(unique_intensities, start=1):
      # Filter dataframe for the current intensity
      df_filtered = df[df['Intensity'] == intensity]
      
      # Create a new figure
      # fig = go.Figure()
      # Check if the legend for this identifier has already been shown
      # Only show the legend for the central trace
      showlegend = (idx == len(unique_intensities) // 2 + 1)

      # Add a scatter trace for each unique identifier
      for identifier in df_filtered['Identifier 1'].unique():
          df_identifier = df_filtered[df_filtered['Identifier 1'] == identifier]

          # Get the color and marker for the identifier
          color, marker = standard_marker_color(identifier)

          fig.add_trace(go.Scatter(
              x=df_identifier.index,
              y=df_identifier['Ratio'],
              mode='markers',
              # mode='lines+markers',
              name=identifier,
              showlegend=showlegend,  # Only show legend for the central trace
              marker=dict(
                  color=color,# standard_marker_color(identifier)[0],  # Set color
                  symbol= marker# standard_marker_color(identifier)[1]  # Set marker style
              ),
              
              customdata=np.stack((df_identifier['Identifier 1'], 
                                  df_identifier['Is Ref _'], 
                                  df_identifier['Intensity'], 
                                  np.datetime_as_string(df_identifier['Time Code'], unit='s')), 
                                  axis=-1),
              hovertemplate="<b>Standard</b>: %{customdata[0]}<br>" +
                          "<b>Is Ref</b>: %{customdata[1]}<br>" +
                          "<b>Intensity</b>: %{customdata[2]}<br>" +
                          "<b>Ratio</b>: %{y}<br>" +
                          "<b>Datetime</b>: %{customdata[3]}<br>" +
                          "<extra></extra>"
          ),
          row=row, col=1)

      fig.update_xaxes(title_text="Cycles", row=row, col=1)
      fig.update_yaxes(title_text=f"{intensity}", row=row, col=1)
      row += 1

  fig.update_layout(
      # autosize=False,
      height=300 * subplot_rows,
      title="Intensity Ratios by Intensity and Standard",
      legend_title="Standard",
      showlegend=True,
  )

  return fig


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

  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"

  # Create a 2x1 subplot layout
  fig = make_subplots(rows=6, cols=1)

  # Placeholder to store slope and r-squared values
  identifier_stats = {}

  # Placeholder for table data
  table_data = {'Identifier': [], 'Is Ref': [], 'Intensity': [], 'Slope': [], 'Intercept': [], 'R²': []}

  for identifier in df['Identifier 1'].unique():
      identifier_stats[identifier] = []
      for is_ref in df['Is Ref _'].unique():
          for intensity in df['Intensity'].unique():
              # Filter the dataframe for each identifier, Is Ref, and Intensity
              id_ref_intensity_group = df[(df['Identifier 1'] == identifier) & 
                                                            (df['Is Ref _'] == is_ref) &
                                                            (df['Intensity'] == intensity)]

              # Extract x and y values for regression
              x = id_ref_intensity_group.index.values
              y = id_ref_intensity_group['Ratio'].astype(float)  # Ensure that ratios are floats for regression

              # Perform linear regression
              slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

              # Rˆ2 value
              r_squared = r_value ** 2

              # Store identifier, is_ref, intensity, slope, and r-squared in the dictionary
              identifier_stats[identifier].append({'is_ref': is_ref, 'intensity': intensity, 'slope': slope, 'intercept': intercept, 'R²': r_squared})

              # Add data to table and get color for each identifier
              color = standard_marker_color(identifier)[0]  # Get color
              table_data['Identifier'].append(identifier) 
              table_data['Is Ref'].append(is_ref)
              table_data['Intensity'].append(intensity)
              table_data['Slope'].append(f'{slope:.2e}')  # Format in scientific notation
              table_data['Intercept'].append(f'{intercept:.2e}')  # Format in scientific notation
              table_data['R²'].append(f'{r_squared:.2e}')  # Format in scientific notation

              # Get the color and marker for the identifier
              color, marker = standard_marker_color(identifier)

              # Custom data for fig hovertemplate
              custom_data = [[identifier, is_ref, intensity, slope, intercept, r_squared, 
                              np.datetime_as_string(id_ref_intensity_group['Time Code'].values[0], unit='s'),
                              ]]

              # Add scatterplot points for slope
              fig.add_trace(
                  go.Scatter(
                      y=[identifier], 
                      x=[slope],
                      mode='markers',
                      marker=dict(
                      color=color, # standard_marker_color(identifier)[0],  # Set color
                      symbol= marker # standard_marker_color(identifier)[1]  # Set marker style
                      ),
                      customdata=custom_data, # [[identifier, is_ref, intensity, slope, intercept, r_squared, id_ref_intensity_group['Time Code'].values[0]]],
                      hovertemplate=(
                          "<b>Identifier</b>: %{customdata[0]}<br>" +
                          "<b>Measurement type</b>: %{customdata[1]}<br>" +
                          "<b>Intensity</b>: %{customdata[2]}<br>" +
                          "<b>Slope</b>: %{customdata[3]:.2e}<br>" +  # Use scientific notation
                          "<b>Intercept</b>: %{customdata[4]:.2e}<br>" +  # Use scientific notation
                          "<b>R²</b>: %{customdata[5]:.2e}<br>" +  # Use scientific notation
                          "<b>Datetime</b>: %{customdata[6]}<br>" +  # Add Time Code here
                          "<extra></extra>"
                      )
                  ),
                  row=1, col=1
              )
              # Add scatterplot points for intercept
              fig.add_trace(
                  go.Scatter(
                      y=[identifier],    
                      x=[intercept],
                      marker=dict(
                      color=color,  # Set color
                      symbol=marker  # Set marker style
                      ),
                      customdata=custom_data,
                      hovertemplate=(
                          "<b>Identifier</b>: %{customdata[0]}<br>" +
                          "<b>Measurement type</b>: %{customdata[1]}<br>" +
                          "<b>Intensity</b>: %{customdata[2]}<br>" +
                          "<b>Slope</b>: %{customdata[3]:.2e}<br>" +  # Use scientific notation
                          "<b>Intercept</b>: %{customdata[4]:.2e}<br>" +  # Use scientific notation
                          "<b>R²</b>: %{customdata[5]:.2e}<br>" +  # Use scientific notation
                          "<b>Datetime</b>: %{customdata[6]}<br>" +  # Add Time Code here
                          "<extra></extra>"
                      )
                  ),
                  row=2, col=1
              )
              # Add scatterplot points for Rˆ2
              fig.add_trace(
                  go.Scatter(
                      y=[identifier],
                      x=[r_squared],
                      marker=dict(
                      color=color,  # Set color
                      symbol=marker  # Set marker style
                      ),
                      customdata=custom_data,
                      hovertemplate=(
                          "<b>Identifier</b>: %{customdata[0]}<br>" +
                          "<b>Measurement type</b>: %{customdata[1]}<br>" +
                          "<b>Intensity</b>: %{customdata[2]}<br>" +
                          "<b>Slope</b>: %{customdata[3]:.2e}<br>" +  # Use scientific notation
                          "<b>Intercept</b>: %{customdata[4]:.2e}<br>" +  # Use scientific notation
                          "<b>R²</b>: %{customdata[5]:.2e}<br>" +  # Use scientific notation
                          "<b>Datetime</b>: %{customdata[6]}<br>" +  # Add Time Code here
                          "<extra></extra>"
                      )
                  ),
                  row=3, col=1
              )
              fig.add_trace(
                  go.Scatter(
                      y=[slope], 
                      x=id_ref_intensity_group['start_rIntensity 44'],
                      mode='markers',
                      marker=dict(
                      color=color, # standard_marker_color(identifier)[0],  # Set color
                      symbol= marker # standard_marker_color(identifier)[1]  # Set marker style
                      ),
                      customdata=custom_data, # [[identifier, is_ref, intensity, slope, intercept, r_squared, id_ref_intensity_group['Time Code'].values[0]]],
                      hovertemplate=(
                          "<b>Identifier</b>: %{customdata[0]}<br>" +
                          "<b>Measurement type</b>: %{customdata[1]}<br>" +
                          "<b>Intensity</b>: %{customdata[2]}<br>" +
                          "<b>Slope</b>: %{customdata[3]:.2e}<br>" +  # Use scientific notation
                          "<b>Intercept</b>: %{customdata[4]:.2e}<br>" +  # Use scientific notation
                          "<b>R²</b>: %{customdata[5]:.2e}<br>" +  # Use scientific notation
                          "<b>Datetime</b>: %{customdata[6]}<br>" +  # Add Time Code here
                          "<extra></extra>"
                      )
                  ),
                  row=4, col=1
              )
              fig.add_trace(
                  go.Scatter(
                      y=[intercept], 
                      x=id_ref_intensity_group['start_rIntensity 44'],
                      mode='markers',
                      marker=dict(
                      color=color, # standard_marker_color(identifier)[0],  # Set color
                      symbol= marker # standard_marker_color(identifier)[1]  # Set marker style
                      ),
                      customdata=custom_data, # [[identifier, is_ref, intensity, slope, intercept, r_squared, id_ref_intensity_group['Time Code'].values[0]]],
                      hovertemplate=(
                          "<b>Identifier</b>: %{customdata[0]}<br>" +
                          "<b>Measurement type</b>: %{customdata[1]}<br>" +
                          "<b>Intensity</b>: %{customdata[2]}<br>" +
                          "<b>Slope</b>: %{customdata[3]:.2e}<br>" +  # Use scientific notation
                          "<b>Intercept</b>: %{customdata[4]:.2e}<br>" +  # Use scientific notation
                          "<b>R²</b>: %{customdata[5]:.2e}<br>" +  # Use scientific notation
                          "<b>Datetime</b>: %{customdata[6]}<br>" +  # Add Time Code here
                          "<extra></extra>"
                      )
                  ),
                  row=5, col=1
              )
              fig.add_trace(
                  go.Scatter(
                      y=[r_squared], 
                      x=id_ref_intensity_group['start_rIntensity 44'],
                      mode='markers',
                      marker=dict(
                      color=color, # standard_marker_color(identifier)[0],  # Set color
                      symbol= marker # standard_marker_color(identifier)[1]  # Set marker style
                      ),
                      customdata=custom_data, # [[identifier, is_ref, intensity, slope, intercept, r_squared, id_ref_intensity_group['Time Code'].values[0]]],
                      hovertemplate=(
                          "<b>Identifier</b>: %{customdata[0]}<br>" +
                          "<b>Measurement type</b>: %{customdata[1]}<br>" +
                          "<b>Intensity</b>: %{customdata[2]}<br>" +
                          "<b>Slope</b>: %{customdata[3]:.2e}<br>" +  # Use scientific notation
                          "<b>Intercept</b>: %{customdata[4]:.2e}<br>" +  # Use scientific notation
                          "<b>R²</b>: %{customdata[5]:.2e}<br>" +  # Use scientific notation
                          "<b>Datetime</b>: %{customdata[6]}<br>" +  # Add Time Code here
                          "<extra></extra>"
                      )
                  ),
                  row=6, col=1
              )


  # Update xaxis and yaxis properties if needed
  fig.update_yaxes(title_text="", type='category', row=1, col=1)
  fig.update_xaxes(title_text="Slope", row=1, col=1, tickformat='.1e')  # Use scientific notation
  fig.update_yaxes(title_text="", type='category', row=2, col=1)
  fig.update_xaxes(title_text="Intercept", row=2, col=1, tickformat='.1e')  # Use scientific notation
  fig.update_yaxes(title_text="", type='category', row=3, col=1)
  fig.update_xaxes(title_text="R²", row=3, col=1, tickformat='.1e')  # Use scientific notation
  fig.update_yaxes(title_text="Slope", row=4, col=1, tickformat='.1e')
  fig.update_xaxes(title_text="Starting intensity 44 [mV]", row=4) #, col=1, tickformat='.1e')  # Use scientific notation
  fig.update_yaxes(title_text="Intercept", row=5, col=1, tickformat='.1e')
  fig.update_xaxes(title_text="Starting intensity 44 [mV]", row=5)
  fig.update_yaxes(title_text="R²", row=6, col=1, tickformat='.1e')
  fig.update_xaxes(title_text="Starting intensity 44 [mV]", row=6)

  # Update layout if necessary, e.g., adding titles, adjusting height, etc.
  fig.update_layout(height=2500, 
                    title_text="Intensity cycles linear regression slope, intercept and R²",
                    showlegend=False)

  # Create table
  fig_table = go.Figure(data=[go.Table(
      header=dict(values=list(table_data.keys()),
                  align='left'),
      cells=dict(values=list(table_data.values()),
                # fill_color=cells_fill_color,
                align='left')
  )])

  return fig, fig_table

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
                   'ISOB_2', '_Riedel 2', 'Merck'
                   ] 

    # Print the identifier that the function is trying to find
    # print(f"Looking for identifier: {identifier}")

    # Create dictionaries that map each identifier to a color and a marker
    color_dict = {identifier: colors[i % len(colors)] for i, identifier in enumerate(identifiers)}
    marker_dict = {identifier: markers[i % len(markers)] for i, identifier in enumerate(identifiers)}

    # Return the color and marker corresponding to the given identifier
    # Return 'black' and 'circle' if the identifier is not found
    return color_dict.get(identifier, 'black'), marker_dict.get(identifier, 'circle')