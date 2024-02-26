# Import required libraries
import plotly.graph_objects as go
from database import fetch_kiel_data_as_dataframe
import pandas as pd
import numpy as np
# Import the standard_marker_color() function from standards.py
from standards import standard_marker_color

def plot_kiel_parameters_ts():
    # Fetch the DataFrame containing Kiel parameters
    df = fetch_kiel_data_as_dataframe()
    if df is None or df.empty:
        print("DataFrame is empty. No data to plot.")
        return []
    
    # Ensure 'time' is a datetime column and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

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

    # Create a list to hold all the figures
    figs = []

    # Iterate through each parameter in the DataFrame except 'id'
    for parameter in df.columns.difference(['id']):
        if parameter not in kiel_par_dict:
            continue  # Skip parameters not listed in the custom dictionary
        # Create a time series plot
        fig = go.Figure(layout={'template': "ggplot2+presentation"})

        # Fetch unique identifiers (standards) to plot each with its color and marker
        identifiers = df['standard'].unique()

        for identifier in identifiers:
            # Filter DataFrame per standard
            filtered_df = df[df['standard'] == identifier]

            # Get color and marker for the identifier
            color, marker = standard_marker_color(identifier)

            # Define the custom data line from raw_intensity.py to use in hovertemplate
            custom_data = np.stack((filtered_df["standard"].values, 
                                    filtered_df["line"].values, 
                                    filtered_df.index.strftime("%Y-%m-%d %H:%M:%S"), 
                                    [parameter] * len(filtered_df),    # Added parameter value to custom data
                                    filtered_df[parameter].values     # Added corresponding parameter values
                                   ), axis=-1)

            fig.add_trace(go.Scatter(x=filtered_df.index, 
                                     y=filtered_df[parameter], 
                                     mode='markers', 
                                     name=f"{identifier}",
                                     marker=dict(color=color, symbol=marker),  # Use identified color and marker
                                     customdata=custom_data,
                                     hovertemplate=
                                     "<b>Standard</b>: %{customdata[0]}<br>" + 
                                     "<b>Line</b>: %{customdata[1]}<br>" +
                                     "<b>Datetime</b>: %{customdata[2]}<br>" + 
                                     f"<b>{parameter}</b>: %{{y}}" + 
                                     "<extra></extra>"
                                    )
                                )

        # Setting plot title and axis titles
        fig.update_layout(title=f'{kiel_par_dict.get(parameter, parameter)}',
                          # xaxis_title='Time',
                          yaxis_title=parameter,
                          autosize=True,
                          )

        # Append the figure to the list of figures
        figs.append(fig)
    
    # Return the list of figures
    return figs
