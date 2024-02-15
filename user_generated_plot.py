# user_generated_plot.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def user_generated_plots(df: pd.DataFrame):
  """
  Generates plot of columns, which are in standard isotope dataframe, that are 
  selected by user.

  Parameters
  ----------
  df : pd.DataFrame
    Dataframe containing the isotope data.
  Return
  ------
  None
  """

  # Ensure we have a pd.Dataframe object
  assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
  
  # st.write("User generated plots")
  # Create Streamlit widgets for the dropdowns
  x_dropdown = st.selectbox("X-axis:", df.columns, index=3)
  y_dropdown = st.selectbox("Y-axis:", df.columns, index=7)
  z_columns = ['Row', 'Sample', 'Line']
  z_dropdown = st.selectbox("Z-axis:", z_columns, index=2)

  # Function to update the plot
  def update_plot(x, y, z):
    # Define a dictionary mapping z values to colors
    z_colors = {
        1: 'blue',  # Set color for z value 0
        2: 'red',  # Set color for z value 1
    }
    fig = go.Figure(data=go.Scatter(
        x=df[x],
        y=df[y],
        mode='markers',
        marker=dict(color=df[z].apply(
            lambda z_value: z_colors.get(z_value, "black")
        )  # Map z values to colors using z_colors dictionary
                    ),
        text=df[z],  # Add z values to text
        hovertemplate=f"<b>{x}</b>: %{{x:.1f}}<br>" +
        f"<b>{y}</b>: %{{y:.1f}}<br>" + f"<b>{z}</b>: %{{text}}<br>" +
        "<extra></extra>",  # Hide the extra hover information               
        # hoverinfo='x+y+text'  # Specify the hoverinfo to show only 'x', 'y', and 'text'
    ))

    fig.update_layout(title=f"{y} versus {x} colored by {z}",
                      xaxis_title=x,
                      yaxis_title=y)

    st.plotly_chart(fig, use_container_width=True, theme=None)

  # Create a Streamlit button to update the plot
  button = st.button("Update Plot")

  # Check if the button is clicked
  if button:
    update_plot(x_dropdown, y_dropdown, z_dropdown)
