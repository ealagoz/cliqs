# main.py

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.io as pio

# import functions from file_handler.py
from file_handler import upload_file, open_excel_file
# import functions from kiel.py
from kiel import get_kiel_data, generate_kiel_plots
# import functions from standards.py
from standards import numeric_columns_plots, isotope_std_plots
from user_generated_plot import user_generated_plots
# import functions from raw_intenisty.py
from raw_intensity import group_standard_intensities, raw_standard_intensity_plots, raw_standard_intensity_stats_plots, raw_standard_intensity_ratio_plots, raw_standard_intensity_ratio_fit_plots

# Set the default tamplate globally
pio.templates.default = "ggplot2+presentation"

# Disable the warning for chained assignment
pd.options.mode.chained_assignment = None  # "warn" or "raise"


# Streamlit app specs
def app_specs():
  """
    Set the Streamlit web app page specifications.
    """
  # Set the page title and icon
  st.set_page_config(page_title="CLIQS",
                     page_icon="./farlab_logo.png",
                     layout="wide")

  # Create a custom header with a gradient background
  st.header("CLIQS (Clumped Isotope Data Quality Monitoring Control System)")
  st.markdown("""<style>.block-container{padding-top:2rem;}</style>""",
              unsafe_allow_html=True)

  # Add a logo to the header
  # st.sidebar.image("./docs/farlab_logo.png", use_column_width=True)
  image_url = "https://www.uib.no/sites/w3.uib.no/files/styles/area_bottom/public/media/logo2_1.png?itok=5njSro1G"
  # Add a logo with a hyperlink in the sidebar
  st.sidebar.markdown(
      f'<a href="https://uib.no/en/FARLAB"><img src="{image_url}" width="200"></a>',
      unsafe_allow_html=True)


def main():
  """
    The main function of the Streamlit web app.

    Parameters:
    None

    Returns:
    None 
  """
  # Set the page title and icon
  app_specs()

  # Load your data here
  uploaded_file = upload_file()

  if uploaded_file is not None and uploaded_file != 0:
    # Load the data
    df_std, df_intensity = open_excel_file(uploaded_file)
    # Get kiel data
    df_kiel_par = get_kiel_data(df_std)
    # generate kiel plots
    kiel_figs = generate_kiel_plots(df_kiel_par)
    # Create numeric columns plots
    numeric_figs = numeric_columns_plots(df_std)
    standard_figs = isotope_std_plots(df_std)
    # Generate intensity grouped per std dataframe
    df_intensity_group = group_standard_intensities(df_intensity)
    # Generate raw intensity plots
    raw_intensity_figs = raw_standard_intensity_plots(df_intensity_group)
    # Generate intensity stats plots
    df_intensity_group_stat, intensity_stats_figs = raw_standard_intensity_stats_plots(
        df_intensity_group)
    # Generate intensity ratio plots and dataframe
    df_intensity_ratio, intensity_ratio_figs = raw_standard_intensity_ratio_plots(
        df_intensity_group)
    # Generate intensity ratio linear regression plots
    intensity_ratio_linreg_figs = raw_standard_intensity_ratio_fit_plots(
        df_intensity_ratio)

    # Create a session_state variable to store the processed data
    st.session_state['df_std'] = df_std
    st.session_state['df_intensity'] = df_intensity
    st.session_state['df_kiel_par'] = df_kiel_par
    st.session_state['kiel_figs'] = kiel_figs
    st.session_state['numeric_figs'] = numeric_figs
    st.session_state['standard_figs'] = standard_figs
    st.session_state['raw_intensity_figs'] = raw_intensity_figs
    st.session_state['intensity_stats_figs'] = intensity_stats_figs
    st.session_state['intensity_ratio_figs'] = intensity_ratio_figs
    st.session_state[
        'intensity_ratio_linreg_figs'] = intensity_ratio_linreg_figs

    # Create a sidebar with tabs
    with st.sidebar:
      selected_tab = option_menu(
          menu_title=None,  #"Main", # Required
          options=[
              "Kiel Parameters", "Isotope Data", "Generate Plot", "Standards",
              "Intensities"
          ],  # Required
          default_index=0,  # Required
          menu_icon="📊",  # Optional
      )

    # Display the selected page
    if selected_tab == "Kiel Parameters":
      st.header("Kiel Parameters")
      st.plotly_chart(kiel_figs, use_container_width=True, theme=None)
    elif selected_tab == "Isotope Data":
      st.header("Isotope Data")
      st.plotly_chart(numeric_figs, use_container_width=True, theme=None)
    elif selected_tab == "Generate Plot":
      st.header("Generate Plot")
      user_generated_plots(df_std)
    elif selected_tab == "Standards":
      st.header("Standards")
      st.plotly_chart(standard_figs, use_container_width=True, theme=None)
    elif selected_tab == "Intensities":
      st.header("Intensities")
      raw_int, raw_int_stat, raw_int_ratio, raw_int_linreg = st.tabs([
          "Raw Intensities", "Intensity Stats", "Intensity Ratio",
          "Intensity Ratio Linear Regression"
      ])
      with raw_int:
        st.write("Raw intensities per standard")
        st.plotly_chart(raw_intensity_figs,
                        use_container_width=True,
                        theme=None)
      with raw_int_stat:
        st.write("Raw intensities stats per standard")
        st.plotly_chart(intensity_stats_figs,
                        use_container_width=True,
                        theme=None)
      with raw_int_ratio:
        st.write("Raw intensities ratio per standard")
        st.plotly_chart(intensity_ratio_figs,
                        use_container_width=True,
                        theme=None)
      with raw_int_linreg:
        st.write("Raw intensities ratio linear regression per standard")
        st.plotly_chart(intensity_ratio_linreg_figs,
                        use_container_width=True,
                        theme=None)
    else:
      st.error("Something has gone terribly wrong.")


if __name__ == "__main__":
  main()
