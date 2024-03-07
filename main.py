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
from raw_intensity import (group_standard_intensities, 
                           raw_standard_intensity_plots, 
                           raw_standard_intensity_ratio_plots, 
                           raw_standard_intensity_ratio_fit_plots,
                           raw_standard_intensity_ratio_fit_par_to_database)
from database import (create_standard_table, 
                      insert_kiel_data, 
                      create_kiel_table, 
                      insert_standard_parameters_data,
                      create_intensity_ratio_fit_pars_table,
                      insert_intensity_ratio_fit_pars)
from time_series import (plot_kiel_parameters_ts, 
                         plot_standard_parameters_ts,
                         plot_intensity_ratio_fit_par_ts)

# Set the default tamplate globally
pio.templates.default = "ggplot2+presentation"

# Disable the warning for chained assignment
pd.options.mode.chained_assignment = None  # "warn" or "raise"

# Initialize 'previous_selection' if it doesn't exist in st.session_state
# if 'previous_selection' not in st.session_state:
#     st.session_state.previous_selection = None

# Initialize 'data_processed' if it doesn't exist in st.session_state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False

# Initialize 'selected_tab' in st.session_state at the beginning of main()
if 'selected_tab' not in st.session_state:
    st.session_state['selected_tab'] = "Kiel Parameters"  # Default tab

# Streamlit app specs
def app_specs():
  """
    Set the Streamlit web app page specifications.
    """
  # Set the page title and icon
  st.set_page_config(page_title="CLIQS",
                     page_icon="./docs/farlab_logo.png",
                     layout="wide")

  # Create a custom header with a gradient background
  st.header("CLIQS (Clumped Isotope Data Quality Monitoring Control System)")
  st.markdown("""<style>.block-container{padding-top:2rem;}</style>""",
              unsafe_allow_html=True)

  # Add a logo to the header
  # st.sidebar.image("../docs/farlab_logo.png", use_column_width=True)
  image_url = "https://www.uib.no/sites/w3.uib.no/files/styles/area_bottom/public/media/logo2_1.png?itok=5njSro1G"
  # Add a logo with a hyperlink in the sidebar
  st.sidebar.markdown(
      f'<a href="https://uib.no/en/FARLAB"><img src="{image_url}" width="200"></a>',
      unsafe_allow_html=True)
  # Add a logo to the header
  # st.sidebar.image("../docs/CLIMB_no_txt.pdf", use_column_width=True)

# Generate dataframes from the uploaded file
def generate_dataframes():
  """
    Generate dataframes from the uploaded file.
    """
  # Load your data here
  uploaded_file = upload_file(should_display=True)

  # Read the uploaded file into a pandas DataFrame
  if uploaded_file is not None and uploaded_file != 0:
    # Load the data
    df_std, df_intensity = open_excel_file(uploaded_file)
    # Get kiel data
    df_kiel_par = get_kiel_data(df_std)

    # print(df_kiel_par.head())

    return df_std, df_kiel_par, df_intensity

  else:
    return None, None, None

# Insert data into the database
def database_data_insert(df_kiel_par: pd.DataFrame, df_intensity: pd.DataFrame, df_intensity_ratio_fit: pd.DataFrame, instrument: str):
  """
    Insert data into the database.
    """
  # Add instrument column to df_kiel_par
  df_kiel_par['instrument'] = instrument
  df_intensity_ratio_fit['instrument'] = instrument
  # Change dtype of instrument column to category
  df_intensity_ratio_fit['instrument'] = df_intensity_ratio_fit['instrument'].astype('string')
  df_kiel_par['instrument'] = df_kiel_par['instrument'].astype('string')

  # print(df_intensity_ratio_fit.groupby('standard')['time'].first())

  # Create kiel table in the database
  create_kiel_table()
  # Insert kiel data into the database
  insert_kiel_data(df_kiel_par)
  # Create standard table in the database
  create_standard_table()
  # Insert standard data into the database
  insert_standard_parameters_data(df_kiel_par)
  # Create intensity ratio fit table in the database
  create_intensity_ratio_fit_pars_table()
  # Insert intensity ratio fit data into the database
  insert_intensity_ratio_fit_pars(df_intensity_ratio_fit)

# Insert intensity data into the database
def process_uploaded_file(df_std: pd.DataFrame, df_kiel_par: pd.DataFrame, df_intensity: pd.DataFrame):
  """
    Process the uploaded file and generate plots.
    """
  # generate kiel plots
  kiel_figs = generate_kiel_plots(df_kiel_par)
  # Create numeric columns plots
  numeric_figs = numeric_columns_plots(df_kiel_par)
  standard_figs = isotope_std_plots(df_kiel_par)
  # Generate intensity grouped per std dataframe
  df_intensity_group, df_intensity_group_ratio = group_standard_intensities(df_intensity)
  # Generate raw intensity plots
  raw_intensity_figs = raw_standard_intensity_plots(df_intensity_group)
  # Generate intensity ratio plots and dataframe
  intensity_ratio_figs = raw_standard_intensity_ratio_plots(
      df_intensity_group_ratio)
  # Generate intensity ratio linear regression plots
  intensity_ratio_linreg_figs, intensity_ratio_linreg_table = raw_standard_intensity_ratio_fit_plots(
      df_intensity_group_ratio)

  # Create a session_state variable to store the processed data
  st.session_state['df_std'] = df_std
  st.session_state['df_intensity'] = df_intensity
  st.session_state['df_kiel_par'] = df_kiel_par
  st.session_state['kiel_figs'] = kiel_figs
  st.session_state['numeric_figs'] = numeric_figs
  st.session_state['standard_figs'] = standard_figs
  st.session_state['raw_intensity_figs'] = raw_intensity_figs
  st.session_state['intensity_ratio_linreg_figs'] = intensity_ratio_linreg_figs
  st.session_state['intensity_ratio_figs'] = intensity_ratio_figs
  st.session_state[
      'intensity_ratio_linreg_table'] = intensity_ratio_linreg_table

  # Create a sidebar with tabs
  with st.sidebar:
      selected_tab = option_menu(
          menu_title=None,  # "Main", # Required
          options=[
              "Kiel Parameters", "Isotope Data", "Generate Plot",
              "Intensities"
          ],  # Required
          default_index=0,  # Required
          # menu_icon="📊",  # Optional
      )

  # Display the selected page
  if selected_tab == "Kiel Parameters":
      st.header("Kiel Parameters")
      st.plotly_chart(kiel_figs, use_container_width=True, theme=None)
  elif selected_tab == "Isotope Data":
      st.header("Isotope Data")
      numeric_plots, standard_plots = st.tabs(["Numeric Columns", "Standards"])
      with numeric_plots:
          st.header("Numeric Columns")
          st.plotly_chart(numeric_figs, use_container_width=True, theme=None)
      with standard_plots:
          st.header("Standards")
          st.plotly_chart(standard_figs, use_container_width=True, theme=None)
  elif selected_tab == "Generate Plot":
      st.header("Generate Plot")
      user_generated_plots(df_std)
  elif selected_tab == "Intensities":
      st.header("Intensities")
      raw_int, raw_int_ratio, raw_int_linreg, raw_int_linreg_table = st.tabs([
          "Raw Intensities", "Intensity Ratio",
          "Intensity Ratio Linear Regression",
          "Intensity Ratio Linear Regression Table"
      ])
      with raw_int:
          st.write("Raw intensities per standard")
          st.plotly_chart(raw_intensity_figs,
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
      with raw_int_linreg_table:
          st.write("Raw intensities ratio linear regression table")
          st.plotly_chart(intensity_ratio_linreg_table, use_container_width=True)
  else:
      st.error("Something has gone terribly wrong.")

# Refactored process_uploaded_file function
# Refactor existing process_uploaded_file function to preprocess and not directly visualize
def preprocess_uploaded_file(df_std: pd.DataFrame, df_kiel_par: pd.DataFrame, df_intensity: pd.DataFrame):
    """
    Preprocess the uploaded file for further analysis and visualization.
    """
    # Generate kiel plots data
    df_kiel_plots_data = generate_kiel_plots(df_kiel_par)
    # Create numeric columns plots data
    df_numeric_plots_data = numeric_columns_plots(df_kiel_par)
    df_standard_plots_data = isotope_std_plots(df_kiel_par)
    # Generate grouped by standard intensity DataFrame
    df_intensity_group, df_intensity_group_ratio = group_standard_intensities(df_intensity)
    # Generate intensity ratio DataFrame for linear regression analysis
    df_intensity_ratio_fit = raw_standard_intensity_ratio_fit_par_to_database(df_intensity_group_ratio)
    # Generate intensity ratio linear regression plots
    intensity_ratio_linreg_figs, intensity_ratio_linreg_table = raw_standard_intensity_ratio_fit_plots(
      df_intensity_group_ratio)

    # Storing processed data in session_state for further use in visualizations
    st.session_state['df_intensity_group'] = df_intensity_group
    st.session_state['df_intensity_group_ratio'] = df_intensity_group_ratio
    st.session_state['kiel_figs'] = df_kiel_plots_data
    st.session_state['numeric_figs'] = df_numeric_plots_data
    st.session_state['standard_figs'] = df_standard_plots_data
    st.session_state['df_std'] = df_std
    st.session_state['intensity_ratio_linreg_figs'] = intensity_ratio_linreg_figs
    st.session_state['intensity_ratio_linreg_table'] = intensity_ratio_linreg_table

    
    # Insert data into the database
    database_data_insert(df_kiel_par, df_intensity, df_intensity_ratio_fit, st.session_state.get('selected_instrument', 'Nessie'))

# Visualization Functions
def kiel_parameters_visualization():
    st.header("Kiel Parameters")
    if 'kiel_figs' in st.session_state:
      st.plotly_chart(st.session_state['kiel_figs'], use_container_width=True, theme=None)

def isotope_data_visualization():
    st.header("Isotope Data")
    numeric_plots, standard_plots = st.tabs(["Numeric Columns", "Standards"])
    with numeric_plots:
        st.header("Numeric Columns")
        if 'numeric_figs' in st.session_state:
            st.plotly_chart(st.session_state['numeric_figs'], use_container_width=True, theme=None)
    with standard_plots:
        st.header("Standards")
        if 'standard_figs' in st.session_state:
            st.plotly_chart(st.session_state['standard_figs'], use_container_width=True, theme=None)

def generate_plot_visualization():
    st.header("Generate Plot")
    if 'df_std' in st.session_state:
        user_generated_plots(st.session_state.get('df_std'))

def intensities_visualization():
    st.header("Intensities")
    raw_int, raw_int_ratio, raw_int_linreg, raw_int_linreg_table = st.tabs([
        "Raw Intensities", "Intensity Ratio",
        "Intensity Ratio Linear Regression",
        "Intensity Ratio Linear Regression Table"
    ])
    with raw_int:
        st.write("Raw intensities per standard")
        raw_intensity_figs = raw_standard_intensity_plots(st.session_state.get('df_intensity_group', pd.DataFrame()))
        st.plotly_chart(raw_intensity_figs, use_container_width=True, theme=None)
    with raw_int_ratio:
        st.write("Raw intensities ratio per standard")
        intensity_ratio_figs = raw_standard_intensity_ratio_plots(st.session_state.get('df_intensity_group_ratio', pd.DataFrame()))
        st.plotly_chart(intensity_ratio_figs, use_container_width=True, theme=None)
    with raw_int_linreg:
        st.write("Raw intensities ratio linear regression per standard")
        intensity_ratio_linreg_figs, _ = raw_standard_intensity_ratio_fit_plots(st.session_state.get('df_intensity_group_ratio', pd.DataFrame()))
        st.plotly_chart(intensity_ratio_linreg_figs, use_container_width=True, theme=None)
    with raw_int_linreg_table:
        st.write("Raw intensities ratio linear regression table")
        st.plotly_chart(st.session_state['intensity_ratio_linreg_table'], use_container_width=True, theme=None)

# Display the time series plots of Kiel parameters
def kiel_par_time_series(instrument: str):
    """
    Display the time series plots of Kiel parameters.
    """
    figs = plot_kiel_parameters_ts(instrument)  # Call the function from time_series.py
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True, theme=None)

# Display the time series plots of Standard parameters
def standard_par_time_series(instrument: str):
    """
    Display the time series plots of Standard parameters.
    """
    figs = plot_standard_parameters_ts(instrument)  # Call the function from time_series.py
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True, theme=None)

def intensity_ratio_fit_par_time_series(instrument: str):
    """
    Display the time series plots of Intensity Ratio Fit parameters.
    """
    figs = plot_intensity_ratio_fit_par_ts(instrument)  # Call the function from time_series.py
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True, theme=None)

def select_instrument():
    """
    Select the instrument to visualize.
    """
    st.sidebar.subheader("Instrument Selection")
    visualize_option = ["Nessie", "Yeti"]

    # Ensure there is always a default selection when options are present.
    # If 'visualize_option' list could ever be empty, set a sensible default instrument.
    default_instrument = "Nessie" if "Nessie" in visualize_option else None
    
    if visualize_option: # Check if the list is not empty
      instrument = st.sidebar.selectbox("Select Instrument:", visualize_option, index=0, key="instrument_selection")
    else:
      instrument = default_instrument

    # Safety check: Ensure instrument is not None
    instrument = instrument or "Nessie"
  
    # Write selected instrument to the main page with bold and red color
    st.markdown(f"""**Selected instrument**: <span style="font-size: 20px; color:red; font-weight: bold;">{instrument}
                </span>""", unsafe_allow_html=True)
    return instrument

def main():
  """
    The main function of the Streamlit web app.
  """

  app_specs()

  # Initialize a flag in st.session_state to track database operations
  if 'data_inserted' not in st.session_state:
      st.session_state['data_inserted'] = False

  st.sidebar.title("Visualize")
  option = st.sidebar.selectbox("Select an option:", ["Time Series", "Run File"], index=0)

  instrument = select_instrument()

  if option == "Run File":
    if not st.session_state.get('data_processed', False):
      df_std, df_kiel_par, df_intensity = generate_dataframes()
      if df_std is not None and df_kiel_par is not None and df_intensity is not None:
            # Initialize database and data insertion only once
            if not st.session_state['data_inserted']:
                # Your database operations go here
                preprocess_uploaded_file(df_std, df_kiel_par, df_intensity)
                st.session_state['data_inserted'] = True
            # st.session_state['data_processed'] = True
            display_visualizations_based_on_tab()
    else:
        st.warning("Please upload a file.")
        st.warning("And select instrument name to proceed.")

  elif option == "Time Series":
    display_time_series_visualizations(instrument)
  else:
    st.error("Something has gone terribly wrong.")

  st.sidebar.image("./docs/climb_logo.jpeg", width=200)

def display_visualizations_based_on_tab():
  """
  Display visualizations based on the selected tab.
  """
  # Debug print statement
  # st.write(f"Currently Selected: {st.session_state.selected_tab}")

  with st.sidebar:
      selected_option = option_menu(
          menu_title=None,
          options=["Kiel Parameters", "Isotope Data", "Generate Plot", "Intensities"],
          default_index=0,
      )
      # Update the session state after option selection
      st.session_state.selected_tab = selected_option  # Ensure this happens on every rerun

  if st.session_state.selected_tab == "Kiel Parameters":
      kiel_parameters_visualization()
  elif st.session_state.selected_tab == "Isotope Data":
      isotope_data_visualization()
  elif st.session_state.selected_tab == "Generate Plot":
      generate_plot_visualization()
  elif st.session_state.selected_tab == "Intensities":
      intensities_visualization()
  else:
      st.error("Something has gone terribly wrong.")

def display_time_series_visualizations(instrument):
  """
  Create tabs for time series plots.
  """
  kiel_par, std_par, init_par = st.tabs(["Kiel Parameters Time Series",
                                        "Standard Parameters Time Series",
                                        "Intensity Ratio Fit Parameters"])

  with kiel_par:
    st.header("Kiel Parameters Time Series")
    kiel_par_time_series(instrument)

  with std_par:
    st.header("Standard Parameters Time Series")
    standard_par_time_series(instrument)

  with init_par:
    st.header("Intensity Ratio Fit Parameters")
    intensity_ratio_fit_par_time_series(instrument)

if __name__ == "__main__":
  main()
