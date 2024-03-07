# file_handler.py
import streamlit as st
import tempfile
import pandas as pd
import xlrd
import os


def upload_file(should_display: bool = False):
  """
      Upload a file using Streamlit File Uploader widget.

      Returns:
          str or None: The path to the uploaded file if available, otherwise None.
      """
  if should_display:
    with st.expander(":file_folder: UPLOAD FILE", expanded=False):
      st.write("""
    ...         Click **Browse files** or **Drag and drop** to upload clump instrument run export file in the format:
                :blue[RunXXXX.xls] 
    ...     """)
      # file_uploader label should be define with " " to hide the label
      file = st.file_uploader(" ",
                              type=["xls", "xlsx"],
                              label_visibility="collapsed")
      if file is not None:
        # Save the file name in session state
        st.session_state.uploaded_filename = file.name  # Include this line to save the uploaded file name
        # Create a temporary file to store the uploaded file contents
        with tempfile.NamedTemporaryFile(suffix="xls", delete=False) as tmp:
          tmp.write(file.read())

        return tmp.name
  else:
    return None


def open_excel_file(file):
  """
    Read and filter data from two Excel shhets into Pandas DataFrames.

    Args:
    file (str): The path to the Excel file containing data.

    Returns:
        tuple: A tuple containing two Pandas DataFrames.
            - df_std_cp: A filtered DataFrame from 'clumped_export.wke' sheet.
            - df_intensity_cp: A filtered DataFrame from 'clumped_all_cycles_extra_workin' sheet.

    Raises:
        FileNotFoundError: If the file is not found.
    """
  # Check if the file name has changed or is new before printing
  if 'uploaded_filename' in st.session_state and st.session_state.uploaded_filename not in st.session_state.get('printed_filenames', []):
      print("Uploaded file name:", st.session_state.uploaded_filename)
  try:
    # Read the 'clumped_export.wke' and 'clumped_all_cycles_extra_workin' sheets
    workbook = xlrd.open_workbook(file, logfile=open(os.devnull, "w"))
    df_std = pd.read_excel(workbook, sheet_name=workbook.sheet_names()[0])
    df_intensity = pd.read_excel(workbook,
                                 sheet_name=workbook.sheet_names()[1])
  except FileNotFoundError:
    raise FileNotFoundError("File not found!")

  # # Filter both DataFrames to include only 'standard' and 'standard_refill' identifiers
  df_std_cp = df_std[df_std['Identifier 2'].isin(
      ['standard', 'standard_refill'])]
  df_intensity_cp = df_intensity[df_intensity['Identifier 2'].isin(
      ['standard', 'standard_refill'])]

  # Drop unnecessary columns
  df_std_cp = df_std_cp.drop(columns=["Time", "Date"])

  # Rename column Weight (mg) (mg) to Weight
  df_std_cp = df_std_cp.rename(columns={"Weight (mg)": "Weight"})

  # Convert 'Time Code' to datetime format
  df_std_cp["Time Code"] = pd.to_datetime(df_std_cp["Time Code"])
  df_intensity_cp["Time Code"] = pd.to_datetime(df_intensity_cp["Time Code"])

  # Identify and convert columns containing text values to strings
  text_columns_std = df_std_cp.select_dtypes(include=["object"]).columns
  text_columns_intensity = df_intensity_cp.select_dtypes(
      include=["object"]).columns

  df_std_cp[text_columns_std] = df_std_cp[text_columns_std].astype("string")
  df_intensity_cp[text_columns_intensity] = df_intensity_cp[
      text_columns_intensity].astype("string")
  # numeric columns
  # numeric_columns = df_std_cp.select_dtypes(include=["float64", "Int64"]).columns

  return df_std_cp, df_intensity_cp
