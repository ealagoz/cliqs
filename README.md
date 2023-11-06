# farlab_clumb_data_quality
# Data Visualization and Analysis

This section of the code focuses on data visualization and analysis. It includes code snippets that utilize various Python libraries to create insightful visualizations and perform data analysis. The code examples are grouped into different sections, each serving a specific purpose. 

## Table of Contents

- [Python libraries](#Libraries)
- [Multiple Scatter Plots](#multiple-scatter-plots)
- [Interactive Scatter Plot with Dropdown Menus](#interactive-scatter-plot-with-dropdown-menus)
- [Delta Values Scatter Plots with Error Bars](#delta-values-scatter-plots-with-error-bars)
- [Delta Values Scatter Plots per Standard vs Time](#delta-values-scatter-plots-per-standard-vs-time)

## Libraries

These import statements include the necessary libraries and modules required for the code.

- `pandas`: A data manipulation library for Python, providing data structures and operations for working with numerical, textual, and categorical data.
- `plotly.express`: A high-level API for creating interactive visualizations with Plotly, simplifying the creation of common plot types like scatter plots, line plots, and bar charts.
- `plotly.graph_objects`: A lower-level API for creating interactive visualizations with Plotly, offering more fine-grained control over plot appearance and behavior.
- `plotly.subplots make_subplots`: A module for creating subplots with Plotly, providing a function for generating figures with multiple plots.
- `flask`: A lightweight Python web framework for building web applications, providing tools for routing requests, handling data, and rendering templates.
    - `flask send_file`: A Flask function for sending files as responses.
    - `flask url_for`: A Flask function for generating URLs for routes and static files.
    - `flask render_template`: A Flask function for rendering HTML templates.
- `re`: The Python regular expressions library, used for pattern matching and searching.
- `plotly.io`: A module for reading and writing Plotly figures, including functions for saving and loading figures in various formats.
- `re`: A module for working with regular expressions, providing functions for matching patterns in text and extracting data from text.
- `ipywidgets`: A library for creating interactive widgets in Jupyter notebooks, offering various widget types like sliders, dropdown menus, and text boxes.
- `IPython.display`: A module for displaying objects in Jupyter notebooks, providing functions for displaying text, images, and visualizations.
    - `display`: A function for displaying objects in Jupyter notebooks, including text, images, and visualizations.
    - `clear_out`: A function for clearing the output area of a Jupyter notebook cell.

**Installation**
Python package installer "pip" (https://pypi.org/project/pip/) is used to install libraries:
- Python 3.11.0 version is used for the analysis
- pip install pandas plotly flask IPython ipywidgets
- re is already included in Python built-in libraries

## Multiple Scatter Plots

The code in this section demonstrates how to create multiple scatter plots to visualize the relationship between `Weight (mg)` and various numeric columns, categorized by `Kiel acid line number`. It follows a step-by-step explanation of the code, including defining subplot height, calculating subplot layout, creating subplot grids, and more.

## Interactive Scatter Plot with Dropdown Menus

In this part, an interactive scatter plot with dropdown menus is created. Users can dynamically select the x-axis, y-axis, and z-axis variables using the dropdowns. The README provides a detailed explanation of how the dropdowns are created, how to set their options, and how the update plot function works to create interactive visualizations.

## Delta Values Scatter Plots with Error Bars

This section focuses on generating scatter plots with error bars to visualize the relationship between weight (mg) and various numeric columns, categorized by line type. It also displays the mean and standard deviation for each numeric column. The README explains how error columns and mean columns are identified, and how the scatter plots are created.

## Delta Values Scatter Plots per Standard vs Time

The code in this section generates scatter plots with markers for each group_column and displays custom information on hover. It utilizes the plotly.graph_objects library to create and manipulate the figures. The README provides an overview of how the code creates scatter plots for different group columns and displays additional information on hover.

## Usage

You can use the provided code snippets as a reference for data visualization and analysis in your projects. The README sections are organized to help you understand each part of the code and its purpose.

# File Operations and Flask Web Application

This section of the code focuses on file operations and the creation of a basic Flask web application. It includes code snippets that connect to a remote file server and copy files, as well as the setup of a Flask web application with defined routes.

## Table of Contents

- [Connect to the Lab-IT klient file server and copy a file](#connect-to-the-lab-it-klient-file-server-and-copy-a-file)
- [Flask Web Application](#flask-web-application)

## Connect to the Lab-IT klient file server and copy a file
--does not work

The code in this section attempts to connect to a remote file server (Lab-IT klient) and copy a file from it. The README provides an overview of the code, which includes using environment variables for authentication and file copying. Note that the code may require adjustments for specific server configurations and permissions.

## Flask Web Application
-- not yet active and one could use another simple web app library such as "Streamlit"

This part of the code defines a basic Flask web application with two routes to handle web requests. The README explains the purpose of the Flask application and provides details on the defined routes and their functionalities. To run the Flask web application, uncomment the necessary lines as indicated in the code.

## Usage

You can use the provided code snippets as a reference for file operations and the creation of a Flask web application. The README sections are organized to help you understand each part of the code and its purpose.

