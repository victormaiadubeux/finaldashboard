import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import textwrap
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go




# Seed for reproducibility
np.random.seed(0)




# Example DataFrame creation
dates = pd.date_range('2021-01-01', periods=100)
data = {
  'Date': np.random.choice(dates, 1000),  # Adjusted for simplified example
  'Status': np.random.choice(['Patient', 'Provider'], 1000, p=[0.8, 0.2]),  # Assuming 20% providers
  'Gender': np.random.choice(['Male', 'Female', 'Unknown or Not Reported'], 1000),
  'Race': np.random.choice(['American Indian or Alaska Native', 'Asian', 'Black or African American','Native Hawaiian or Other Pacific Islander', 'White', 'More than One Race', 'Unknown or Not Reported'], 1000),
  'Ethnicity': np.random.choice(['Hispanic or Latino', 'Not Hispanic or Latino', 'Unknown or Not Reported'], 1000)
}
df = pd.DataFrame(data)




# Assign patients to providers and generate random scores
# Identifying provider IDs first
provider_ids = df[df['Status'] == 'Provider'].index.to_series()
# Assigning each patient a provider
df['Assigned Provider'] = np.nan
df.loc[df['Status'] == 'Patient', 'Assigned Provider'] = np.random.choice(provider_ids, size=len(df[df['Status'] == 'Patient']))
# Generating random scores for patients on various metrics
scores_columns = ['Health', 'Satisfaction', 'Experience with Digital Health', 'Comfort with Recording Devices', 'Communication Quality']
for score in scores_columns:
  df[score] = np.nan  # Initialize columns with NaNs for providers
  df.loc[df['Status'] == 'Patient', score] = np.random.uniform(0, 1, df[df['Status'] == 'Patient'].shape[0])




# Streamlit app UI starts here
st.title('Patient Demographics Dashboard')




# Filters
st.sidebar.header("Demographic Filters")
status_filter = st.sidebar.selectbox('Select Status', ['All'] + sorted(df['Status'].unique()))
gender_filter = st.sidebar.selectbox('Select Gender', ['All', 'Male', 'Female', 'Unknown or Not Reported'])
race_filter = st.sidebar.selectbox('Select Race', ['All', 'American Indian or Alaska Native', 'Asian', 'Black or African American','Native Hawaiian or Other Pacific Islander', 'White', 'More than One Race', 'Unknown or Not Reported'])
ethnicity_filter = st.sidebar.selectbox('Select Ethnicity', ['All', 'Hispanic or Latino', 'Not Hispanic or Latino', 'Unknown or Not Reported'])




# Filtering the dataframe based on selections
filtered_df = df
if status_filter != 'All':
  filtered_df = filtered_df[filtered_df['Status'] == status_filter]
if gender_filter != 'All':
  filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
if race_filter != 'All':
  filtered_df = filtered_df[filtered_df['Race'] == race_filter]
if ethnicity_filter != 'All':
  filtered_df = filtered_df[filtered_df['Ethnicity'] == ethnicity_filter]




# Displaying filtered data
st.write('Filtered Data', filtered_df)








def plot_demographic_trends(dataframe, column):


   dataframe['Month'] = dataframe['Date'].dt.strftime('%Y-%m') 
   monthly_counts = dataframe.groupby([column, 'Month']).size().reset_index(name='Counts')
  


   fig = px.line(monthly_counts, x='Month', y='Counts', color=column, title=f'Monthly Changes in {column}',
                 markers=True, labels={'Counts': 'Count'})
  


   fig.update_layout(legend=dict(
       title=column,
       yanchor="top",
       y=0.4,
       xanchor="left",
       x=0,
       bgcolor="rgba(255, 255, 255, 0.5)",
       bordercolor="Black",
       borderwidth=1
   ))


   fig.update_layout(height=600)
  
   st.plotly_chart(fig)










# Analytics options for the new functionalities
st.sidebar.header("Analytics Options")
option = st.sidebar.selectbox(
  "Choose an analysis",
  ["None", "Demographic Changes Over Time", "Mean Survey Responses Over Time","Average Scores by Demographic", "Satisfaction by Provider Demographic","Communication vs. Recording Comfort"],
  index=0
)


# Calculating global average scores
global_average_scores = {}
for score in scores_columns:
  global_average_scores[score] = df[df['Status'] == 'Patient'][score].mean()




# Global average satisfaction
global_average_satisfaction = df[df['Status'] == 'Patient']['Satisfaction'].mean()








def wrap_labels(labels, width=20):
   """Wrap labels to a new line if exceeding the specified width."""
   wrapped_labels = ['\n'.join(textwrap.wrap(label, width)) for label in labels]
   return wrapped_labels




def plot_average_scores_by_demographic(dataframe, demographic):
   for score in scores_columns:
       if dataframe[score].isnull().all():  # Check if all values are NaN
           continue  # Skip if no data is available for the score
      
       # Calculate average scores by demographic
       averages = dataframe.groupby(demographic)[score].mean().reset_index(name='Average')
      
       # Create a bar for demographic averages
       bars = [go.Bar(x=averages[demographic], y=averages['Average'], name='Demographic Average', marker_color='skyblue')]
      
       # Add a bar for the global average if it exists for the score
       global_avg = global_average_scores.get(score)
       if global_avg is not None:
           bars.append(go.Bar(x=['Global Average'], y=[global_avg], name='Global Average', marker_color='salmon'))
      
       # Create the layout for the plot
       layout = go.Layout(
           title=f'Average {score} by {demographic}',
           xaxis=dict(title=demographic),
           yaxis=dict(title='Score'),
           legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
           barmode='group'
       )
      
       # Create the figure and add the bars
       fig = go.Figure(data=bars, layout=layout)
      
       st.plotly_chart(fig)


# Ensuring provider demographics are included in the dataset
df_providers = df[df['Status'] == 'Provider'].copy()
df_providers['Provider_ID'] = df_providers.index




# Adjust the patient data to include a provider ID for merging
df_patients = df[df['Status'] == 'Patient'].copy()




# Merge patient and provider dataframes on the assigned provider
df_merged = df_patients.merge(df_providers[['Provider_ID', 'Gender', 'Race', 'Ethnicity']], left_on='Assigned Provider', right_on='Provider_ID', how='left')








def plot_satisfaction_by_provider_demographic(merged_df, demographic):
   if demographic + '_x' not in merged_df.columns:
       st.write(f"No data available for patient demographic: {demographic}")
       return
   if demographic + '_y' not in merged_df.columns:
       st.write(f"No data available for provider demographic: {demographic}")
       return
    # Calculate satisfaction by provider demographic
   satisfaction_by_provider_demographic = merged_df.groupby(demographic + '_y')['Satisfaction'].mean()
   global_average_satisfaction = merged_df['Satisfaction'].mean()


   # Create the base figure for bar chart
   fig = px.bar(
       satisfaction_by_provider_demographic.reset_index(),
       x=demographic + '_y',
       y='Satisfaction',
       labels={demographic + '_y': f"Provider's {demographic}", 'Satisfaction': 'Average Satisfaction Score'},
       title=f"Patient Satisfaction by Provider's {demographic}"
   )


   # Add a horizontal line for the global average satisfaction
   fig.add_hline(y=global_average_satisfaction, line_dash="dash", line_color="red", annotation_text="Global Average Satisfaction", annotation_position="top right")
  
   # Update the layout to enhance readability and add legend
   fig.update_layout(
       xaxis_title=f"Provider's {demographic}",
       yaxis_title='Average Satisfaction Score',
       xaxis_tickangle=-45,  # Rotate the labels for better visibility
       legend_title="Legend",
       legend=dict(
           yanchor="top",
           y=0.99,
           xanchor="left",
           x=0.01
       )
   )


   st.plotly_chart(fig)






def plot_comm_corr_by_demographic(dataframe, demographic):
   # Filter for patients only
   patient_df = dataframe[dataframe['Status'] == 'Patient'].copy()
  
   # Get unique demographic values
   demographics = patient_df[demographic].unique()
  
   # Iterate over each demographic value
   for dem in demographics:
       # Filter the DataFrame for the current demographic
       dem_df = patient_df[patient_df[demographic] == dem].dropna(subset=['Comfort with Recording Devices', 'Communication Quality'])
      
       # Ensure there's enough data to plot
       if len(dem_df) < 2:
           st.write(f"Not enough data to plot correlations for {demographic}: {dem}")
           continue
      
       # Linear regression with scipy.stats
       slope, intercept, r_value, p_value, std_err = stats.linregress(dem_df['Comfort with Recording Devices'], dem_df['Communication Quality'])
      
       # Create a scatter plot using Plotly Express
       fig = px.scatter(
           dem_df,
           x='Comfort with Recording Devices',
           y='Communication Quality',
           title=f"'Comfort with Recording Devices' vs 'Communication Quality'\n{demographic}: {dem}",
           labels={'Comfort with Recording Devices': 'Comfort with Recording Devices', 'Communication Quality': 'Communication Quality'},
           trendline='ols'  # Automatically adds a regression line using Plotly's OLS
       )


       # Add annotation for the regression equation and R^2 value
       fig.add_annotation(
           x=max(dem_df['Comfort with Recording Devices']),
           y=max(dem_df['Communication Quality']),
           text=f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}',
           showarrow=False,
           xshift=10,
           yshift=10
       )


       st.plotly_chart(fig)








def plot_mean_scores_over_time(dataframe, demographic):
   # Filter the DataFrame for patients only
   patient_df = dataframe[dataframe['Status'] == 'Patient']
  
   # Convert 'Date' to datetime if not already done
   patient_df['Date'] = pd.to_datetime(patient_df['Date'])


   # Iterate over each score in scores_columns
   for score in scores_columns:
       if score not in dataframe.columns or demographic not in dataframe.columns:
           st.error(f"The score {score} or demographic {demographic} is not available.")
           continue  # Skip this score and continue with the next
      
       # Getting unique demographic values
       unique_demographics = patient_df[demographic].unique()
      
       # Prepare a DataFrame to collect all the grouped data for plotting
       all_dem_data = pd.DataFrame()


       for dem in unique_demographics:
           # Filter data for the current demographic
           dem_df = patient_df[patient_df[demographic] == dem]
          
           # Group data by Date and calculate the mean score
           scores_over_time = dem_df.groupby('Date')[score].mean().reset_index()
           scores_over_time[demographic] = dem  # Add a column for demographic to differentiate in plot
          
           # Append the current demographic data
           all_dem_data = pd.concat([all_dem_data, scores_over_time], ignore_index=True)
      
       # Create a Plotly line plot
       fig = px.line(all_dem_data, x='Date', y=score, color=demographic,
                     title=f'{score} Over Time by {demographic}',
                     labels={score: 'Average Score', 'Date': 'Date'},
                     markers=True)
      
       # Improve layout and format date axis
       fig.update_layout(
           xaxis_title='Date',
           yaxis_title='Average Score',
           xaxis_tickangle=-45,
           width = 800
       )
      
       st.plotly_chart(fig) 




# Execute selected analysis
if option == "Average Scores by Demographic":
   st.header("Average Scores by Demographic")
   demographic = st.sidebar.selectbox("Select Demographic", ["Gender", "Race", "Ethnicity"])
   patient_df = filtered_df[filtered_df['Status'] == 'Patient']
   plot_average_scores_by_demographic(patient_df, demographic)
elif option == "Satisfaction by Provider Demographic":
   st.header("Satisfaction by Provider Demographic")
   provider_demographic = st.sidebar.selectbox("Select Provider Demographic", ["Gender", "Race", "Ethnicity"])
   plot_satisfaction_by_provider_demographic(df_merged, provider_demographic)
elif option == "Communication vs. Recording Comfort":
   st.header("Communication vs. Recording Comfort")
   demographic_for_correlation = st.sidebar.selectbox("Select Demographic for Correlation", ["Gender", "Race", "Ethnicity"])
   plot_comm_corr_by_demographic(filtered_df, demographic_for_correlation)
elif option == "Demographic Changes Over Time":
   # Plotting trends if a specific status is selected
   if status_filter != 'All':


       # Visualization for demographic changes over time
       st.header("Demographic Changes Over Time")


       plot_demographic_trends(filtered_df, 'Gender')
       plot_demographic_trends(filtered_df, 'Race')
       plot_demographic_trends(filtered_df, 'Ethnicity')


   else:
       st.header("Please select 'patient' or 'provider' in status menu.")
elif option == "None":
   st.header("Please choose an analysis.")
elif option == "Mean Survey Responses Over Time":
   st.header("Mean Survey Responses Over Time")
   demographic = st.sidebar.selectbox("Select Demographic", ["Gender", "Race", "Ethnicity"])
   patient_df = filtered_df[filtered_df['Status'] == 'Patient']
   plot_mean_scores_over_time(patient_df, demographic)


