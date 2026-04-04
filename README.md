# Statistical and Probability-Based Analysis of OPD Patient Arrival Patterns for Better Appointment Planning and Queue Management

A complete Streamlit-based Python project for analyzing hospital OPD patient arrival patterns using statistics, probability, prediction, and simulation techniques. This project uses synthetic but realistic datasets to study patient flow, waiting time behavior, peak-hour trends, appointment load, and queue management scenarios.

## Project Objective

The main objective of this project is to analyze OPD patient arrivals and use data-driven methods to support:

- better appointment planning
- improved queue management
- reduced waiting time
- better understanding of peak-hour pressure
- department-wise OPD load analysis
- simulation-based planning for overload conditions

This project is designed as a college experiential learning project and is fully runnable locally using Streamlit.

## Features

The application includes the following sections:

### 1. Home
- project overview
- quick metrics
- peak-hour and peak-day summary
- overcrowding and waiting-time highlights

### 2. Dataset Overview
- preview of all generated datasets
- row and column count
- column names
- missing value check

### 3. Statistical Analysis
- descriptive statistics:
  - mean
  - median
  - mode
  - variance
  - standard deviation
  - range
  - quartiles
  - interquartile range
- hour-wise patient analysis
- day-wise patient analysis
- department-wise analysis
- correlation analysis
- linear regression analysis

### 4. Probability and Prediction
- waiting time prediction from patient count
- overload risk prediction
- overcrowding probability
- high waiting time probability
- peak-hour probability by hour
- peak-day probability
- department peak probability
- walk-in vs booked probability
- Poisson probability estimation for patient arrivals

### 5. Simulation
- queue simulation for a selected day
- scenario comparison using arrival and capacity adjustment
- Monte Carlo simulation for operational risk estimation

### 6. Report Insights
- report-ready summary text
- presentation-friendly findings
- key observations for submission or viva explanation

## Project Structure

```text
opd_patient_arrival_app/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── sample_generator.py
│   ├── opd_patient_data.csv
│   ├── department_wise_opd_data.csv
│   ├── doctor_schedule_data.csv
│   ├── appointment_data.csv
│   └── patient_category_data.csv
│
└── utils/
    ├── __init__.py
    ├── data_loader.py
    ├── stats_utils.py
    ├── predictor.py
    ├── probability_utils.py
    ├── simulation_utils.py
    └── chart_utils.py
Technologies Used
Python
Streamlit
Pandas
NumPy
Matplotlib
Synthetic Datasets Used

The project automatically generates realistic synthetic OPD datasets.

1. opd_patient_data.csv

Contains overall hourly OPD data.

Columns:

date
day
hour_slot
patients_arrived
doctors_available
capacity_per_hour
avg_consultation_time
avg_waiting_time
load_ratio
peak_indicator
2. department_wise_opd_data.csv

Contains department-specific hourly OPD data.

Columns:

date
day
hour_slot
department
patients_arrived
doctors_available
capacity_per_hour
avg_consultation_time
avg_waiting_time
load_ratio
peak_indicator
3. doctor_schedule_data.csv

Contains doctor schedule information.

Columns:

date
day
department
doctor_name
shift_start
shift_end
max_patients_per_hour
4. appointment_data.csv

Contains appointment and walk-in details.

Columns:

date
day
hour_slot
department
booked_appointments
walk_in_patients
total_expected_patients
actual_patients
5. patient_category_data.csv

Contains patient type/category information.

Columns:

date
day
hour_slot
department
patient_category
count
Installation
Step 1: Clone or create the project folder

Create the folder structure exactly as shown above.

Step 2: Install dependencies

Open terminal in the project root folder and run:

pip install -r requirements.txt

If pip does not work on Windows, use:

py -m pip install -r requirements.txt
How to Run the Project
Step 1: Generate the datasets

From the project root folder, run:

python data/sample_generator.py

If needed on Windows:

py data/sample_generator.py

This will generate all required CSV files inside the data/ folder.

Step 2: Run the Streamlit app

From the project root folder, run:

python -m streamlit run app.py

If needed:

py -m streamlit run app.py
Notes for PyCharm Users
Open the project using the root folder: opd_patient_arrival_app
Make sure all files are placed exactly in the correct folders
Run all commands from the project root
Ensure utils/__init__.py exists
Do not rename files such as:
stats_utils.py
probability_utils.py
simulation_utils.py
chart_utils.py

If file names are changed, imports in app.py will fail.

Important Stability Notes

This project is intentionally designed to be simple and stable.

uses minimal dependencies
avoids fragile package assumptions
keeps imports straightforward
uses local CSV generation
avoids naming collisions with built-in Python modules
uses modular files without overengineering
Example Analysis Included

The project can be used to study:

which hour slot has the highest patient load
which day has the highest average OPD pressure
which department experiences the highest waiting time
relationship between patient count and waiting time
risk of overcrowding
waiting-time probability
queue growth under different scenarios
Monte Carlo risk behavior under uncertain arrivals
Academic Use

This project is suitable for:

experiential learning project
mini project
statistics-based hospital analytics demo
probability-based queue analysis
Streamlit data app demonstration
report and presentation submission
Troubleshooting
1. ModuleNotFoundError

Make sure:

you are running from the project root folder
file names match exactly
utils/__init__.py exists
2. CSV files not found

Run:

python data/sample_generator.py

before starting the Streamlit app.

3. Streamlit not opening

Try:

py -m streamlit run app.py
4. Wrong import errors in PyCharm

Do not create extra conflicting files like:

statistics.py
probability.py
simulation.py
charts.py

These can interfere with imports.

Future Improvements

Possible future extensions:

use real hospital OPD data
add patient no-show probability
include doctor utilization dashboard
add queue optimization suggestions
integrate machine learning forecasting
export report as PDF
add department-specific simulation controls
Conclusion

This project demonstrates how statistical analysis, probability estimation, predictive modeling, and simulation can be combined to better understand OPD patient arrival patterns. It provides a practical and presentation-ready approach to better appointment planning and queue management in a hospital OPD environment.

Developed as a Python + Streamlit experiential learning project by Math Debaters.

Team Members:
Namgay D Wangchuk
Prajwal C Pradhan
Malavika Vinod
Monish Kandanuru
Pakalapati Monkika
Mumukka Sanjana Reddy
P. Navya Shree