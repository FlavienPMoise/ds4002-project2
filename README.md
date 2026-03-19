# DS 4002 - SFW Hourly Pollution Levels Analysis

## Repository Overview
This repository has all the files and necessary documentation for DS 4002 Project 2, a project analyzing the pollution levels in an Italian city over time. This project specifically looks at how day/night and weekday/weekend cycles effect pollution levels. 

## 1. Software and Platform

### Software Used
- **Python**

### Required Python Packages
The following add-on packages are required to run the project:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

### Platform
This project was developed and tested on: **Windows** and **Linux**

## 2. Documentation Map (Project Structure)

Below is an outline of the folder and file structure of the project repository:
```
Project-Folder/
├── LICENSE
├── README.md
├── how-to-write-a-readme.md
├── data/
│   ├── AirQualityUCI.csv
│   ├── Data Appendix.pdf
│   └── day_weekday_data_flags.csv
├── output/
│   ├── day_weekday_correlations.csv
│   ├── linear_reg_day_weekday_results.csv
│   ├── linear_regression_predictive_evaluation.csv
│   ├── linear_regression_test_metrics_by_pollutant.csv
│   ├── figure1_measurements_over_time.png
│   ├── figure2_day_night.png
│   ├── figure3_weekday_weekend.png
│   ├── figure4_volatility.png
│   ├── figure5_correlation_heatmap.png
│   └── day_weekday_correlations.png
└── scripts/
    ├── correlations.py
    ├── eda.py
    └── linear_reg.py
```

**Folder descriptions:**
- `data/`: Contains the raw csv dataset and the final version used for statistical analysis.
- `output/`: Output files (exploratory and analysis-based plots).
- `scripts/`: Contains the Python scripts used to perform EDA, linear regression, and Pearson's correlation tests. 

## 3. Instructions for Reproducing Results
Here is an outline for how we produced our results: 

### Step 1: Clone the Repository
1. Open a terminal or command prompt.
2. Clone the repository: `git clone https://github.com/FlavienPMoise/ds4002-project2`
3. Navigate into the project directory by doing `cd ds4002-project-2`

### Step 2: Set up virtual environment
1. Ensure that the latest version of Python is installed and create a virtual environment: `python3 -m venv venv`
2. Activate the virtual environment:
   Windows: `venv\Scripts\activate`
   macOS/Linux: `source venv/bin/activate`
4. Install the required packages as mentioned above using `pip install [PACKAGE NAMES HERE]`

### Step 3: Prepare the dataset
1. Download the pollution dataset from UCI at https://archive.ics.uci.edu/dataset/360/air+quality, or use the one already in the data folder. 
2. Unzip and place the CSV file in the data/ directory, or use the one already there. 
3. Ensure the file name matches `AirQualityUCI.csv`

### Step 4: Run EDA
1. Run the EDA script: `python3 src/EDA.py`
2. This script generates exploratory plots in the output/ folder. 

### Step 5: Run linear regression
1. Run the linear regression script: `python linear_reg.py`
2. This script generates correlation and test values in  various .csv's the output/ folder, and a new data file in the data/folder

### Step 6: Run correlation analysis
1. Run the final correlation analysis: `python correlations.py`
2. This script generates correlations and p values, saving them in a table .csv in the output/ folder. 

### Step 7: Review Final Results
1. Inspect plots and tables in the output/ directory.