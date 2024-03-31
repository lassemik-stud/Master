import pandas as pd
import glob
import json
import re

# Initialize an empty DataFrame
df = pd.DataFrame()

# Define the regex pattern
pattern = r'.*/report-s(?P<s>\d+)-se(?P<s_s>\d+)-fa(?P<f_a>\w+)-ft-(?P<f_t>\w+)-fng(?P<fe_ng>\d\d)-fm(?P<fe_m_f>\d+)-k(?P<k>\d+)-d(?P<d>\d+)-svm-c(?P<svc_c>\d+)-d(?P<s_d>\d+).txt'
# Iterate over all .txt files in the results directory
for filename in glob.glob('/home/lasse/Master/results/*.txt'):
    #print(f'Reading file: {filename}')  # Debugging line

    # Use the re.match function to match the pattern in the filename
    match = re.match(pattern, filename)

    # If a match is found, extract the groups to a dictionary
    if match:
        #print('Regex matched')  # Debugging line
        file_info = match.groupdict()

        # Open the file and load the content as a JSON
        with open(filename, 'r') as f:
            content = json.load(f)
            #print(f'Loaded JSON content: {content}')  # Debugging line

        # Extract the required fields from the JSON object
        # Define the number of decimal places
        decimal_places = 2

        fields = {
            'p_0': round(content['0']['precision'], decimal_places),
            'r_0': round(content['0']['recall'], decimal_places),
            'f1_0': round(content['0']['f1-score'], decimal_places),
            's_0': content['0']['support'],
            'p_1': round(content['1']['precision'], decimal_places),
            'r_1': round(content['1']['recall'], decimal_places),
            'f1_1': round(content['1']['f1-score'], decimal_places),
            's_1': content['1']['support'],
            'a': round(content['accuracy'], decimal_places),
            'p_m_avg': round(content['macro avg']['precision'], decimal_places),
            'r_m_avg': round(content['macro avg']['recall'], decimal_places),
            'f1_m_avg': round(content['macro avg']['f1-score'], decimal_places),
            's_m_avg': content['macro avg']['support'],
            'p_wavg': round(content['weighted avg']['precision'], decimal_places),
            'r_wavg': round(content['weighted avg']['recall'], decimal_places),
            'f1w_avg': round(content['weighted avg']['f1-score'], decimal_places),
            's_w_avg': content['weighted avg']['support']
        }

        # Combine the file_info and fields dictionaries
        data = {**file_info, **fields}

        # Convert the dictionary to a DataFrame and append it to the main DataFrame
        df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

# Print the DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 250)

from colorama import Fore, Style
from tabulate import tabulate

# Convert the DataFrame to a list of lists
df = df.sort_values(by=['a'])
data = df.values.tolist()

# Convert the column names to a list
columns = df.columns.tolist()

# Find the indices of the 'p_0' and 'p_1' columns
p0_index = columns.index('p_0')
p1_index = columns.index('p_1')
p2_index = columns.index('a')

# Iterate over the data and colorize the 'p_0' and 'p_1' columns
# Iterate over the data and colorize the 'p_0' and 'p_1' columns
for row in data:
    # Round the numbers to 2 decimal places before converting them to strings
    row[p0_index] = f"{Fore.YELLOW}{round(row[p0_index], 2)}{Style.RESET_ALL}"
    row[p1_index] = f"{Fore.YELLOW}{round(row[p1_index], 2)}{Style.RESET_ALL}"
    row[p2_index] = f"{Fore.RED}{round(row[p2_index], 2)}{Style.RESET_ALL}"

# Print the colorized data
print(tabulate(data, headers=columns))