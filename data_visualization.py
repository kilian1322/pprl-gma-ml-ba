import pandas as pd
import matplotlib.pyplot as plt



# Load the TSV file into a pandas DataFrame
file_path = './data/benchmark_error_BloomFilter_Both_2.tsv'
df = pd.read_csv(file_path, sep='\t')


# Set the default font sizes
plt.rcParams.update({'font.size': 14})        # General font size
plt.rcParams.update({'axes.titlesize': 18})   # Title font size
plt.rcParams.update({'axes.labelsize': 16})   # Axis labels font size
plt.rcParams.update({'legend.fontsize': 14})  # Legend font size
plt.rcParams.update({'xtick.labelsize': 12})  # X-axis tick labels font size
plt.rcParams.update({'ytick.labelsize': 12})  # Y-axis tick labels font size

#grouped_data = df.groupby('Data')
#grouped_data = grouped_data[df['Data'] == "./data/Daten_PPRL/fakename_1k.tsv"]

filtered_df = df[df['Data'] == './data/Daten_PPRL/fakename_1k.tsv']

# Group the filtered data by the 'Category' column
grouped_data = filtered_df.groupby('Data_error')

# Dictionary to map labels to their order
""" label_order = {
    'Titanic': 0,
    '1000 Fake Names': 1,
    '2000 Fake Names': 2,
    '5000 Fake Names': 3,
    '10,000 Fake Names': 4,
    '20,000 Fake Names': 5,
    '50,000 Fake Names': 6
} """

# Dictionary to map labels to their order
label_order = {
    '2% error rate': 0,
    '5% error rate': 2,
    '10% error rate': 3,
    '15% error rate': 4,
    '20% error rate': 5,
}
# Plot the data
plt.figure(figsize=(12, 3.5))

handles_labels = []

for name, group in grouped_data:
    """ if 'titanic' in name:
        label = 'Titanic'
    elif '_1k' in name:
        label = '1000 Fake Names'
    elif '_10k' in name:
        label = '10,000 Fake Names'
    elif '_2k' in name:
        label = '2000 Fake Names'
    elif '_5k' in name:
        label = '5000 Fake Names'
    elif '_20k' in name:
        label = '20,000 Fake Names'
    elif '_50k' in name:
        label = '50,000 Fake Names'
    else:
        label = name """
    if 'error_2.' in name:
        label = '2% error rate'
    elif 'error_5' in name:
        label = '5% error rate'
    elif 'error_10' in name:
        label = '10% error rate'
    elif 'error_15' in name:
        label = '15% error rate'
    elif 'error_20' in name:
        label = '20% error rate'
    else:
        label = name
  
    handle, = plt.plot(group['Overlap'], group['wrong_rate'], marker='o', linestyle='-', label=label)
    handles_labels.append((handle, label))

# Sort the handles and labels based on the label order
handles_labels.sort(key=lambda hl: label_order[hl[1]])

# Unzip the sorted handles and labels
handles, labels = zip(*handles_labels)

plt.subplots_adjust(bottom=0.20, top=0.98)  # Adjust the bottom parameter to fit the plot

plt.xlabel('Overlap')
plt.ylabel('Elapsed Time')
plt.grid(True)

# Create legend with sorted labels
plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Dataset")
#plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect parameter to fit the legend
plt.subplots_adjust(right=0.75)  # Adjust the right parameter to fit the legend



plt.show()



# Plot the df
#plt.figure(figsize=(10, 6))
#plt.plot(df['Overlap'], df['success_rate'], marker='o', linestyle='-', color='b')
#plt.title('Success Rate vs Overlap')















