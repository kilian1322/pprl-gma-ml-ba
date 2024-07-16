import matplotlib.pyplot as plt
import pandas as pd

# Load the TSV file into a pandas DataFrame
file_path = './data/benchmark_error_BloomFilter_Both.tsv'
df = pd.read_csv(file_path, sep='\t')


# Set the default font sizes
plt.rcParams.update({'font.size': 14})        # General font size
plt.rcParams.update({'axes.titlesize': 18})   # Title font size
plt.rcParams.update({'axes.labelsize': 16})   # Axis labels font size
plt.rcParams.update({'legend.fontsize': 16})  # Legend font size
plt.rcParams.update({'xtick.labelsize': 12})  # X-axis tick labels font size
plt.rcParams.update({'ytick.labelsize': 12})  # Y-axis tick labels font size


# Dictionary to map labels to their order
label_order = {
    '2% error rate': 0,
    '5% error rate': 2,
    '10% error rate': 3,
    '15% error rate': 4,
    '20% error rate': 5,
}

# Get unique dummy values
unique_dummies = df["Data"].unique()
print(unique_dummies)

fig, axs = plt.subplots(3, 2, figsize=(10, 5 * len(unique_dummies)))
axs = axs.flatten()
for ax in axs[4:]:
    fig.delaxes(ax)


handles_labels = {}

# Iterate through each unique dummy value and plot
for ax, dummy_val in zip(axs, unique_dummies):
    print(dummy_val)
    dummy_group = df[df['Data'] == dummy_val]
    print(len(dummy_group))

    for name, group in dummy_group.groupby('Data_error'):
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
    

        
        handle, = ax.plot(group['Overlap'], group['success_rate'], marker='o', linestyle='-', label=label)
        #handles_labels[name] = handle

        if label not in handles_labels:
            handles_labels[label] = handle

        #handles_labels.append((handle, label))

    if "_1k" in dummy_val:
        ax.set_title('1000 fake names')
    elif '_2k' in dummy_val:
        ax.set_title('2000 fake names')
    elif '_5k' in dummy_val:
        ax.set_title('5000 fake names')
    elif '_10k' in dummy_val:
        ax.set_title('10,000 fake names')

    ax.set_xlabel('Overlap')
    ax.set_ylabel('Success Rate')
    #ax.legend(title='Dataset')
    ax.grid(True)
    ax.set_ylim(0, 1)  # Set the same y-axis limits for all subplots


# Sort the handles and labels based on the predefined order
sorted_labels_handles = sorted(handles_labels.items(), key=lambda item: label_order[item[0]])

# Unzip the sorted labels and handles
sorted_labels, sorted_handles = zip(*sorted_labels_handles)

# Add the legend to the figure
fig.legend(sorted_handles, sorted_labels, loc='center right', bbox_to_anchor=(0.31, 0.2), title='Dataset')

 
#plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make space for the legend
plt.show()


