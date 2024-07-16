import matplotlib.pyplot as plt
import pandas as pd

# Load the TSV file into a pandas DataFrame
file_path = './data/benchmark_BloomFilter_Both_GraphMatching_2.tsv'
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
    'Titanic': 0,
    '1000 Fake Names': 2,
    '2000 Fake Names': 3,
    '5000 Fake Names': 4,
    '10,000 Fake Names': 5,
}

# Get unique dummy values
unique_dummies = ["MinWeight", "Stable", "Symmetric", "NearestNeighbor"]

fig, axs = plt.subplots(3, 2, figsize=(10, 5 * len(unique_dummies)))
axs = axs.flatten()
for ax in axs[4:]:
    fig.delaxes(ax)


handles_labels = {}

# Iterate through each unique dummy value and plot
for ax, dummy_val in zip(axs, unique_dummies):
   
    dummy_group = df[df['Matching'] == dummy_val]

    for name, group in dummy_group.groupby('Data'):
        if 'titanic' in name:
            label = 'Titanic'
        elif '_1k' in name:
            label = '1000 Fake Names'
        elif '_10k' in name:
            label = '10,000 Fake Names'
        elif '_2k' in name:
            label = '2000 Fake Names'
        elif '_5k' in name:
            label = '5000 Fake Names'
      
        else:
            label = name
    

        
        handle, = ax.plot(group['Overlap'], group['wrong_rate'], marker='o', linestyle='-', label=label)
        handles_labels[name] = handle

        #handles_labels.append((handle, label))


    ax.set_title(f'{dummy_val} Matching')
    ax.set_xlabel('Overlap')
    ax.set_ylabel('False Positive Rate')
    #ax.legend(title='Dataset')
    ax.grid(True)
    ax.set_ylim(0, 1)  # Set the same y-axis limits for all subplots



 
handles, labels = list(handles_labels.values()), list(handles_labels.keys())
#labels = dict(sorted(labels, key=lambda item: label_order[item[0]]))
 
modified_labels = []
print(labels)
for name in labels:
    if '_1k' in name:
        modified_labels.append('1000 Fake Names')
    elif '_10k' in name:
        modified_labels.append('10,000 Fake Names')
    elif '_2k' in name:
        modified_labels.append('2000 Fake Names')
    elif '_5k' in name:
        modified_labels.append('5000 Fake Names')
    elif 'titanic' in name:
        modified_labels.append('Titanic')
    else:
        #modified_labels.append(name)
        pass

# Sort the modified labels based on label_order
sorted_labels_handles = sorted(zip(modified_labels, handles), key=lambda item: label_order.get(item[0], float('inf')))

# Unzip the sorted labels and handles
sorted_labels, sorted_handles = zip(*sorted_labels_handles)
fig.legend(sorted_handles, sorted_labels, loc='center right', bbox_to_anchor=(0.31, 0.2), title='Dataset')

#plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make space for the legend
plt.show()


