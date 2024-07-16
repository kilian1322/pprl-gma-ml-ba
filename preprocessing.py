import numpy as np
import pandas as pd
import random
import string

def random_modify_string(s, change_probability=0.25):
    # Define the set of possible characters (you can customize this as needed)
    possible_characters = string.ascii_letters + string.digits + string.punctuation
    
    # Initialize an empty list to collect the new characters
    new_chars = []
    
    for char in s:
        if random.random() < change_probability:
            # Replace the character with a random character
            new_char = random.choice(possible_characters)
            new_chars.append(new_char)
        else:
            # Keep the original character
            new_chars.append(char)
    
    # Join the list into a new string
    modified_string = ''.join(new_chars)
    
    return modified_string

#OVERLAP = 1

#data = pd.read_csv("./data/feb14.csv")
data = pd.read_csv("./data/Daten_PPRL/fakename_10k.tsv", sep="\t")
#data = data.dropna()
data.replace(np.nan, "", inplace=True)

data = data[["GivenName", "Surname", "Birthday", "uid"]]
data = data.astype({'Birthday': 'str'})


# Creates a list of the indexes in Eve's dataset
#ind = list(range(data.shape[0]))

#data["uid"] = ind
#eve = data

# Random sampling: Shuffle and select the first n=OVERLAP*len(ind) entries
#random.shuffle(ind)
#ind = ind[:int(OVERLAP*len(ind))]
#data = data.iloc[ind]

if False:
    data_red = data.head(5000).copy()
    ind = list(range(data_red.shape[0]))
    data_red["uid"] = ind

    data_disj = data.tail(5000).copy()
    ind = list(range(data_disj.shape[0]))
    data_disj["uid"] = ind
    # Save data
    data_red.to_csv("./data/fakename_5k.tsv", index=False, sep="\t")
    data_disj.to_csv("./data/disjoint_5k.tsv", index=False, sep="\t")
else:
    
    for index, item in data.iterrows():
        for col in item.index:
            row = item[col]
            print(f"Original value at index {index}, column {col}: {row}")
            if not isinstance(row, int):
                new_string = random_modify_string(row, change_probability=0.20)
                print(f"Modified string: {new_string}")
                # Assign the modified string back to the DataFrame
                data.at[index, col] = new_string
        print("\nModified row:")
        print(data.loc[index])
        print("\n")

    data.to_csv("./data/Daten_PPRL/fakename_10k_error_20.tsv", index=False, sep="\t")
#eve.to_csv("./data/eve.tsv", index=False, sep = "\t")

print("Done")





