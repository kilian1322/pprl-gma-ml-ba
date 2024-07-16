import numpy as np
import pandas as pd
import random


"""
data = pd.read_csv("./data/Daten_PPRL/fakename_dummy_10k.tsv", sep="\t")
print(data.head())

data['uid'] = data['uid'] * 10000




data.to_csv("./data/fakename_dummy_10k.tsv", index=False, sep="\t") """

alice_uids_new = [10,7,3,5,2,23]
alice_uids_old = [5,10,23,7,3,2]
alice_enc_data = ['a', 'b', 'c', 'd', 'e', 'f', 'a']
eve_enc_data = ['f', 'b', 'a', 'e', 'd','a', 'c']

uid_to_index_new = {uid: index for index, uid in enumerate(alice_uids_new)}

print(uid_to_index_new)
print(alice_uids_new)

user_data_sorted = [None] * len(alice_uids_old)
for old_index, uid in enumerate(alice_uids_old):
    new_index = uid_to_index_new[uid]
    user_data_sorted[old_index] = alice_enc_data[new_index]
alice_enc_data = user_data_sorted
print(len(alice_enc_data))
print(alice_enc_data) 


dict_equal_entries = {}
i, j = 0, 0
for alice_entry in alice_enc_data:
    for eve_entry in eve_enc_data:
        if np.array_equal(alice_entry, eve_entry):
            dict_equal_entries[i] = j
            j = 0
            continue
        j += 1
    j = 0 
    i += 1

#print(dict_equal_entries)