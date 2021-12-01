import json
from collections import defaultdict
from file_data_extractor_train import extract_files_train


def default_value():
    return 0

def get_status_dictionary():
    docs = extract_files_train()
    status_dictionary = defaultdict(default_value)

    for doc in docs:
        status = doc.get_status()
        if status != '---':
            status_dictionary[doc.get_status()] += 1

    return status_dictionary

dict = get_status_dictionary()
with open('all_statuses_2.json', 'w') as outfile:
    json.dump(dict, outfile)