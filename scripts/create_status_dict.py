import json


with open('all_statuses.json') as statuses_json:
    status_dict1 = json.load(statuses_json)

with open('all_statuses_2.json') as statuses_json:
    status_dict2 = json.load(statuses_json)

for status in status_dict1:
    if status in status_dict2:
        status_dict2[status] = status_dict1[status] + status_dict2[status]
    else:
        status_dict2[status] = status_dict1[status]

with open('all_statuses_3.json', 'w') as outfile:
    json.dump(status_dict2, outfile)