import json

def set_status(filepath):
    with open('all_statuses.json') as statuses_json:
        status_dict = json.load(statuses_json)

    with open(filepath, "r") as d:
        data = d.read()

    for status in status_dict:
        if status in data:
            return status

    return '---'

set_status('hi')