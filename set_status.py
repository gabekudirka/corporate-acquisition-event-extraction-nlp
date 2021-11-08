import os
from get_status_dict import get_status_dictionary


def set_status(filepath):
    status_dict = get_status_dictionary()

    with open(filepath, "r") as d:
        data = d.read()

    for status in status_dict:
        if status in data:
            return status

    return '---'