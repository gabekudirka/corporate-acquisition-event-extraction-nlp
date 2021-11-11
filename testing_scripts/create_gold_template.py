import os

keys_directory = './data/anskeys'
key_files = os.listdir(keys_directory)

f = open('gold_template', "a")   

for file in key_files:
    key_filepath = os.path.join(keys_directory, file)
    key = open(key_filepath, "r").read()
    f.write(key)

f.close()