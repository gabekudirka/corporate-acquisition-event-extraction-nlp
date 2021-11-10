import os
from file_data_extractor_train import extract_files_train

extracted_files = extract_files_train()

f = open('blank_template', "a")   

for file in extracted_files:
    f.write('TEXT: ' + file.text + '\n')
    f.write('ACQUIRED: ---' + '\n')
    f.write('ACQBUS: ---' + '\n') 
    f.write('ACQLOC: ---' + '\n')
    f.write('DLRAMT: ---' + '\n')
    f.write('PURCHASER: ---' + '\n')
    f.write('SELLER: ---' + '\n') 
    f.write('STATUS: ---' + '\n \n')

f.close()