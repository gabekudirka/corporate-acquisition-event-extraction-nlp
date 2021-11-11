import os

docs_directory = '.\data\docs'
doc_files = os.listdir(docs_directory)
f = open('full_doclist', 'a')

for docs_filename in doc_files:
    doc_filepath = os.path.join(docs_directory, docs_filename)
    f.write(doc_filepath + '\n')

f.close()