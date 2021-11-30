import os

testans_directory = './data/testset1/testset1-anskeys/'
gold_template_filepath = './data/testset1/gold.templates'


template_lines = open(gold_template_filepath, "r").readlines()

for i in range(0, len(template_lines), 9):
    doc_lines = template_lines[i:i+8]
    for j, line in enumerate(doc_lines):
        if j == 0:
            ans_name = testans_directory + line[6:].strip() + '.key'
            f = open(ans_name, "a")
        f.write(line)
    f.close()

# for line in template_lines:
#     for i in range(8):
#         if i == 0:
#             ans_name = testans_directory + line[7:].strip() + '.key'
#             f = open(ans_name, "a")
#         f.write(line)
        
#     f.close()
#     continue
