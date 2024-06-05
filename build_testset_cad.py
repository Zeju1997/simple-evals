import csv
import os
from tqdm import tqdm


data_path = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/MathVista_cad/gb_testmini_final/gb_testmini_full_1000_with_code.csv'
output_file = 'gb_cad_testset.csv'

testset = []
testset.append(['', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'])

print("Reading", data_path)

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    for i, item in tqdm(enumerate(data)):
        image_id, question, A, B, C, D, answer = item
        testset.append([i, question, A, B, C, D, answer, 'QA'])

print('number of questions', len(testset))

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(testset)