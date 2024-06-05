import csv
import os
from tqdm import tqdm


data_path = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/svg/mnist_test.csv'
output_file = 'mnist_testset.csv'

testset = []
testset.append(['', 'Question', 'Answer'])

print("Reading", data_path)

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    for i, item in tqdm(enumerate(data)):
        # print(data)
        image_id, question, answer = item
        testset.append([i, question, answer])

print('number of questions', len(testset))

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(testset)