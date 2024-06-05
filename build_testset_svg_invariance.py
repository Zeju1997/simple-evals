import csv
import os
import random
from collections import Counter, defaultdict


import csv

data_path = 'gb_testset.csv'
output_path = 'gb_inv_testset.csv'

translated_data_dir = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/svg/augmented_gb_testset_translate'
rotated_data_dir = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/svg/augmented_gb_testset_rotate'
selected_data_path = 'filtered_list.txt'
selected_data_subset_path = 'filtered_list_subset.txt'

QUESTION_TEMPLATE = """
Examine the following SVG code carefully and answer the question based on your interpretation of the rendered image.

{SVG}

Question: {Question}
""".strip()

'''
with open(selected_data_path, 'r') as f:
    selected_data = f.read().splitlines()

random.shuffle(selected_data)
selected_data_subset = selected_data[:300]

with open(selected_data_subset_path, 'w') as file:
    # Write each string to the file
    for line in selected_data_subset:
        file.write(line + '\n')
'''

with open(selected_data_subset_path, 'r') as f:
    selected_data_subset = f.read().splitlines()

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    testset = list(reader)

testset_new_orig = []
testset_new_translation = {key: [] for key in range(5)}
testset_new_rotation = {key: [] for key in range(5)}
for row in testset:
    idx, image_id, question, A, B, C, D, answer, category = row
    if image_id not in selected_data_subset:
        continue
    parts = question.split(" The corresponding SVG code to generate image is: ")
    question = parts[0]
    orig_code = parts[1].strip()
    testset_new_orig.append([idx, QUESTION_TEMPLATE.format(SVG=orig_code, Question=question), A, B, C, D, answer, category])

    for i in range(5):
        cat, image_name = image_id.split('/')
        translated_code_path = os.path.join(translated_data_dir, f'{cat}_{image_name}_copy{i}.svg')
        with open(translated_code_path, 'r', encoding='utf-8') as file:
            translated_svg = file.read()
        testset_new_translation[i].append([idx, QUESTION_TEMPLATE.format(SVG=translated_svg, Question=question), A, B, C, D, answer, category])

        rotated_code_path = os.path.join(rotated_data_dir, f'{cat}_{image_name}_copy{i}.svg')
        with open(rotated_code_path, 'r', encoding='utf-8') as file:
            rotated_svg = file.read()
        testset_new_rotation[i].append([idx, QUESTION_TEMPLATE.format(SVG=rotated_svg, Question=question), A, B, C, D, answer, category])

with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'])
    for row in testset_new_orig:
        writer.writerow(row)

for i in range(5):
    with open(output_path.replace('_inv', f'_inv_t{i}'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'])
        for row in testset_new_translation[i]:
            writer.writerow(row)
    
    with open(output_path.replace('_inv', f'_inv_r{i}'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'])
        for row in testset_new_rotation[i]:
            writer.writerow(row)