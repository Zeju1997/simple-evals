import csv
import os
import random
import shutil

full_benchmark_dir = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/svg/benchmark'
benchmark_dir = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/svg/benchmark_mini'

data_path = 'gb_testset.csv'
output_file = 'gb_testset_mini.csv'

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    testset = list(reader)
    random.shuffle(testset)
    testset_mini = testset[:500]

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in testset_mini:
        idx, image_id, question, A, B, C, D, answer, category = row
        parts = question.split(" The corresponding SVG code to generate image is: ")

        cat, image_name = image_id.split('/')
        image_name += '.png'
        source_path = os.path.join(full_benchmark_dir, cat, image_name)
        target_dir = os.path.join(benchmark_dir, cat)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        shutil.copy(source_path, os.path.join(target_dir, image_name))

        new_row = [image_id, parts[0], A, B, C, D, answer, category]
        writer.writerow(new_row)

print('number of questions', len(testset_mini))