import csv
import os
import random
from collections import Counter, defaultdict


categories = ["accessory", "animal", "book", "clothing", "dairy", "food", "furniture", "human", "musical_instrument", "time_clock", "aerial_crafts", "beverage", "building", "computer", "entertainment", "fruit", "land_crafts", "science", "tool"]

data_dir = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/MathVista_svg/gb_testmini_final_unfiltered'
output_file = 'gb_unfiltered_testset.csv'

testset = []
# testset.append(['', 'Image ID', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'])


# only use qa with 4 identical items
image_count = {}
category_index = {cat: 0 for cat in categories}  

count = 0
for category in categories:
    data_path = os.path.join(data_dir, f'{category}_test.csv')
    print("Reading", data_path)
    
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        count += len(data)
        for i, item in enumerate(data):
            if len(item) != 7 or not all(item):  # Ensure data integrity
                continue
            image_id = item[0]
            if image_id in image_count:
                image_count[image_id].append((item, category))
            else:
                image_count[image_id] = [(item, category)]

for image_id, entries in image_count.items():
    for item, category in entries:
        index = category_index[category]
        image_id, question, A, B, C, D, answer = item
        testset.append([index, image_id, question, A, B, C, D, answer, category])
        category_index[category] += 1 

answer_counts = Counter(row[7] for row in testset)  # Answer is at index 6
total_rows = len(testset)
target_count_per_answer = total_rows // 4

# Identify how many of each answer are needed to reach target
adjustments_needed = {key: target_count_per_answer - answer_counts[key] for key in 'ABCD'}

print('answer_counts before', answer_counts)
print('adjustments_needed before', adjustments_needed)

answer_indices = defaultdict(list)
for index, row in enumerate(testset):
    answer_indices[row[7]].append(index)

# Process each answer type to balance counts
for answer, over_or_under in adjustments_needed.items():
    if over_or_under > 0:  # Need more of this answer
        source_answers = [k for k, v in adjustments_needed.items() if v < 0]
        for source in source_answers:
            while adjustments_needed[answer] > 0 and adjustments_needed[source] < 0:
                # Randomly pick and swap an entry from source to answer
                source_index = random.choice(answer_indices[source])
                answer_indices[source].remove(source_index)
                answer_indices[answer].append(source_index)

                # Swap in data
                old_answer = testset[source_index][7]  # Adjusted index for answer
                testset[source_index][7] = answer  # Adjusted index for answer
                # Swapping corresponding options (adjusted indices for options)
                testset[source_index][3 + 'ABCD'.index(old_answer)], testset[source_index][3 + 'ABCD'.index(answer)] = \
                    testset[source_index][3 + 'ABCD'.index(answer)], testset[source_index][3 + 'ABCD'.index(old_answer)]

                # Update counts
                adjustments_needed[answer] -= 1
                adjustments_needed[source] += 1


answer_counts = Counter(row[7] for row in testset)  # Answer is at index 6
total_rows = len(testset)
target_count_per_answer = total_rows // 4

# Identify how many of each answer are needed to reach target
adjustments_needed = {key: target_count_per_answer - answer_counts[key] for key in 'ABCD'}

print('answer_counts after', answer_counts)
print('adjustments_needed after', adjustments_needed)


with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(testset)