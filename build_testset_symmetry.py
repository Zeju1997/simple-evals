import csv
import os
import random

symmetry_data_path = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/svg/symmetry_data/symmetry'
non_symmetry_data_path = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/svg/symmetry_data/non-symmetry'
data_path = 'gb_testset.csv'
output_path = 'gb_symmetry_testset.csv'

question_prompts = [
    "Is the object in the image vertically symmetrical, with both halves being mirror reflections along a central vertical line?",
    "Can the object in the image be divided into two identical halves by a vertical line down the center?",
    "Does the image's object exhibit vertical symmetry, meaning each half mirrors the other along a vertical axis?",
    "If a vertical line is drawn down the center of the object, would both sides be symmetrical reflections?",
    "Is the object symmetrical along a vertical axis, with each side mirroring the other?",
    "Would a vertical line through the center of the object create two mirror-image halves?",
    "Does the object display vertical symmetry, with both halves reflecting each other across a vertical line?",
    "Is there vertical symmetry in the object, meaning both sides mirror each other if split by a vertical line?",
    "Would both halves of the object be identical if a vertical line is drawn down the center?",
    "Is the object in the image vertically symmetrical, creating two mirror images along a central vertical line?",
    "If divided by a vertical line, does the object form two identical halves?",
    "Does the object have vertical symmetry, meaning a vertical line down the center would split it into mirror images?",
    "Is the object symmetrical with respect to a vertical axis, with each side being a reflection of the other?",
    "Would a central vertical line create two mirrored halves of the object in the image?",
    "Does the image depict an object with vertical symmetry, where a vertical line splits it into two identical parts?",
    "If a vertical line is drawn down the center, do both sides of the object mirror each other?",
    "Is the object in the image symmetrically reflected along a vertical axis?",
    "Would both sides of the object be symmetrical if divided by a vertical line?",
    "Is the object in the image vertically symmetrical, with each half being a reflection of the other?",
    "Does the object show vertical symmetry, where a vertical line down the center creates mirror-image halves?"
]



QUESTION_TEMPLATE = """
Examine the following SVG code carefully and answer the question based on your interpretation of the rendered image.

{SVG}

Question: {Question}
""".strip()


symmetry_data = os.listdir(symmetry_data_path)
non_symmetry_data = os.listdir(non_symmetry_data_path)
symmetry_data = [file_name.replace('_copy0.png', '') for file_name in symmetry_data]
non_symmetry_data = [file_name.replace('_copy0.png', '') for file_name in non_symmetry_data]

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    testset = list(reader)


testset_new = []
visited = []
i = 0
for row in testset:
    idx, image_id, question, A, B, C, D, answer, category = row
    image_id = image_id.replace('/', '_')
    if image_id in visited:
        continue
    if image_id in symmetry_data:
        answer = 'Yes'
    elif image_id in non_symmetry_data:
        answer = 'No'
    else:
        continue
    parts = question.split(" The corresponding SVG code to generate image is: ")
    question = QUESTION_TEMPLATE.format(SVG=parts[1].strip(), Question=random.choice(question_prompts))
    testset_new.append([i, question, answer])
    i += 1
    visited.append(image_id)

with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'Question', 'Answer'])
    for row in testset_new:
        writer.writerow(row)


counter = {'Yes': 0, 'No': 0}
with open(output_path, 'r') as f:
    reader = csv.reader(f)
    testset = list(reader)
    for row in testset[1:]:
        idx, question, answer = row
        counter[answer] += 1

print(counter)
