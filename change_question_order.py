import csv

data_path = 'gb_unfiltered_testset.csv'
output_path = 'gb_raw_testset.csv'


QUESTION_TEMPLATE = """
Examine the following SVG code carefully and answer the question based on your interpretation of the rendered image.

{SVG}

Question: {Question}
""".strip()

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    testset = list(reader)

testset_new = []
for row in testset[1:]:
    idx, image_id, question, A, B, C, D, answer, category = row
    parts = question.split(" The corresponding SVG code to generate image is: ")
    question = QUESTION_TEMPLATE.format(SVG=parts[1].strip(), Question=parts[0])
    testset_new.append([idx, question, A, B, C, D, answer, category])

with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'])
    for row in testset_new:
        writer.writerow(row)


counter = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
with open(output_path, 'r') as f:
    reader = csv.reader(f)
    testset = list(reader)
    for row in testset[1:]:
        idx, question, A, B, C, D, answer, category = row
        counter[answer] += 1

print(counter)
