import csv
import os


categories = ["animals", "body", "building", "cloth", "face", "food_drinks_eatable", "music_instruments", "pen_writing", "vehicle"]


data_dir = '/lustre/fast/fast/groups/ps-invsolid/InvSolid/LLaMA-Factory/evaluation/local/data/test'
output_file = 'gb_testset.csv'

testset = []
testset.append(['', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'])

for category in categories:
    data_path = os.path.join(data_dir, f'{category}_test.csv')
    
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        for i, item in enumerate(data):
            image_id, question, A, B, C, D, answer = item
            testset.append([i, question, A, B, C, D, answer, category])

print('number of questions', len(testset))

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(testset)