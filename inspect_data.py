import csv


data_path = 'mmlu.csv'


with open(data_path, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    for i, item in enumerate(data):
        print(item)
        print('---')
        if i == 10:
            break