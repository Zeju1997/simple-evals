import csv

benchmark_path = "/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/gb_main_testset.csv"


with open(benchmark_path, mode='r', newline='') as f:
    reader = csv.reader(f)
    testset_main = list(reader)
    
    
header = testset_main[0]
testset_semantics = []
testset_color = []
testset_shape = []
testset_count = []
testset_reasoning = []

for row in testset_main[1:]:
    question = row[1]
    ind = row[0]

    if int(ind) % 4 == 0:
        testset_semantics.append(row)

    elif 'color' in question:
        testset_color.append(row)

    elif 'count' in question or 'many' in question:
        testset_count.append(row)

    elif 'shape' in question or 'line' in question or 'geometry' in question:
        testset_shape.append(row)

    else:
        testset_reasoning.append(row)


print('number of semantics question:', len(testset_semantics))
print('number of color question:', len(testset_color))
print('number of shape question:', len(testset_shape))
print('number of count question:', len(testset_count))
print('number of reasoning question:', len(testset_reasoning))
print('total number of questions:', len(testset_semantics) + len(testset_color) + len(testset_count) + len(testset_shape) + len(testset_reasoning))
exit()

with open('gb_semantics_testset.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(testset_semantics)

with open('gb_color_testset.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(testset_color)

with open('gb_shape_testset.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(testset_shape)

with open('gb_reasoning_testset.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(testset_reasoning)

with open('gb_count_testset.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(testset_count)