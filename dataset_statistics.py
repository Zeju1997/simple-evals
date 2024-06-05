import csv

benchmark_path = "/lustre/fast/fast/groups/ps-invsolid/InvSolid/simple-evals/gb_main_testset.csv"


with open(benchmark_path, mode='r', newline='') as f:
    reader = csv.reader(f)
    testset_main = list(reader)
    
    
header = testset_main[0]
counter = {}

for row in testset_main[1:]:
    cat = row[-1]
    if cat not in counter:
        counter[cat] = 0
    counter[cat] += 1

print(counter)