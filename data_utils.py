import csv


data = []
with open('train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        data.append(row)

print(len(data))
print(data[0])