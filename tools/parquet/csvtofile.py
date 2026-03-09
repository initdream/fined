import csv

with open('filename.csv', 'r') as df:
    reader = csv.reader(df)
    for rownumber, row in enumerate(reader):
        with open(f'{rownumber}.txt', 'w') as f:
            f.write('\n'.join(row) + '\n')
