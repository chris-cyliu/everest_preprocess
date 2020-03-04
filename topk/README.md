# TOP-K 

The top-k algorithm here (ZL and CX)

### Example of csv file
```
timestamp,prob
0.09106963020656711,[0.30468503 0.37727905 0.22411554 0.09392038]
0.5417467517178416,[0.42740343 0.01290236 0.24133435 0.31835987]
0.4995340785341853,[0.09151351 0.50143734 0.2623566  0.14469255]
```

### How to read and write

#### Save
```
csv_path = './test.csv'
table = [
    {'timestamp': 0.1, 'prob': [0.1, 0.2, 0.7]},
    {'timestamp': 0.3, 'prob': [0.5, 0.3, 0.2]}
    {'timestamp': 0.4, 'prob': [0.3, 0.3, 0.4]}
]

with open(csv_path, 'w', newline='') as f:
    fieldnames = ['timestamp', 'prob']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for row in table:
        writer.writerow(row)
```

#### Read
```
csv_path = './test.csv'
table = []

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        table.append(row)
```
