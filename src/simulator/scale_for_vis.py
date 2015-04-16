import sys
import csv

MAX = 100
results = []
with open(sys.argv[1], 'r') as f:
    f.readline() # pass over configs
    for line in f:
        split = line.split(",")
        results.append([float(split[0]) / MAX, float(split[1]) / MAX, float(split[2]) / MAX])

writer = csv.writer(open('simulation.csv', 'wb'))
writer.writerows(results)
