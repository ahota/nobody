#!/usr/bin/env python
import sys, random
#generate an nxn identity matrix

if len(sys.argv) < 3:
    print 'Usage: ' + sys.argv[0] + ' n filename'
    sys.exit()

transpose = False
r = False
if len(sys.argv) > 3:
    if sys.argv[3] == '-t':
        transpose = True
    if sys.argv[3] == '-r':
        r = True
        random.seed()

n    = int(sys.argv[1])
path = sys.argv[2]

with open(path, 'w') as outfile:
    outfile.write(str(n) + ' ' + str(n))
    outfile.write('\n')
    for i in range(n):
        for j in range(n):
            if r:
                outfile.write("%.5f" % random.uniform(0, 100) + ' ')
            elif transpose and i == j:
                outfile.write("%.5f" % 1.0 + ' ')
            elif not transpose and i == n - j - 1:
                outfile.write("%.5f" % 1.0 + ' ')
            else:
                outfile.write("%.5f" % 0.0 + ' ')
        outfile.write('\n')

print 'Done.'
