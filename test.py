import myNet
import csv
import math
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
    content  = f.read()
    for x in content:
        print(x)


