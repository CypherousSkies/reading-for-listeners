from scipy.stats import linregress
import csv
import numpy as np

with open('time_data.csv') as f:
    reader = csv.reader(f)
    data = np.array(list(reader),dtype='float')
    xs = list(data.T[0])
    ys = list(data.T[1])
    result = linregress(xs,ys)
print(f"time = {result.slope}*words+{result.intercept} with r={result.rvalue}")
