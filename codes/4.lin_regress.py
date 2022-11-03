import matplotlib.pyplot as plt
from scipy import stats
import csv

f = open("sample data\data.csv", 'r')
reader = csv.reader(f)
next(reader)
data = [list(map(eval, i)) for i in reader]
x = [i[0] for i in data]
y = [i[1] for i in data]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show() 
