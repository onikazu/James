import matplotlib.pyplot as plt
import pandas as pd
import sys

args = sys.argv

df = pd.read_csv("./logs/{0}_{1}_reward.log".format(args[1], args[2]))
x = df.columns[0]
y = df.columns[1]
ax = df.plot(kind='scatter', x=x, y=y)
df[y] = pd.rolling_mean(df[y], window=20)
df.plot(kind='line', x=x, y=y, ax=ax)
plt.show()

