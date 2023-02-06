import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv('test.csv', header=None)
    ax = df.plot.scatter(x=0, y=1, style='orange')
    df.plot.line(x=0, y=1, ax=ax, xlabel="Episodes trained", ylabel="Average reward", legend=False, style='cyan')
    plt.savefig('plot.png')
