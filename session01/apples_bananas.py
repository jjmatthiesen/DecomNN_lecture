import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd


# %%
df = pd.read_csv('data/apple_bananas.csv')
df['apple'] = [1 if x == 'Apple' else 0 for x in df['fruit']]
df.drop('fruit', axis=1, inplace=True)
# data without mini banana
df_mini = df.iloc[0:-1]

c_list = ['#c99642', '#803841']
labels = ['banana', 'apple']


# %%
def plot_data(dataframe, dim1, dim2, save=True):
    target = np.array(dataframe['apple'])
    f, ax = plt.subplots(1)
    for i in np.unique(target):
        mask = target == i
        plt.scatter(dataframe[dim1][mask], dataframe[dim2][mask], c=c_list[i], label=labels[i])
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    if save:
        now = datetime.now()
        plt.savefig('img/apple_banana' + now.strftime("%H%M%S") + '.png')
    else:
        plt.show()


# %%
plot_data(df_mini, 'length', 'height', save=False)
plot_data(df, 'length', 'height', save=False)
plot_data(df, 'length', 'soft', save=False)
