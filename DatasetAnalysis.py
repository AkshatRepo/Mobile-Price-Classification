import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv('train.csv')
dataset.head()
dataset.info()
dataset.describe()
sns.pairplot(dataset,hue='price_range')
sns.jointplot(x='ram',y='price_range',data=dataset,color='red',kind='kde');
sns.pointplot(y="int_memory", x="price_range", data=dataset)
labels = ["3G-supported",'3G Not supported']
values=dataset['three_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

labels4g = ["4G-supported",'4G Not supported']
values4g = dataset['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

sns.boxplot(x="price_range", y="battery_power", data=dataset)

plt.figure(figsize=(10,6))
dataset['fc'].hist(alpha=0.5,color='blue',label='Front camera')
dataset['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')

sns.jointplot(x='mobile_wt',y='price_range',data=dataset,kind='kde');

sns.pointplot(y="talk_time", x="price_range", data=dataset)