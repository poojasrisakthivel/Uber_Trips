'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("UberDataset.csv")
dataset.head()

dataset.shape

dataset.info()

dataset['PURPOSE'].fillna("NOT", inplace=True)
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], 
                                       errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], 
                                     errors='coerce')

from datetime import datetime
 
dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour
 
#changing into categories of day and night
dataset['day-night'] = pd.cut(x=dataset['time'],
                              bins = [0,10,15,19,24],
                              labels = ['Morning','Afternoon','Evening','Night'])


dataset.dropna(inplace=True)

dataset.drop_duplicates(inplace=True)

obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
 
unique_values = {}
for col in object_cols:
  unique_values[col] = dataset[col].unique().size
unique_values



plt.figure(figsize=(10,5))
 
plt.subplot(1,2,1)
sns.countplot(dataset['CATEGORY'])
plt.xticks(rotation=90)
 
plt.subplot(1,2,2)
sns.countplot(dataset['PURPOSE'])
plt.xticks(rotation=90)

sns.countplot(dataset['day-night'])
plt.xticks(rotation=90)


plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()


from sklearn.preprocessing import OneHotEncoder
object_cols = ['CATEGORY', 'PURPOSE']
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)


plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(), 
            cmap='BrBG', 
            fmt='.2f', 
            linewidths=2, 
            annot=True)


dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'April',
               5.0: 'May', 6.0: 'June', 7.0: 'July', 8.0: 'Aug',
               9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'}
dataset["MONTH"] = dataset.MONTH.map(month_label)
 
mon = dataset.MONTH.value_counts(sort=False)
 
# Month total rides count vs Month ride max count
df = pd.DataFrame({"MONTHS": mon.values,
                   "VALUE COUNT": dataset.groupby('MONTH',
                                                  sort=False)['MILES'].max()})
 
p = sns.lineplot(data=df)
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")


dataset['DAY'] = dataset.START_DATE.dt.weekday
day_label = {
    0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thus', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
dataset['DAY'] = dataset['DAY'].map(day_label)

day_label = dataset.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label);
plt.xlabel('DAY')
plt.ylabel('COUNT')

sns.boxplot(dataset['MILES'])

sns.boxplot(dataset[dataset['MILES']<100]['MILES'])
sns.distplot(dataset[dataset['MILES']<40]['MILES'])

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
dataset = pd.read_csv("UberDataset.csv")

# Display first few rows of the dataset
print(dataset.head())

# Display the shape of the dataset
print(dataset.shape)

# Display dataset info
dataset.info()

# Fill missing values in the 'PURPOSE' column
dataset['PURPOSE'].fillna("NOT", inplace=True)

# Convert 'START_DATE' and 'END_DATE' to datetime
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], errors='coerce')

# Create 'date' and 'time' columns
dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour

# Create 'day-night' column
dataset['day-night'] = pd.cut(x=dataset['time'],
                              bins=[0, 10, 15, 19, 24],
                              labels=['Morning', 'Afternoon', 'Evening', 'Night'])

# Drop missing values and duplicates
dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)

# Identify object columns
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

# Display unique values in object columns
unique_values = {col: dataset[col].unique().size for col in object_cols}
print(unique_values)

# Plot count plots
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(dataset['CATEGORY'])
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
sns.countplot(dataset['PURPOSE'])
plt.xticks(rotation=90)

plt.figure(figsize=(10, 5))
sns.countplot(dataset['day-night'])
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()

# One-hot encode categorical columns
object_cols = ['CATEGORY', 'PURPOSE']
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out(object_cols)
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)

# Plot heatmap
'''plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()'''

# Add 'MONTH' column and map month names
dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'April',
               5: 'May', 6: 'June', 7: 'July', 8: 'Aug',
               9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
dataset['MONTH'] = dataset['MONTH'].map(month_label)

mon = dataset['MONTH'].value_counts(sort=False)

# Month total rides count vs Month ride max count
df = pd.DataFrame({"MONTHS": mon.values,
                   "VALUE COUNT": dataset.groupby('MONTH', sort=False)['MILES'].max()})

plt.figure(figsize=(12, 6))
p = sns.lineplot(data=df)
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")
plt.show()

# Add 'DAY' column and map day names
dataset['DAY'] = dataset['START_DATE'].dt.weekday
day_label = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thus', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
dataset['DAY'] = dataset['DAY'].map(day_label)

day_label = dataset['DAY'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=day_label.index, y=day_label)
plt.xlabel('DAY')
plt.ylabel('COUNT')
plt.show()

# Plot boxplot and distribution plot for 'MILES'
plt.figure(figsize=(12, 6))
sns.boxplot(dataset['MILES'])
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(dataset[dataset['MILES'] < 100]['MILES'])
plt.show()

plt.figure(figsize=(12, 6))
sns.distplot(dataset[dataset['MILES'] < 40]['MILES'])
plt.show()
