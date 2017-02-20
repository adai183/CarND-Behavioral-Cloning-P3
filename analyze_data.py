import pandas as pd
import matplotlib.pyplot as plt


csv_path = 'Data/train_data.csv'
df = pd.DataFrame.from_csv(csv_path)
df.hist(column='steering',  bins=1000)
plt.show()
counts = df['steering'].value_counts()
print (df.shape[0])
print (counts)
print(df['steering'].value_counts().idxmax())
print(df['steering'].value_counts().max())
print (counts.loc[0.0])
