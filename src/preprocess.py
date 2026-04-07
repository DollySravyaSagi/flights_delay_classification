import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/flights.csv",nrows=400000)

#analysis
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:",df.info())
print("\nMissing values:\n", df.isnull().sum())
print(df.head())

#preprocessing
df = df.drop(columns=['CANCELLATION_REASON'])
df = df[df['CANCELLED'] == 0]
df = df.dropna(subset=['ARRIVAL_DELAY', 'DEPARTURE_DELAY'])
df = df[['MONTH','DAY','DAY_OF_WEEK','AIRLINE','DISTANCE','DEPARTURE_DELAY','ARRIVAL_DELAY','SCHEDULED_DEPARTURE']]

#EDA
df['AIRLINE'].value_counts().plot(kind='bar')
plt.title("Flights by Airline")
plt.savefig("output/flights_by_airline.png")
plt.show()

sns.histplot(df['ARRIVAL_DELAY'], bins=50)
plt.savefig("output/arrival_delay.png")
plt.show()

#feature engineering
df = df[['MONTH','DAY_OF_WEEK','AIRLINE','DISTANCE', 'DEPARTURE_DELAY','ARRIVAL_DELAY','SCHEDULED_DEPARTURE']]
df['ARRIVAL_DELAY'] = df['ARRIVAL_DELAY'].clip(-100, 300)
df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].clip(-100, 300)
df['WEEKEND'] = (df['DAY_OF_WEEK'] >= 6).astype(int)
df['DIST_BIN'] = pd.cut(df['DISTANCE'], bins=5, labels=False)
df['IS_LATE_DEP'] = (df['DEPARTURE_DELAY'] > 0).astype(int)
df['DEP_HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
df['DEP_DELAY_BIN'] = pd.cut(df['DEPARTURE_DELAY'], bins=5, labels=False)
df['DEP_DELAY_RATIO'] = df['DEPARTURE_DELAY'] / (df['DISTANCE'] + 1)
df = df[['MONTH','DAY_OF_WEEK','DISTANCE','DEPARTURE_DELAY','WEEKEND','DIST_BIN','IS_LATE_DEP','DEP_HOUR','DEP_DELAY_BIN','DEP_DELAY_RATIO','AIRLINE','ARRIVAL_DELAY']]
df = pd.get_dummies(df,columns=['AIRLINE'], drop_first=True)

df.to_csv("data/processed.csv", index=False)