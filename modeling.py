from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

import polars as pl
import numpy as np
path = '2021-2023/{}.parquet'

# concat 2021-2023 into single dataframe
df = pl.concat([
    pl.scan_parquet(path.format(2021)),
    pl.scan_parquet(path.format(2022)),
    pl.scan_parquet(path.format(2023))
])

# drop unnecessary columns and create duration and date/time columns
df = df.with_columns([
    (pl.col('ended_at') - pl.col('started_at')).dt.seconds().alias('duration'),
    pl.col('started_at').dt.strftime('%m-%d').alias('startmonthday'),
    pl.col('started_at').dt.strftime('%H:%M:%S').alias('starttime'),
    pl.col('ended_at').dt.strftime('%m-%d').alias('endmonthday'),
    pl.col('ended_at').dt.strftime('%H:%M:%S').alias('endtime'),
]).drop(['ride_id', 'start_station_id', 'end_station_id']).drop_nulls()

# map values to integers
df_rideabletype = df.filter(pl.col('rideable_type').is_in(['docked_bike', 'electric_bike', 'classic_bike']))

map_dict = {
    'electric_bike': 0,
    'docked_bike': 1,
    'classic_bike': 2,
}

df_rideabletype = df_rideabletype.with_columns(
    pl.col('rideable_type').map_dict(map_dict).alias('rideable_type').cast(pl.Int32)
).collect()

# split into X and y
X = df_rideabletype[['duration', 'startmonthday', 'starttime', 'endmonthday', 'endtime', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual', 'start_station_name', 'end_station_name', 'started_at', 'ended_at']]
y = df_rideabletype['rideable_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# split into train and test
rf = RandomForestClassifier()
rf.fit(X_train.to_numpy(), y_train.to_numpy())

# predict
y_pred = rf.predict(X_test.to_numpy())

# evaluate
accuracy = accuracy_score(y_test.to_numpy(), y_pred)
print("Accuracy:", accuracy)