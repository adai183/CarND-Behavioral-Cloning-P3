import pandas as pd

csv_path = 'Data/driving_log.csv'
measurements = pd.DataFrame.from_csv(csv_path)
measurements['center'] = measurements.index
measurements.index = range(measurements.shape[0])
# drop unnecessary columns
measurements = measurements.drop(['throttle', 'brake', 'speed'], 1)

# img_prev = np.array(cv2.imread(measurements.loc[9080]['center']))
# plt.imshow(img_prev)
# plt.show()


# # add data for avoiding dirt road multiple times in order to positively
# # bias agent :-)
# for i in range(100):
#     measurements = measurements.append(
#         measurements.iloc[8037:8197], ignore_index=True)


# Make paths local
measurements['center'] = measurements[
    'center'].apply(lambda x: x.split('/')[-1])

measurements['left'] = measurements[
    'left'].apply(lambda x: x.split('/')[-1])

measurements['right'] = measurements[
    'right'].apply(lambda x: x.split('/')[-1])


# Create Series for each camera angle and drop unnecessary columns
measurements_center = measurements.copy().set_index(
    'center').drop(['left', 'right'], 1)
measurements_right = measurements.copy().set_index(
    'right').drop(['left', 'center'], 1)
measurements_left = measurements.copy().set_index(
    'left').drop(['right', 'center'], 1)

# add a correction angle to center the car
measurements_right['steering'] -= 0.08
measurements_left['steering'] += 0.08

measurements = pd.concat(
    [measurements_center, measurements_right, measurements_left])


# decrease most frequent steering angles
i = 0
measurements_length = measurements.shape[0]
for index, row in measurements.iterrows():
    counts = measurements['steering'].value_counts()
    angle = row['steering']
    oc = counts.loc[angle]

    if angle >= -0.08 and angle <= 0.08:
        if oc > 2000:
            measurements = measurements.drop(row.name)
    i += 1
    print ('Processing Data: {}/{}'.format(i, measurements_length))


# measurements = measurements[measurements['steering']!=0.0]
# measurements = measurements[measurements['steering']!=0.08]
# measurements = measurements[measurements['steering']!=-0.08]


# Data split: Train/Test/Validation
train_size = int(0.8 * measurements.shape[0])
valid_size = int(0.2 * measurements.shape[0])
# test_size = int(0.2 * measurements.shape[0])
# shuffle Data before creating batches
measurements = measurements.sample(frac=1)

measurements_train = measurements[:train_size]
measurements_valid = measurements[train_size:train_size + valid_size]
# measurements_test = measurements[train_size + valid_size:]

# clear cache
del measurements_right, measurements_center, measurements_left, measurements

measurements_train.to_csv('Data/train_data.csv')
measurements_valid.to_csv('Data/valid_data.csv')
# measurements_test.to_csv('Data/test_data.csv')
