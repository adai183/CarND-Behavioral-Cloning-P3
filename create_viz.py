from keras.utils.visualize_util import plot
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# model architecture viz
model = load_model('model.h5')
plot(model, to_file='model.png', show_layer_names=False, show_shapes=True)

# # training viz
# csv_path = 'stats.csv'
# df = pd.DataFrame.from_csv(csv_path)
# df['i'] = pd.Series(range(df.shape[0]), index=df.index)
# df = df.set_index('i', drop=True)
# df = df.drop(['Batch'], 1)
# # df.plot(use_index=True)
# # plt.show()

# x = df.index
# plt.plot(x, df['Loss'])
# plt.plot(x, df['Validation Loss'])
# plt.savefig('img/train_stats.png')
