from keras.utils.visualize_util import plot
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# model architecture viz
model = load_model('model.h5')
plot(model, to_file='model.png', show_layer_names=False, show_shapes=True)

# training viz
csv_path = 'train_stats.csv'
df = pd.DataFrame.from_csv(csv_path)
df.plot()

# x = df.index
# plt.plot(x, df['loss'])
# plt.plot(x, df['val_loss'])
plt.savefig('train_stats.png')
