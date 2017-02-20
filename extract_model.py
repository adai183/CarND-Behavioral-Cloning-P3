from keras.models import load_model
from keras.models import model_from_json
import json




# with open('model.json', 'r') as jfile:
# 	model = model_from_json(jfile.read())

# model.load_weights('model.h5')

# model.save('model.h5')


model = load_model('model.h5')

with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())

model.save_weights("model_weights.h5")