from keras.models import model_from_json
from imageio import imread
import numpy as np
import glob, os
import pandas as pd

# load json and create model
json_file = open('model_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("new_custom_network_hope_epoch_65.h5")
print("Loaded model from disk")

model.summary()


BASE_TEST_DIR = os.path.abspath("../C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data")

test_imgs = glob.glob(BASE_TEST_DIR + "\*")


output = []
img_names = []
for img_dir in test_imgs:
    img_names.append(int(os.path.split(img_dir)[-1].split('.')[0]))
    img = imread(img_dir)
    img = np.expand_dims(img, axis = 0)
    result = model.predict(img, batch_size = 1)
    output.append(result.argmax(axis = 1)[0])
#    output.append(result)
    
#C-NMC_test_prelim_phase_data_labels
#ground_truth = 


df = pd.DataFrame(columns=['new_names', 'labels'])
df['new_names'] = img_names
df['labels'] = output
df.sort_values(by = ['new_names'], inplace = True)
df['new_names'] = df['new_names'].astype(str)+'.bmp'
#df.reset_index(inplace = True) 
df.to_csv("test_result_new.csv", index=False)