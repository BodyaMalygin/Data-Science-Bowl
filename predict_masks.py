import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.io import imshow
from datasets import get_dataset
from model import dice_coef

X, y = get_dataset('stage1_train/')

model = load_model('model-dsbowl2018-1.h5', custom_objects={'dice_coef': dice_coef})
preds_val = model.predict(X[int(X.shape[0]*0.9):], verbose=1)

img_num = np.random.choice(list(range(63)), 10)
fig, ax = plt.subplots(10, 3, figsize=(16, 32))
for i, num in enumerate(img_num):
    ax[i, 0].grid(False)
    ax[i, 0].set_title('origin')
    ax[i, 0].imshow(X[603+num])
    ax[i, 1].grid(False)
    ax[i, 1].set_title('mask')
    ax[i, 1].imshow(np.squeeze(y[603+num]))
    ax[i, 2].grid(False)
    ax[i, 2].set_title('predict')
    ax[i, 2].imshow(np.squeeze(preds_val[num]))
plt.tight_layout()
plt.show()