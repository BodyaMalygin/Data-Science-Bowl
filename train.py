from model import get_model
from datasets import get_dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint

X, y = get_dataset('stage1_train/')

model = get_model()
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X, y, validation_split=0.1, batch_size=16, epochs=15, 
                    callbacks=[earlystopper, checkpointer])