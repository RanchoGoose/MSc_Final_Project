from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from tensorflow.keras.models import Sequential, load_model
# import keras
# from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
# from keras.models import Sequential, load_model
# from keras_layer_normalization import LayerNormalization
import matplotlib.pyplot as plt


DATASET_PATH = "D:/car_accident_detection/dataset/manual/sherbrooke_frames"
SINGLE_TEST_PATH = "D:/car_accident_detection/dataset/manual/extracted_frames/000002"
BATCH_SIZE = 4
EPOCHS = 10
MODEL_PATH = "C:/Users/REXzh/PycharmProjects/MSc_Final_Project/graymodelep10.hdf5"

def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 256 X 256
    sequence_size: int
        The size of the lstm sequence
    Returns
    -------
    list
        A list of clips , 10 frames each
    """
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, :] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(clip)
                cnt = 0
    return clips

def get_training_set():
    """
    Returns
    -------
    list
        A list of training sequences of shape (NUMBER_OF_SEQUENCES,SINGLE_SEQUENCE_SIZE,FRAME_WIDTH,FRAME_HEIGHT,1)
    """
    clips = []
    # loop over the training folders (Train000,Train001,..)
    for f in sorted(listdir(DATASET_PATH)):
        directory_path = join(DATASET_PATH, f)
        if isdir(directory_path):
            all_frames = []
            # loop over all the images in the folder (0.tif,1.tif,..,199.tif)
            for c in sorted(listdir(directory_path)):
                img_path = join(directory_path, c)
                if str(img_path)[-3:] == "jpg":
                    img = tf.io.read_file(img_path)
                    img = tf.image.decode_image(img, channels=1)
                    img = (tf.cast(img, tf.float32))
                    img = (img / 256.0)
                    img = tf.image.resize(img, (256, 256))

                    # img = Image.open(img_path).resize((256, 256))
                    # img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    return clips

def get_model(reload_model=False):
# def get_model():
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """


    if not reload_model:
        return load_model(MODEL_PATH, custom_objects={'LayerNormalization': tf.keras.layers.LayerNormalization})
    training_set = get_training_set()
    training_set = np.array(training_set)
    seq = Sequential()
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(tf.keras.layers.LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(tf.keras.layers.LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    print(seq.summary())
    seq.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
    seq.fit(training_set, training_set,
            batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False)
    seq.save(MODEL_PATH)
    return seq

def get_single_test():
    framenum = 0
    for f in sorted(listdir(SINGLE_TEST_PATH)):
        if str(join(SINGLE_TEST_PATH, f))[-3:] == "jpg":
            framenum = framenum + 1
    sz = framenum
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    # all_frames = []
    # test = []
    for f in sorted(listdir(SINGLE_TEST_PATH)):
        if str(join(SINGLE_TEST_PATH, f))[-3:] == "jpg":
            # img = Image.open(join(SINGLE_TEST_PATH, f)).resize((256, 256))
            # img = np.array(img, dtype=np.float32) / 256.0

            img = tf.io.read_file(join(SINGLE_TEST_PATH, f))
            img = tf.image.decode_image(img, channels=1)
            img = (tf.cast(img, tf.float32))
            img = (img / 256.0)
            img = tf.image.resize(img, (256, 256))

            # all_frames.append(img)
            # sz = len(all_frames)
            # test = np.zeros(shape=(sz, 256, 256, 3))
            test[cnt, :, :, :] = img
            cnt = cnt + 1
    return test

model = get_model(False)
print("got model")
test = get_single_test()
print("got test")
sz = test.shape[0] - 10
sequences = np.zeros((sz, 10, 256, 256, 1))
# apply the sliding window technique to get the sequences
for i in range(0, sz):
    clip = np.zeros((10, 256, 256, 1))
    for j in range(0, 10):
        clip[j] = test[i + j, :, :, :]
    sequences[i] = clip

# get the reconstruction cost of all the sequences
reconstructed_sequences = model.predict(sequences,batch_size=4)
sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0, sz)])
sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
sr = 1.0 - sa

# plot the regularity scores
plt.plot(sr)
plt.ylabel('regularity score Sr(t)')
plt.xlabel('frame t')
plt.savefig('C:/Users/REXzh/PycharmProjects/MSc_Final_Project/results/grayep10for02.png')
plt.show()

