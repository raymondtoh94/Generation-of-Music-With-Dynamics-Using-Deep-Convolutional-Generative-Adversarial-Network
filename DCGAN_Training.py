import os
import muspy
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input, Flatten, Reshape, Conv2DTranspose, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential, Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SONGS_PATH = "./songs/"

#48x48 image
img_shape = (48,48,1)
gen_shape = (6,6,256)

def retrieve_midi_file(filepath):
    if not os.path.exists(filepath):
        raise ValueError("Midi File Paths cannot be found!")
    
    midi_file = []
    for midi in os.listdir(filepath):
        if midi.lower().endswith(("midi","mid")):
            midi_file.append(filepath+midi)
    return midi_file

def muspy_read_midi_file(midi_file, resolution=24):
    songs = []
    for song in midi_file:
        file = muspy.read(song)
        file.clip(lower=64, upper=127)
        songs.append(muspy.adjust_resolution(file, resolution))
    return songs

def save_imgs(epoch, velocity=True):
    image_file = "./images"
    if not os.path.exists(image_file):
        os.mkdir(image_file)

    r, c = 2,2
    noise = np.random.normal(0,1,(r*c, 100))
    gen_imgs = generator.predict(noise)

    if velocity == True:
        gen_imgs = (63.5 * gen_imgs) + 63.5
        gen_imgs = np.squeeze(gen_imgs)
        gen_imgs = gen_imgs.astype(int)
    else:
        gen_imgs = (0.5 * gen_imgs) + 0.5
        gen_imgs = np.squeeze(gen_imgs)
        gen_imgs = (gen_imgs+0.5).astype(int)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    plt.savefig("images/GAN_%d.png" % epoch)
    plt.close()


def transpose_to_Amin(song):
    key_interval = {"B": -2,
                    "A#": -1,
                    "G#": 1,
                    "G": 2,
                    "F#": 3,
                    "F": 4,
                    "E": 5,
                    "D#": 6,
                    "D": 7,
                    "C#": 8,
                    "C": 9}
    
    key = song.metadata.source_filename.split('-')[2].split(' ')[1]
    
    if key != "A":
        print(f"Current Key is: {key}min, transposing by {key_interval[key]}")
        song.transpose(key_interval[key])
    return song
    
def build_generator():
    #Input - Noise
    #Output - Fake Image with label True to fake discriminator
    
    noise_shape = (100,)
    
    model = Sequential()
    model.add(Dense(np.prod(gen_shape), input_shape=noise_shape))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))
    model.add(Reshape(gen_shape))

    model.add(Conv2DTranspose(128, 3, (2,2), padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(64, 3, (2,2), padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(1, 3, (2,2), padding="same", activation="tanh"))
    
    noise = Input(shape=noise_shape)
    img = model(noise)
    model.summary()
    
    return Model(noise,img)

def build_discriminator():
    #Input - Image
    #Output - Prediction of Image True/False
    
    model = Sequential()
    model.add(Conv2D(64, 3, (2,2), padding="same", input_shape=img_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, 3, (2,2), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(256, 3, (2,2), padding="same"))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    
    model.add(Flatten())

    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))
    
    img = Input(shape=img_shape)
    validity = model(img)
    model.summary()
    
    return Model(img, validity)

def train(sequence, epochs, batch_size=64, save_interval=50, velocity=True):
    model_file = "./model"
    loss_img = "./loss_img"

    if not os.path.exists(model_file):
        os.mkdir(model_file)
    if not os.path.exists(loss_img):
        os.mkdir(loss_img)

    #Normalize sequence to -1 to 1
    if velocity == True:
        sequence = (sequence-63.5)/63.5
    else:
        sequence = (sequence-0.5)/0.5
    sequence = np.expand_dims(sequence, -1)

    dloss = []
    gloss = []

    for epoch in range(epochs):
        start = time.time()
        idx = np.random.randint(0,sequence.shape[0],batch_size)
        imgs = sequence[idx]
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        
        gen_imgs = generator.predict(noise)
    
        if epoch % 5 == 0:
            d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
        end = time.time()
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        print(f"Time taken = {end-start}")
        dloss.append(d_loss[0])
        gloss.append(g_loss)

        if epoch % save_interval == 0:
            save_imgs(epoch, velocity=velocity)

            plt.plot(dloss, label='d_loss')
            plt.plot(gloss, label='g_loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend(loc='best')
            plt.savefig(f"{loss_img}/loss_{epoch}.png")
            plt.close()

            generator.save(f'{model_file}/generator_model_{epoch}.h5')


def get_lowest_note(song):
    for num, row in enumerate(song):
        if any(row != 0):
            return num
        
def back_to_format(song):
    update_song = np.zeros((128,48))
    update_song[40:88] = song
    return update_song


if __name__ == '__main__':
    #Load midi files
    midi_file = retrieve_midi_file(SONGS_PATH)
    velocity=True
    
    #Read midi in muspy format & clip velocity min=64, max=127
    songs = muspy_read_midi_file(midi_file, resolution=12) #Before update resolution = 1
    
    sequences = []
    
    for song in songs:
        #Print current preprocessing song
        print(f"Preprocessing {song.metadata.source_filename}")
        
        #Transpose song to cmaj or amin key
        song = transpose_to_Amin(song)
        
        #Convert to muspy pianoroll representation (time,pitches)
        song = song.to_pianoroll_representation(encode_velocity=velocity)
        
        #truncate song if song > 8sec (Remove if update fail)
        if song.shape[0]>192:
            song = song[:192]
        
        #Convert shape into Pitch(X-axis)=128, Time(Y-axis)=48
        song = cv2.resize(song, (128,48), interpolation=cv2.INTER_AREA)
        
        #Transpose the shape -> Pitch(Y-axis)=128, Time(X-axis)=48
        song = song.transpose(1,0)
        
        #Extract a total of 4 octave range
        low = get_lowest_note(song)
        song = song[low:low+48]
        
        #Add each sequence tensor into list
        sequences.append(song)
        
    #Combine sequences into (M sample, 48, 48) dimension
    sequence = np.stack(sequences)
    print(f"Sequences Shape = {sequence.shape}")  
    
    gen_optimizer = Adam(5e-6, 0.5)
    disc_optimizer = Adam(5e-6, 0.5)
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy",
        optimizer=disc_optimizer,
        metrics=['accuracy'])
    
    generator = build_generator()
    generator.compile(loss="binary_crossentropy", optimizer=gen_optimizer)
    
    z = Input(shape=(100,))
    img = generator(z)
    discriminator.trainable = False  
    valid = discriminator(img)
    
    combined = Model(z, valid)
    combined.compile(loss="binary_crossentropy", optimizer=gen_optimizer)
    
    train(sequence, epochs=500000, batch_size=32, save_interval=1000, velocity=velocity)
    
