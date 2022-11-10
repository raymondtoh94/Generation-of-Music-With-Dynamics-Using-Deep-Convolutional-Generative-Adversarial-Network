import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import muspy

def back_to_format(song):
    update_song = np.zeros((128,48))
    update_song[45:93] = song
    return update_song

def decode_img(imgs, velocity=True):
    if velocity == True:
        imgs = (63.5 * imgs) + 63.5
        imgs = np.squeeze(imgs)
        imgs[gen_imgs < 40] = 0
        imgs = (imgs).astype(int)
    else:
        imgs = (0.5 * imgs) + 0.5
        imgs = np.squeeze(imgs)
        imgs = (imgs+0.5).astype(int)
    return imgs

generator = load_model('generator_model_200000.h5')

dynamics=False

for num in range(101, 201, 1):
    print(f"Generating GAN_{num}")
    noise = np.random.normal(0,1,(1, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = decode_img(imgs=gen_imgs, velocity=dynamics)
    gen_imgs = back_to_format(gen_imgs)
    gen_imgs = cv2.resize(gen_imgs, (192,128), interpolation=cv2.INTER_AREA)
    
    plt.figure()
    plt.imshow(gen_imgs, cmap="gray", aspect='auto')
    
    for j in range(12,192,12):
        plt.axvline(x=j, color='gray', linestyle='--')
        
    for i in range(48,192,48):
        plt.axvline(x=i, color='gray', linewidth=2)
    
    plt.axis('off')
    plt.savefig(f"images/GAN_{num}.png")
    plt.close()

#song = muspy.from_pianoroll_representation(gen_imgs.transpose(1,0), resolution=12, encode_velocity=velocity) #Before update resolution = 3
#muspy.write("generator_model.midi", song, kind="midi")

