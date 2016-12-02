import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import skimage
import skimage.transform
import skimage.io

import pickle
from PIL import Image

def LoadImage(file_name, resize=448, crop=448):
    image = Image.open(file_name)
    width, height = image.size

    if width > height:
        width = (width * resize) / height
        height = resize
    else:
        height = (height * resize) / width
        width = resize

    left = (width  - crop) / 2
    top  = (height - crop) / 2
    image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
    data = np.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
    data = data.astype('float32') / 255
    return data

def plotAttention (image_file, question, alpha, smooth=True):
    
    ## Parameters
    #
    # image_file : Path to image file.
    # question   : List of question string words (tokenised)
    # alpha      : NP array of size (len(question), 196) or List of len(question) NP vectors of shape (196, )
    # smooth     : Parameter for scaling alpha
    #

    img = LoadImage(image_file)
    n_words = len(question) + 1
    w = np.round(np.sqrt(n_words))
    h = np.ceil(np.float32(n_words) / w)
            
    plt.subplot(w, h, 1)
    plt.imshow(img)
    plt.axis('off')

    for ii in xrange(alpha.shape[0]):
        plt.subplot(w, h, ii+2)
        lab = question[ii]
        plt.text(0, 1, lab, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, lab, color='black', fontsize=13)
        plt.imshow(img)
        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alpha[ii].reshape(14,14), upscale=32)
        else:
            alpha_img = skimage.transform.resize(alpha[ii].reshape(14,14), [img.shape[0], img.shape[1]])
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

answer = pickle.load(open(sys.argv[1]))
alpha, pred, confidence, question, image_id = answer

plotAttention('Images/'+ str(image_id)+'.png', question, alpha[:,0,:])
plt.savefig('Attention.png')


