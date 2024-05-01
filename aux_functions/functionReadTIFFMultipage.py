import cv2
import numpy as np
from PIL import Image

def read_multipage_tiff(dirImage, bitdepth):
    img = Image.open(dirImage)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))

    del img

    height, width = np.shape(images[0])
    numImgs = len(images)
    #print(height, width, numImgs)

    if bitdepth == 8:
        volume = np.uint8(np.zeros((height, width, numImgs)))
    else:
        volume = np.uint16(np.zeros((height, width, numImgs)))

    for i in range(numImgs):
        sliceSingle = images[i]
        volume[:, :, i] = sliceSingle
    
    return volume
    
def read_multipage_tiff_as_list(file_path):
    img = Image.open(file_path)
    images = []
    while True:
        try:
            img.seek(len(images))  # Go to the next frame
            images.append(img.copy())
        except EOFError:
            break
    return images

def split_list_images(images, n_channels):
    list_list_images = []
    n_images = len(images)
    n_images_per_channel = int(n_images / n_channels)
    
    for i_channel in range(n_channels):
        images_channel = []
        #print('i_channel: ' + str(i_channel))
        for i_img_in_channel in range(n_images_per_channel):
            #print('i_img_in_channel: ' + str(i_img_in_channel))
            idx = i_img_in_channel * n_channels + i_channel
            #print(idx)
            img = images[idx]
            images_channel.append(img)
        list_list_images.append(images_channel)
        
    return list_list_images

def get_projected_image(images):
    # Convert images to numpy arrays
    images_array = np.array([np.array(img, dtype=np.float32) for img in images])

    # Compute the average of the images
    avg_image = np.mean(images_array, axis=0)

    # Convert back to uint8
    avg_image = np.uint16(avg_image)

    return avg_image
