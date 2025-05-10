import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_hub as hub

def load_image(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    img = img.convert('RGB')
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    img = np.array(img)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.
    return tf.convert_to_tensor(img)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

content_image = load_image('content.jpg')
style_image = load_image('style.jpg')

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

output_image = tensor_to_image(stylized_image)
output_image.save('output.png')

plt.imshow(output_image)
plt.axis('off')
plt.show()
