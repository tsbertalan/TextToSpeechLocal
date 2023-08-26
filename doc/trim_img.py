# Remove one pixel from all sides of the image, because Windows adds a border of the background content.
import os
HERE = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(HERE, 'local_tts.png')

from PIL import Image
img = Image.open(img_path)
img = img.crop((1, 1, img.size[0] - 1, img.size[1] - 1))
img.save(img_path)
