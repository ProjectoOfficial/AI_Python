from PIL import Image
import os
import numpy as np
import shutil
from pathlib import Path

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


parent = os.path.dirname(Path(__file__).parent)
print(parent)
outPath = os.path.join(parent, "ConvertedOutputs")

if 'ConvertedOutputs' in os.listdir(parent):
    shutil.rmtree(outPath) 
print("Creating ConvertedOutputs directory...\n")
os.makedirs(outPath)

for path, dire, fname in os.walk(parent):
    for f in fname:
      
        # creating a image1 object  
        im1 = Image.open(path+'/'+f)  
  
        # applying greyscale method  
        im2 = im1.convert('L')
  
        im2.save(outPath+'/'+f, 'png') 
