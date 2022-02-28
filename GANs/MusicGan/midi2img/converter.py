from PIL import Image
import os
from numpy import array
import shutil

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


pt = '/home/daniel/Scrivania/Gan/output'
outPath = '/home/daniel/Scrivania/Gan/ConvertedOutputs'

if 'ConvertedOutputs' in os.listdir('/home/daniel/Scrivania/Gan/'):
    shutil.rmtree(outPath) 
print("Creating ConvertedOutputs directory...\n")
os.makedirs(outPath)

for path, dire, fname in os.walk(pt):
    for f in fname:
      
        # creating a image1 object  
        im1 = Image.open(path+'/'+f)  
  
        # applying greyscale method  
        im2 = im1.convert('L')
  
        im2.save(outPath+'/'+f, 'png') 
