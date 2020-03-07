import os

for path,dire,filename in os.walk('/home/daniel/Scrivania/Gan/TrainingImages'):
    for filename in filename:
        os.replace(filename,filename.replace('.mp3_Piano_0',''))
