import os

datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "0")
for path, dire, filenames in os.walk(datapath):
    for filename in filenames:
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "0", filename)
        print(file.replace('.mp3',''))
        os.rename(file, file.replace('.mp3', ''))
