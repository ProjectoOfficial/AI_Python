import numpy as np
import cv2


# variabili che contengono i percorsi dei file:
# - prototxt è il file che contiene l'architettura della rete neurale
# - caffemodel è il modello già allenato
# - points contiene i punti dell'inviluppo convesso
# - img contiene l'immagine
prototxt = r"C:\Users\daniel\PycharmProjects\YT\models\colorization_deploy_v2.prototxt"
caffemodel = r"C:\Users\daniel\PycharmProjects\YT\models\colorization_release_v2.caffemodel"
points = r"C:\Users\daniel\PycharmProjects\YT\models\pts_in_hull.npy"
img = r"C:\Users\daniel\PycharmProjects\YT\images\volto2.jpg"

# preleviamo l'immagine
frame = cv2.imread(img)

# definiamo la dimensione di input della rete neurale
W_in = 224
H_in = 224

# leggiamo la rete dai file
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# carichiamo i punti dell'inviluppo convesso
pts_in_hull = np.load(points)

# popoliamo il cluster di punti con kernel 1x1 che corrispondono
# ad ognuno dei 313 punti a lo assegnamo al layer corrispondente
# della rete neurale
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

#converto i colori rgb dell'immagine in input in un range di 0 e 1
image_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
img_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
img_l = img_lab[:, :, 0] #prelevo il canale luminosità

# ridimensiono l'immagine nel canale della luminosità
img_l_rs = cv2.resize(img_l, (W_in, H_in))
img_l_rs -= 50 #sottraggo 50  per centrarlo in 0

# forniamo in input alla rete neurale il canale della luminosità
blob = cv2.dnn.blobFromImage(img_l_rs)
net.setInput(blob)
# l'output del forward è il canale di colori che la rete neurale
# ha predetto
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0)) # questo è il risultato

# andiamo a resettare le dimensioni originali
(H_orig, W_orig) = image_rgb.shape[:2] # dimensioni originali dell'immagine
ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
# effettuiamo il merge con il canale luminosità originale
img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
#convertiamo in rgb l'immagine ottenuta
bgr_out = np.clip(cv2.cvtColor(img_lab_out,cv2.COLOR_Lab2BGR), 0, 1)

cv2.imshow("originale", image_rgb)
cv2.imshow("colorata", bgr_out)
cv2.waitKey(0)