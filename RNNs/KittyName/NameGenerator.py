from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

path = '/home/daniel/PycharmProjects/gpuproject/KittyName/names2.txt'
with open(path, 'r') as f:
    text = f.readlines()
    text = " ".join(text).lower()
print("lunghezza del testo: {}\n".format(len(text)))

#set elimina i doppioni, ritrasfortmo in lista e ordino la lista in ordine alfabetico
chars = sorted(list(set(text)))
print("caratteri totali: {}\n".format(len(chars)))

#creo due dizionari, uno con il carattere come chiave, l'altro con l'indice come chiave
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#per creare features e labels dal testo andiamo a creare sentenze
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    print(i)
    sentences.append(text[i: i + maxlen])#features sono parti di testo di lunghezza maxlenght, considerando uno step di 3
    next_chars.append(text[i + maxlen])#labels sono il carattere successivo
print("sequenza: {}\n".format(len(sentences)))

#effettuiamo una vettorizzazione e un encoding dei dati delle features e labels
print("vettorizzazione...\n")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)#tensore che utilizziamo nel layer LSTM
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        #lunghezza delle features - lunghezza di ogni singola feature - indice del carattere (se il carattere è tra quelli che vogliamo usare, poniamo a 1)
        x[i, t, char_indices[char]] = 1
    #stessa cosa per le labels
    y[i, char_indices[next_chars[i]]] = 1

print(x)
print(y)

# build the model: a single LSTM
print("Creazione modello: \n")
#Un Sequential modello è appropriato per una semplice pila di livelli in cui ogni livello ha esattamente un tensore di input e un tensore di output .
model = Sequential()
#definiamo batch size e input shape
model.add(LSTM(128, input_shape=(maxlen, len(chars))))

#Lo strato denso è il normale livello di rete neurale profondamente connesso.
#Il layer denso esegue l'operazione seguente sull'input e restituisce l'output.
#output = activation(dot(input, kernel) + bias)
#input rappresenta i dati di input
#il kernel rappresenta i dati relativi al peso
#punto rappresenta il prodotto punto intorpidito di tutti gli input e i relativi pesi
#il bias rappresenta un valore distorto utilizzato nell'apprendimento automatico per ottimizzare il modello
#L'attivazione rappresenta la funzione di attivazione.

model.add(Dense(len(chars), activation='softmax'))

#L'essenza di RMSprop è:
#Mantenere una media mobile del quadrato dei gradienti
#Dividi il gradiente per la radice di questa media
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, diversity=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print("Generazione di testo, Epoca: {}\n".format(epoch))

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("diversity: {}\n".format(diversity))

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print("----- Generato con Seed: "+sentence+"\n")
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

#interrompe il training quando l'accuratezza è sufficiente
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=100,
          callbacks=[print_callback])