# DATASET https://download.pytorch.org/tutorial/data.zip
# THIS EXAMPLE CODE IS INSPIRED BY https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random
import time
import math

import torch
from torch import nn
import matplotlib.pyplot as plt

PATH = r"C:\Users\daniel\PycharmProjects\Python2021\RNNs\NAMES_GENERATOR"

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker


def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


def timeSince(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.1)
        self.flatten = nn.Flatten()
        self.mlp_1 = nn.Linear(hidden_dim, 64)
        self.mlp_2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        x, (hn, cn) = self.lstm(x)

        x = torch.transpose(hn, 0, 1)
        x = self.flatten(x)
        x = self.relu(self.mlp_1(x))
        x = self.softmax(self.mlp_2(x))
        return x


# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)

        output_name = start_letter

        for i in range(max_length):
            output = rnn((input[0].unsqueeze(0) + category_tensor.unsqueeze(-1)).to(device))
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


if __name__ == "__main__":
    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')

    print('# categories:', n_categories, all_categories)
    print(unicodeToAscii("O'Néàl"))

    learning_rate = 0.0001

    hidden_size = 128

    assert torch.cuda.is_available(), "Notebook is not configured properly!"
    device = 'cuda:0'
    rnn = RNN(n_letters, hidden_size, n_letters).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    if not os.path.exists(PATH + r"\RNN.pth"):
        n_iters = 1000000
        print_every = 5000
        plot_every = 500
        all_losses = []
        total_loss = 0  # Reset every plot_every iters

        start = time.time()

        for iter in range(1, n_iters + 1):
            category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
            hidden = torch.zeros(1, hidden_size)

            category_tensor = category_tensor.expand(input_line_tensor.shape[0], category_tensor.shape[1])

            # training
            rnn.zero_grad()
            out = rnn((category_tensor.unsqueeze(-1) + input_line_tensor).to(device))
            loss = criterion(out, target_line_tensor.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss

            if iter % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

            if iter % plot_every == 0:
                all_losses.append(total_loss.to(torch.device("cpu")).detach().numpy() / plot_every)
                total_loss = 0

        plt.figure()
        plt.plot(all_losses)
        plt.show()

        torch.save(rnn.state_dict(), PATH + r"\RNN.pth")
    else:
        rnn.load_state_dict(torch.load(PATH + r"\RNN.pth"))

    max_length = 20

    samples('Russian', 'RUS')
    print("")
    samples('German', 'GER')
    print("")
    samples('Spanish', 'SPA')
    print("")
    samples('Chinese', 'CHI')
    print("")
    samples('Italian', 'ITA')
