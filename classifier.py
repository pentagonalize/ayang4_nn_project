import torch
import math, random, copy, sys, os, csv
from layers import *
from utils import *
import torch
import math, random, copy, sys, os, csv
# from layers import *
# from utils import *

# https://ai.stanford.edu/~amaas/data/sentiment/

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")

max_len =  128
layers = 1

class ClassifierEncoder(torch.nn.Module):
    """
        IBM Model 2 encoder, modified for classfication.
        A CLS token is added to the input sentence.
        Then classification is based on the value of the CLS position.
    """

    def __init__(self, vocab_size, dims):
        super().__init__()
        self.emb = Embedding(vocab_size, dims) # This is called V in the notes
        self.pos = Embedding(max_len, dims)    # This is called K in the notes
        # Initialize a list of 2 self-attention layers using the SelfAttentionLayer class
        self.attention_layers = torch.nn.ModuleList([MaskedSelfAttentionLayer(dims).to(device) for i in range(layers)])
        # Initialize a list of 4 linear layers using the LinearLayer class
        self.linear_layers = torch.nn.ModuleList([LinearLayer(dims, dims).to(device) for i in range(2*layers)])
        # final linear layer to get the output dimension to be 1
        self.out = LinearLayer(dims, 1).to(device)

    def forward(self, words):
        """ Classify a sentence
        Argument: Input sentence (list of n ints)
        Returns: results we only care about CLS position (Tensor of size n)"""

        # Get word embeddings and position embeddings
        femb = self.emb(words)

        # Sum the word embeddings and position embeddings into a single vector
        H = femb

        # Loop through the self-attention layers and feed-forward layers
        for i in range(layers):
            # Apply the self-attention layer
            H1 = self.attention_layers[i].forward(H).to(device)
            # Apply the feed-forward layer
            H2 = self.linear_layers[2*i](H1).to(device)
            # Apply ReLU
            H3 = torch.relu(H2).to(device)
            # Apply one more feed-forward layer
            H4 = self.linear_layers[2*i+1](H3).to(device)
            # Apply residual connection
            H = torch.add(H4, H1)
            # Loop
            del H1
            del H2
            del H3
            del H4

        # Apply the final linear layer
        H5 = self.out(H).to(device)
        del H
        # Apply sigmoid to get the probability
        res = torch.sigmoid(H5)
        del H5

        return res

class Model(torch.nn.Module):
    """IBM Model 2.

    You are free to modify this class, but you probably don't need to;
    it's probably enough to modify Encoder and Decoder.
    """
    def __init__(self, vocab, d):
        super().__init__()

        # Store the vocabularies inside the Model object
        # so that they get loaded and saved with it.
        self.vocab = vocab
        self.encoder = ClassifierEncoder(len(vocab), 2*d)

    def classify_train(self, words):
        """
            Compute a forward pass for classification

            Arguments:
                - words: list of words

            Output:
                - prob: float from 0 to 1
        """

        # truncate to max_len
        words = words[:max_len-1]

        if not words[-1] == "<EOS>":
            words[-1] = "<EOS>"

        # convert words to numbers
        words = torch.tensor([self.vocab.numberize(word) for word in words]).to(device)

        # Compute the forward pass

        H = self.encoder.forward(words)
        output = H[-1]
        del H
        del words
        torch.cuda.empty_cache()

        return output

def process(sentence):
    # remove all non-alphabetic characters
    sentence = ''.join(e for e in sentence if e.isalnum() or e.isspace())
    # convert to lowercase
    sentence = sentence.lower()
    # split the sentence into words
    words = sentence.split()
    # add <BOS> and <EOS> tokens
    words = ['<BOS>'] + words + ['<EOS>']
    return words

if __name__ == "__main__":
    # Specify the file path
    data_directory = 'movie_data/'

    data = []
    test = []

    # Read the concatenated file train neg
    with open(data_directory + 'train_neg_combined.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            data.append((0, process(line)))

    # Read the concatenated file train pos
    with open(data_directory + 'train_pos_combined.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            data.append((1, process(line)))


    # Read the concatenated file
    with open(data_directory + 'test_neg_combined.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            test.append((0, process(line)))

    # Read the concatenated file
    with open(data_directory + 'test_pos_combined.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            test.append((1, process(line)))



    # shuffle the data
    random.shuffle(data)
    random.shuffle(test)
    test = test
    data = data
    print(len(data))

    train = data[:int(len(data)*0.8)]
    dev = data[int(len(data)*0.8):]


    vocab = Vocab()

    for data in data:
        vocab |= data[1]

    model = Model(vocab, 32).to(device)
    lr = 0.000008
    epochs = 10

    opt = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=1)

    best_dev_loss = float('inf')
    for epoch in range(epochs):
        random.shuffle(train)

        ## Update model on train
        criterion = torch.nn.BCELoss()
        train_loss = 0.
        correct = 0
        for data in progress(train):
            label, words = data

            # Forward pass
            probabilities = model.classify_train(words)

            # loss
            label_tensor = torch.tensor(label).float().to(device)
            loss = criterion(probabilities.squeeze(), label_tensor)

            # Zero gradients, backward pass, and optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()
            del label_tensor

            train_loss += loss.item()

            # Calculate train accuracy
            if (probabilities > 0.5 and label == 1) or (probabilities <= 0.5 and label == 0):
                correct += 1

        train_accuracy = correct / len(train)

        ## Evaluate model on dev

        dev_loss = 0.
        dev_outputs = []
        correct = 0
        for data in progress(dev):
            label, words = data

            # Forward pass
            probabilities = model.classify_train(words)

            # loss
            label_tensor = torch.tensor(label).float().to(device)
            loss = criterion(probabilities.squeeze(), label_tensor)
            dev_loss += loss.item()

            # Calculate dev accuracy
            if (probabilities > 0.5 and label == 1) or (probabilities <= 0.5 and label == 0):
                correct += 1

        dev_accuracy = correct / len(dev)
        print(f'[{epoch+1}] train_loss={train_loss} train_accuracy={train_accuracy} dev_loss={dev_loss} dev_accuracy={dev_accuracy}', file=sys.stderr, flush=True)

        if dev_loss < best_dev_loss:
            best_model = copy.deepcopy(model)
            best_dev_loss = dev_loss

    # evaluate accuracy on test
    torch.cuda.empty_cache()
    correct = 0
    test_accuracy = 0
    # shuffle the test data
    random.shuffle(test)
    for data in progress(test):
        label, words = data
        classification = model.classify_train(words)
        if (classification > 0.5 and label == 1) or (classification <= 0.5 and label == 0):
            correct += 1
        torch.cuda.empty_cache()

    test_accuracy = correct / len(test)

    print(f'test_accuracy={test_accuracy}', file=sys.stderr, flush=True)

    # put all paremeters in a string
    params = f'layers={layers}_data={len(data)}_lr={lr}_epochs={epochs}_train_accuracy={train_accuracy}_test_accuracy={test_accuracy}'

    torch.save(best_model, 'models/model_'+params+'.pt')
