import torch
import math, random, copy, sys, os, csv
from layers import *
from utils import *
from classifier import ClassifierEncoder, Model
import argparse

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='evaluate model')

    # Add an argument
    parser.add_argument('abc_model', type=str, help='the abc model to evaluate')

    # Parse the arguments
    args = parser.parse_args()

    # Now use args.data_directory instead of data_directory
    model_name = args.abc_model

    test = []

    data_directory = ''

    # Read the concatenated file
    with open('formal_language_data/abc_noncontiguous/test_negs.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            text = line.strip()
            words = ['<BOS>'] + text.split() + ['<EOS>']
            test.append((0, words))

    # Read the concatenated file
    with open('formal_language_data/abc_noncontiguous/test_pos.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            text = line.strip()
            words = ['<BOS>'] + text.split() + ['<EOS>']
            test.append((1, words))

    # load model from models folder
    model = torch.load('abc_models/' + model_name + '.pt')

    # evaluate accuracy on test

    correct = 0
    test_outputs = []
    test_accuracy = 0
    # shuffle the test data
    random.shuffle(test)
    test = test[0:100]
    for data in progress(test):
        label, words = data
        classification = model.classify_train(words)
        if (classification > 0.5 and label == 1) or (classification <= 0.5 and label == 0):
            correct += 1

    test_accuracy = correct / len(test)

    print(f'test_accuracy={test_accuracy}', file=sys.stderr, flush=True)

