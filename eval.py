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

    data_directory = ''

    test = []

    # Read the concatenated file
    with open(data_directory + 'movie_data/test_neg_combined.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            text = line.strip()
            words = ['<BOS>'] + text.split() + ['<EOS>']
            test.append((0, words))

    # Read the concatenated file
    with open(data_directory + 'movie_data/test_pos_combined.txt', 'r') as f:
        lines = f.readlines()
        # Process each line
        for line in lines:
            text = line.strip()
            words = ['<BOS>'] + text.split() + ['<EOS>']
            test.append((1, words))

    # load model from models folder
    model = model = torch.load('models/' + model_name, map_location=torch.device('cpu'))


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
        test_outputs.append((label, classification))
        if (classification > 0.5 and label == 1) or (classification <= 0.5 and label == 0):
            correct += 1

    test_accuracy = correct / len(test)

    print(f'test_accuracy={test_accuracy}', file=sys.stderr, flush=True)

