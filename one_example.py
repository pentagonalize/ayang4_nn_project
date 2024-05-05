import torch
import math, random, copy, sys, os, csv
from layers import *
from utils import *
from classifier import ClassifierEncoder, Model



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")
    # Specify the file path
    data_directory = ''
    test = []

    # load model from models folder
    model = torch.load('models/movie_model.pt',map_location=torch.device('cpu'))

    # example
    postest = "Yes, this gets the full ten stars. It's plain as day that this fill is genius. The universe sent Trent Harris a young, wonderfully strange man one day and Harris caught him on tape, in all that true misfit glory that you just can't fake. Too bad it ended in tragedy for the young man, if only an alternate ending could be written for that fellow's story. The other two steps in the trilogy do retell the story, with Sean Penn and Crispin Glover in the roles of the young men, respectively. The world is expanded upon and the strangeness is contextualized by the retelling, giving us a broader glimpse into growing up weird in vanilla America. Recommended for anyone and everyone!"
    print("positive sample: ", postest)

    # clean
    postest = postest.strip()
    postest = ['<BOS>'] + postest.split() + ['<EOS>']
    # evaluate accuracy on test
    classification = model.classify_train(postest)
    print("classification: ", classification)

    # example
    negtest = 'You might suspect that the plot of this movie was written in the process of filming. It begins as a "punks versus vigilante" movie, but in the middle of the film, the plot changes abruptly when the vigilante turns to be an honest man with his honest girl and his honest gym and has to fight the corrupt "businessmen" who want to turn the gym down at any cost to build a mall or something. Then, the plot changes again, and we forget about the corrupt guys. The villain now is the friend of the leading man, who thinks he is a Ninja. The guy becomes "crazy evil" and wants at any cost to win a Martial Arts Contest. Seeing this movie is like having a nightmare with the television on.'
    print("positive sample: ", negtest)

    # clean
    negtest = negtest.strip()
    negtest = ['<BOS>'] + negtest.split() + ['<EOS>']
    # evaluate accuracy on test
    classification = model.classify_train(negtest)
    print("classification: ", classification)

