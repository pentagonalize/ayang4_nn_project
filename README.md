# nn_project

# Part 1
This is a project I am working on with David Chiang. We want to make a programmatic way to set the weights of transformer encoders for the purposes of studying the kinds of problems they can and cannot solve. In this setting, we are looking at the formal languages that a transformer encoder can and cannot classify with 100% accuracy. For example, can it classify strings of parentheses that are matched “(())()()” from those that are unmatched “((((“ with 100% accuracy? 

Instead of training them, we want to see what is possible in theory. If we come up with a method to classify strings, by hand, can we manually set a transformer’s weights to classify the same strings? So we have a toy “programming language” to write simple programs to classify strings, and then we want to set the weights of a transformer in order for it to function the same as the toy program. 

The programming language has 6 operations 

	                  a(i) = the string has the character ‘a’ at position i
	P1(i) and P2(i) = P1==true and P2==true at position i
 	           Not P(i) = P==false at position i
       COUNT [j < i] V(i) = count the number of positions less than i where V==true
               C1(i) + C2(i) = add C1 and C2 at position i
   C1(i) < C2(i) = check if C1<C2 at position i

And we claim that we can programmatically set the weights of a transformer such that the computations through its layers can simulate these operations in whatever order we put them in. For instance, we should be able to set the weights of a transformer to check whether “the string starts with ‘a’” , “the number of ‘a’s is greater than the number of ‘b’s”, “the string contains no ‘b’s”, and things like that. If this succeeds we can better understand the limits of expressivity of transformers. 

This will involve understanding more deeply how to work with neural networks and their parameters. I will have to understand what the parameters do, in order to set them programmatically. 

# Part 2 

We have been looking at explainable AI in class -  in particular, during practical 3 we looked at visualizing kernels, plotting feature spaces, and inspecting saliency maps for images. These methods adopt the approach of taking previously trained networks and trying to find explanations for the classification procedures they implement. What if we went in the other direction? What if we initialized a network with weights that we know implement particular procedures, so in this way they are explainable by design? 

David and I have been working on some preliminary steps towards this, but specifically for transformers. We have a draft about how to compile some very simple algorithms into transformer weights (for instance checking substrings, counting character occurrences, etc) using formal logic. It remains to actually implement this and test this! I’ve attached the draft in this repository as a PDF. 

If that is possible, then it would be interesting to have some data to compare the procedures we hard-code compared to what a transformer learns. In particular, we can show there exists a perfect solution to some problems, but what can a transformer learn in practice? 

In this setting, we consider transformers as classifiers - the simplest setting is binary classification of strings in formal languages. These would be synthetic datasets, but people use them because they’re well structured, so we know what the network needs to learn in order to succeed at classification, and it’s easy to test for generalization - just increase the length of the string beyond what was seen in the training set. Some sets of strings we can use for classification tasks are:

a*b*  (any number of a’s followed by any number of b’s)
a*b*a* (strings that don’t contain the substring bab)
anbncn  (strings in the order a, b, then c, with same number of a’s, b’s, c’s)
Dyck-1 (Matched and balanced parentheses (())()()
Contains subsequence “hello” (For example ahebllco) 

We can essentially generate as much data as we want, in this setting. A well-cited previous work https://aclanthology.org/2020.emnlp-main.576.pdf did 10,000 samples for training, and tested on ~6,000 other samples, of varying lengths exceeding the lengths encountered during training. 

Beyond this, it might be interesting to look at less synthetic datasets. For example, sentiment analysis using transformer encoders. Maybe we can see if hard-coding heuristics (like presence of certain keywords) and then training the model might make a difference using our method? I don’t know
