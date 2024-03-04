# nn_project


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
