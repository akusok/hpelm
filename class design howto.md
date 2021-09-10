# How to design a class.

from [Stackoverflow thread](https://stackoverflow.com/questions/4203163/how-do-i-design-a-class-in-python)

1. Write down the words. You started to do this. Some people don't and wonder why they have problems.
2. Expand your set of words into simple statements about what these objects will be doing. That is to say, write down the various calculations you'll be doing on these things. Your short list of 30 dogs, 24 measurements, 4 contacts, and several "parameters" per contact is interesting, but only part of the story. Your "locations of each paw" and "compare all the paws of the same dog to determine which contact belongs to which paw" are the next step in object design.
3. Underline the nouns. Seriously. Some folks debate the value of this, but I find that for first-time OO developers it helps. Underline the nouns.
4. Review the nouns. Generic nouns like "parameter" and "measurement" need to be replaced with specific, concrete nouns that apply to your problem in your problem domain. Specifics help clarify the problem. Generics simply elide details.
5. For each noun ("contact", "paw", "dog", etc.) write down the attributes of that noun and the actions in which that object engages. Don't short-cut this. Every attribute. "Data Set contains 30 Dogs" for example is important.
6. For each attribute, identify if this is a relationship to a defined noun, or some other kind of "primitive" or "atomic" data like a string or a float or something irreducible.
7. For each action or operation, you have to identify which noun has the responsibility, and which nouns merely participate. It's a question of "mutability". Some objects get updated, others don't. Mutable objects must own total responsibility for their mutations. 
8. At this point, you can start to transform nouns into class definitions. Some collective nouns are lists, dictionaries, tuples, sets or namedtuples, and you don't need to do very much work. Other classes are more complex, either because of complex derived data or because of some update/mutation which is performed.

Don't forget to test each class in isolation using unittest.
Also, there's no law that says classes must be mutable. In your case, for example, you have almost no mutable data. What you have is derived data, created by transformation functions from the source dataset.


####################################################################################

Extreme Learning Machine (ELM) is a machine learning model universally suitable for classification and regression 
problems. It includes one or several types of hidden neurons concatenated together into the hidden neuron layer.
Each neuron type has its own connection to input layer (dense, sparse or pairwise function based), and an element-wise
transformation function applied on hidden layer output that is usually non-linear and bound. ELM model also includes
a linear solver for the output weights, with several options and multiple parameters available: batch solvers,
L2 and L1 regularization, iterative addition and removal ("forgetting") or training data samples, Lanczos finite
iterative solvers, GPU-accelerated solvers, and distributed solvers.

ELM toolbox supports export of trained models into Scikit-Learn compatible format for inference, and training of
new models with limited solver options (and reduced performance for very large tasks).



