# Aspect-Based-Sentiment-Analysis
1. Team members: Kewei WANG, Jingyi LI

2. Model description:
For the data preprocessing part, we used spacy package to extract words that contributes to the sentiment part most, which are adjectives and verbs. 
We observed that excluding stop words, punctuations and numbers can help reduce the noise when training the classifier.
Then we lemmatize the key word extracted from comments and use the keras package for tokenization.
For the model part, we used neural network layers in keras. It contains word embedding layer that transfer words into densed vectors, LSTM layer and Dense layer.
To control the overfit, we added dropout layers in the model as well.
As the target result contains 3 classes, we used categorical_crossentropy as the loss function.

3. The accuracy that we got on the dev dataset is 0.79
