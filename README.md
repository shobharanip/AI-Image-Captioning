# AI-Image-Captioning
Image Captioning

Load Libraries
1. Prepare Photo Data
VGG16 Model
About VGG:
•	Visual Geometry Group from University of Oxford developed VGG model
•	VGG model won the ImageNet competition in 2014
•	Published as a conference paper at ICLR 2015: https://arxiv.org/pdf/1409.1556.pdf
•	Visual Geometry Group overview: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
About VGG16 Model:
•	3×3 filters in all convolutional layers
•	16 Layers Model
•	Layer Configurations: https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
Applications
•	Given an image, find object name in the image.
•	It can detect any one of 1000 images.
•	It takes input image of size 224 224 3 (RGB image) i.e 224 * 224 pixel image with 3 channels

2. Preparing Text Data
Function to return a dictionary of photo identifiers to the descriptions

Function to clean the descriptions in the following ways:
•	Convert all words to lowercase.
•	Remove all punctuation.
•	Remove all words that are one character or less in length (e.g. ‘a’).
•	Remove all words with numbers in them.

Ideally, we want a vocabulary that is both expressive and as small as possible. A smaller vocabulary will result in a smaller model that will train faster.
For reference, we can transform the clean descriptions into a set and print its size to get an idea of the size of our dataset vocabulary.
Sets are highly optimized, don't contain any duplicate values. There implementation is based on hash table. Hence we get a vocabulary that is both expressive and small.
3. Developing Deep Learning Model
A.) Loading the data
The description text will need to be encoded to numbers before it can be presented to the model as in input or compared to the model’s predictions.
The first step in encoding the data is to create a consistent mapping from words to unique integer values. Keras provides the Tokenizer class that can learn this mapping from the loaded description data.
Below defines the to_lines() to convert the dictionary of descriptions into a list of strings and the create_tokenizer() function that will fit a Tokenizer given the loaded photo description text.
We can now encode the text.
Each description will be split into words. The model will be provided one word and the photo and generate the next word. Then the first two words of the description will be provided to the model as input with the image to generate the next word. This is how the model will be trained.
B.) Defining the Model
The model is in three parts:
1.	Photo Feature Extractor: This is a 16-layer VGG model pre-trained on the ImageNet dataset. We have pre-processed the photos with the VGG model (without the output layer) and will use the extracted features predicted by this model as input.
2.	Sequence Processor: This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.
3.	Decoder: Both the feature extractor and sequence processor output a fixed-length vector. These are merged together and processed by a Dense layer to make a final prediction. The Photo Feature Extractor model expects input photo features to be a vector of 4,096 elements. These are processed by a Dense layer to produce a 256 element representation of the photo.
The Sequence Processor model expects input sequences with a pre-defined length (34 words) which are fed into an Embedding layer that uses a mask to ignore padded values. This is followed by an LSTM layer with 256 memory units.
Both the input models produce a 256 element vector. Further, both input models use regularization in the form of 50% dropout. This is to reduce overfitting the training dataset, as this model configuration learns very fast.
The Decoder model merges the vectors from both input models using an addition operation. This is then fed to a Dense 256 neuron layer and then to a final output Dense layer that makes a softmax prediction over the entire output vocabulary for the next word in the sequence.
4. Evaluate the model
5. Generate new descriptions

