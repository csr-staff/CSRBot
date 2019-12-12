import nltk
nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


def chat():
	print("Start talking with the bot (type quit to stop)!")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		#results will give us a probability
		results = model.predict([bag_of_words(inp, words)])

		#How probable our model thinks each neuron is. Each neuron represents a specific tag
		#print(results)

		#Get index of greatest value (most probable) in our list
		results_index = numpy.argmax(results)
		tag = tags[results_index]
		#print(tag)

		for tag_responses in data["intents"]:
			if tag_responses['tag'] == tag:
				responses = tag_responses['responses']

		print(random.choice(responses))


#Turn input from user into a bag of words
def bag_of_words(sentence, words):
	#Initialize bag of words to be an array of zeros (of size words)
	bag = [0 for _ in range(len(words))]

	#Tokenize and stem words
	s_words = nltk.word_tokenize(sentence)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for s in s_words:
		for i, w in enumerate(words):
			#If word in words list is the word in our sentence
			if w == s:
				bag[i] = 1
	return numpy.array(bag)


#Main.py
with open("intents.json") as file:
	data = json.load(file)

try:
	jklasdfl
	with open("data.pickle", "rb") as f:
		words, tags, training, output = pickle.load(f)

except:
	#List of all patterns (potential projected inputs). Duplicates to be removed.
	words = []
	#List of all "Tags"
	tags = []
	#List of all patterns (potential projected inputs)
	docs_pattern = []
	#Corresponding tag for a given pattern
	docs_tag = []

	#Load in the JSON file and store all the data
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			#Stemming: bring it to the root word. E.g. "what's up" --> "what". Get the essense of work
			#Tokenize (split by space)
			pattern_words = nltk.word_tokenize(pattern)
			words.extend(pattern_words)
			docs_pattern.append(pattern_words)
			docs_tag.append(intent["tag"])

		if intent["tag"] not in tags:
			tags.append(intent["tag"])

	#Stem and sort the words
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))
	tags = sorted(tags)

	#Create a "One hot" representation of our text where each index 
	#in the array represents a word and the number in that index is 
	#either 1 (word exists) or 0 (word doesn't exist)

	#Will have a bunch of "one hot encoded" bags of words (0s and 1s)
	training = []

	#Will be a list of 0s and 1s (with 1s for each "tag" word represented)
	output = []

	#Initialize an array of zeros for each tag 
	out_empty = [0 for _ in range(len(tags))]

	#For each pattern (potential input). X will be the index of the pattern (0 first iteration, then 1, then 2, etc.)
	for x, doc in enumerate(docs_pattern):
		bag = []

		pattern_words = []
		for w in doc:
			pattern_words.append(stemmer.stem(w))

		for w in words:
			if w in pattern_words:
				bag.append(1)
			else:
				bag.append(0)

		#Make a copy (blank list)
		output_row = out_empty[:]
		#Looks through the tags, see where the tag is and set that to 1 in our output_row
		output_row[tags.index(docs_tag[x])] = 1

		training.append(bag)
		output.append(output_row)

	#turn them into np arrays (formatting). All just getting data ready to feed into our model
	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f:
		pickle.dump((words, tags, training, output), f)


#BUILD OUR MODEL
tensorflow.reset_default_graph()

#Define input shape. Every training input is same length so can just use the first
net = tflearn.input_data(shape=[None,len(training[0])])
#Add two hidden neural layers with 8 neurons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#Add output layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	jklasdfl
	model.load("model.tflearn")
except:
	#n_epochs is how many times we show the model our data (play around w. this)
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

	#Want to save the model so don't need to do all the preprocessing
	model.save("model.tflearn")


chat()