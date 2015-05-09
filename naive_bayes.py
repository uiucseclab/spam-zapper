import sys
import os
import codecs
import email
import math
import nltk
from random import shuffle

ham = 0
spam = 1
base_path = "datasets/"
folders = [("easy_ham", ham), ("hard_ham", ham), ("spam", spam)]

def parseFolder(foldername):
	null_count = 0
	path = base_path + foldername + "/"
	feature_list = []
	for filename in os.listdir(path):
		if filename == ".DS_Store":
			continue
		f = open(path+filename, 'r')
		content = email.message_from_file(f)
		sender = content['Sender']
		payload = ""
		if not content.is_multipart():
			payload = content.get_payload()
			if payload != None:
				try:
					features = str(payload)
					feature_list.append(unicode((features), errors='replace'))
				except ValueError:
					null_count += 1
	
	return feature_list

def create_bayesian_classifier(dataset, labels):
	classifier = {}
	n = len(labels)
	for i in range(n):
		current_set = dataset[i]
		words = nltk.word_tokenize(current_set) # Get actual words out from the payload
		for key in words:
			if classifier.get(key) != None:
				classifier[key][2] += 1.0
				if labels == spam:
					classifier[key][spam] += 1.0
				else:
					classifier[key][ham] += 1.0
			else:
				classifier[key] = [0.0, 0.0, 1.0]
				if labels == spam:
					classifier[key][spam] += 1.0
				else:
					classifier[key][ham] += 1.0
	return classifier

def run_classifier(partition_size):
	# Parse different datasets
	y = []
	feature_list = []
	for foldername, value in folders:
		result = parseFolder(foldername)
		feature_list.extend(result)
		y.extend([value] * len(result))

	# Shuffle 
	n = len(y)
	feature_list_shuf = []
	y_shuf = []
	index_shuf = range(n)
	shuffle(index_shuf)
	for i in index_shuf:
	    feature_list_shuf.append(feature_list[i])
	    y_shuf.append(y[i])

	# Partition
	size_training = int(math.floor(partition_size*n))
	training_x = feature_list_shuf[0:size_training]
	training_y = y_shuf[0:size_training]
	test_x = feature_list_shuf[size_training:n]
	test_y = y_shuf[size_training:n]

	classifier = create_bayesian_classifier(training_x, training_y)
	threshold = 0.5

	# Prediction
	correct = 0
	test_set_length = len(test_y)
	for i in range(test_set_length):
		spam_probability = float(0.0)
		ham_probability = float(0.0)
		current_set = test_x[i]
		words = nltk.word_tokenize(current_set)
		for key in words:
			val = classifier.get(key)
			if val != None:
				spam_probability += float(val[spam] / val[2])
				ham_probability += float(val[ham] / val[2])

		if (spam_probability >= ham_probability) and test_y[i] == spam:
			correct += 1
		if (spam_probability < ham_probability) and test_y[i] == ham:
			correct += 1

	print "Prediction Rate is: ", (float(correct) / float(test_set_length)) * 100.0

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage python naive_bayes.py <Number between 0 and 1>"
		sys.exit(0)
	else:
		try:
			partition_size = float(sys.argv[1])
			if partition_size > 0 and partition_size < 1:
				run_classifier(partition_size)
			else:
				print "Invalid partition size range"
				sys.exit(0)
		except ValueError:
			print "Enter a float or int between 0 and 1"
			sys.exit(0)

