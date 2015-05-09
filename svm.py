import sys
import os
import codecs
import email
import math
from random import shuffle
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

def parseFolder(foldername):
	ham = 0
	spam = 1
	base_path = "datasets/"
	folders = [("easy_ham", ham), ("hard_ham", ham), ("spam", spam)]

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

def run_classifier(partition_size):
	ham = 0
	spam = 1
	base_path = "datasets/"
	folders = [("easy_ham", ham), ("hard_ham", ham), ("spam", spam)]

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

	vectorizer = CountVectorizer()
	processed_list = vectorizer.fit_transform(feature_list_shuf)

	training_x = processed_list[0:size_training]
	training_y = y_shuf[0:size_training]
	test_x = processed_list[size_training:n]
	test_y = y_shuf[size_training:n]

	#Train classifier
	clf = svm.SVC()
	clf.fit(training_x, training_y)

	#Check classifier
	prediction = clf.predict(test_x)
	correct = [i for i,j in zip(prediction, test_y) if i==j]
	correct = len(correct)
	print "Predicition Rate is: ", correct*100/len(test_y)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage python svm.py <Number between 0 and 1>"
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
