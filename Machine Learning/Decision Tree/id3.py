#!/usr/bin/python
# 
# CIS 472/572 -- Programming Homework #1
#
#
import sys
import re
# Node class for the decision tree
import node
import math
import csv

# Helper functions:
# - compute entropy of a 2-valued (Bernoulli) probability distribution  
# - compute information gain for a particular attribute
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable 


# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  visited = [False]*(len(varnames)-1)
  namehash = {}
  for l in f:
    data.append([int(x) for x in p.split(l.strip())])
  return (data, varnames, visited)

# Saves the model to a file.  Most of the work here is done in the 
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
  f = open(modelfile, 'w+')
  root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames, visited):
    guess=get_most_label(data) #most common label in data
    #base cases
    #base case 1 and 2 : all examples 0
    if check_pure_example(data,0):
    	#print (data)
    	#print("all zero")
    	return node.Leaf(varnames, 0)
    elif check_pure_example(data,1):
    	#print("all one")
    	return node.Leaf(varnames, 1)
    #base case 2: no more attributes
    #elif check_all_visited(visited):
    #	return node.Leaf(varnames, guess)
    else:
	    #get best atrribute    
	    (att_index,threshold)=calculate_entropy(data,varnames,visited)
	    #check if threshold then stop
	    if threshold:
	    	#print("threshold")
	    	return node.Leaf(varnames, guess)
	    att_val = varnames[att_index]
	    #set the attribute as used
	    visited[att_index]=True
	    #divide the data for att_val os 0 and 1
	    no = get_data_subset(data, 0, att_index)
	    yes = get_data_subset(data, 1, att_index)
	    left = build_tree(no,varnames,visited)
	    right = build_tree(yes,varnames,visited)
	    return node.Split(varnames,att_val,left,right)
	    #return node.Leaf(varnames, 1)

def check_pure_example(data,val):
	count=0
	length=len(data)
	label=len(data[0])-1
	for row in range(len(data)):
		if data[row][label] == val:
			count=count+1

	if count == length :
		return True
	else:
		return False

def check_all_visited(visited):
	count=0
	length=len(visited)

	for val in visited:
		if val == True:
			count=count+1

	if count == length :
		return True
	else:
		return False

def get_most_label(data):
	zero_count=one_count=0
	label=len(data[0])-1
	#print (label)
	for row in range(len(data)):
		if data[row][label] == 0:
			zero_count=zero_count+1
		else:
			one_count=one_count+1
	if zero_count>one_count:
		return 0
	else:
		return 1


def get_data_subset(data, val, att_index):	
	new_data=[]
	for row in range(len(data)):
		if data[row][att_index] == val:
			new_data.append(data[row])
	return new_data

def calculate_entropy(data,varnames, visited):
  #Entropy0[i] and Entropy1[i] are the entropies for val 0 and 1 for the ith varname
  entropy0 = [None]*(len(varnames)-1)
  entropy1 = [None]*(len(varnames)-1)
  total0_exp = [None]*(len(varnames)-1)
  total1_exp = [None]*(len(varnames)-1)
  label=len(varnames)-1
  zero_pos = zero_neg = one_pos=one_neg=0 #keep count
  
  #iterate over varnames to calculate entropy
  for i in range(len(varnames)-1):
    #print(varnames[i])
    #iterate over data rows and get number of class 1's and 0's for val 0 and 1
    zero_pos = zero_neg = one_pos=one_neg=0 #keep count
    for row in range(len(data)):
      if data[row][i] == 0:
        if data[row][label] == 1: #positive example for val 0
          zero_pos=zero_pos+1
        else: #negative example for val 0
          zero_neg=zero_neg+1
      else: # val is 1
        if data[row][label] == 1: #positive example for val 1
          one_pos=one_pos+1
        else: #negative example for val 1
          one_neg=one_neg+1
    #Calculate entropy
    #total examples for 0 and 1

    total0= zero_pos+zero_neg
    total1= one_pos+one_neg
    total0_exp[i]=total0
    total1_exp[i]=total1

    if total0 ==0 : 
    	div0pos=0
    	div0neg=0
    else: 
    	div0pos = zero_pos/total0
    	div0neg = zero_neg/total0
    if total1 ==0 :
    	div1pos =0
    	div1neg =0
    else:
    	div1pos = one_pos/total1
    	div1neg = one_neg/total1
    
    #check for 0's
    if div0pos == 0: int0_1 =0
    else: int0_1 = ((-1)*(div0pos*math.log(div0pos,2)))
    if div0neg ==0 : int0_0 =0
    else : int0_0 = ((-1)*(div0neg*math.log(div0neg,2)))
    if div1pos==0: int1_0=0
    else : int1_0 = ((-1)*(div1pos*math.log(div1pos,2)))
    if div1neg == 0 : int1_1 =0
    else: int1_1 = ((-1)*(div1neg*math.log(div1neg,2)))

    #calculate entropy
    en0 = int0_1+int0_0
    en1 = int1_0+int1_1
    entropy0[i]=en0
    entropy1[i]=en1
    #print(entropy0[i])
    #print(entropy1[i])
    #print(total0_exp[i])
    #print(total1_exp[i])
    #print("total is: ")
    #print(total1_exp[i]+total0_exp[i])
  (att_index, threshold)=calculate_informationgain(data,entropy0,entropy1,total0_exp,total1_exp,visited)
  return (att_index, threshold)



def calculate_informationgain(data,entropy0,entropy1,total0_exp,total1_exp,visited):
  in_gain=[-1]*len(entropy0)
  #Get total number of examples, and number of pos and neg examples
  exp_total=len(data)
  zeros=ones=0
  label=len(data[0])-1
  for row in range(len(data)):
  	if data[row][label] == 0:
  		zeros=zeros+1
  	else:
  		ones=ones+1

  #Get entropy of data
  pos_ratio=ones/exp_total
  neg_ratio=zeros/exp_total
  try:
    data_entropy=((-1)*(pos_ratio*math.log(pos_ratio,2)))+((-1)*(neg_ratio*math.log(neg_ratio,2)))
  except ValueError as E:
    print("problem")
    print(E)
  #calculate information gain
  for i in range(len(entropy0)):

  	information_gain=data_entropy-(((total0_exp[i]/exp_total)*entropy0[i])+((total1_exp[i]/exp_total)*entropy1[i]))
  	in_gain[i]=information_gain
	
  	'''
  	if visited[i] == True:
  		in_gain[i] = -1
  	else:
	  	information_gain=data_entropy-(((total0_exp[i]/exp_total)*entropy0[i])+((total1_exp[i]/exp_total)*entropy1[i]))
	  	in_gain[i]=information_gain
	 '''

  #find maximum information gain
  max_value=max(in_gain)
  max_index=in_gain.index(max_value)
  #print(in_gain)
  threshold = check_gain(in_gain)
  #print("max")
  #print(max_value)
  #print(max_index)
  return (max_index, threshold)
# Load train and test data.  Learn model.  Report accuracy.

def check_gain(inf_gain):
	count=0
	length=len(inf_gain)
	for val in inf_gain:
		if val == 0:
			count=count+1
		elif val <0:
			count=count+1

	if count == length :
		return True
	else:
		return False

def main(argv):
  if (len(argv) != 3):
    print ('Usage: id3.py <train> <test> <model>')
    sys.exit(2)
  # "varnames" is a list of names, one for each variable
  # "train" and "test" are lists of examples.  
  # Each example is a list of attribute values, where the last element in
  # the list is the class value.
  (train, varnames,visited) = read_data(argv[0])
  (test, testvarnames, visited) = read_data(argv[1])
  modelfile = argv[2]

  # build_tree is the main function you'll have to implement, along with
  # any helper functions needed.  It should return the root node of the
  # decision tree.
  root = build_tree(train, varnames, visited)

  print_model(root, modelfile)
  correct = 0
  # The position of the class label is the last element in the list.
  yi = len(test[0]) - 1
  for x in test:
    # Classification is done recursively by the node class.
    # This should work as-is.
    pred = root.classify(x)
    if pred == x[yi]:
      correct += 1
  acc = float(correct)/len(test)
  print ("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
