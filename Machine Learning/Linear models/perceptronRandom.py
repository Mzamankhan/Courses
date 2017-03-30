#!/usr/bin/python
#
# CIS 472/572 - Perceptron Template Code
# Mohsina Zaman - CIS 572
# Perceptron with Randomized ordering at each iteration (Extra credit)
# 
#  

import sys
import re
from math import log
from math import exp
import random

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    # Each example is a tuple containing both x (vector) and y (int)
    data.append( (x,y) )
  return (data, varnames)


# Learn weights using the perceptron algorithm
def train_perceptron(data):
    # Initialize weight vector and bias
    numvars = len(data[0][0]) 
    #print(numvars)
    weights = [0.0] * numvars
    total_examples = len(data)
    b = 0.0
    #print(len(w))

    #
    # YOUR CODE HERE!
    #
    i = 0
    while i<=MAX_ITERS: 
      correct=0
      for exp in data:
        #print(exp)
        #print(exp[0])
        #print(len(exp[0]))
        #calculate total activation
        activation = 0
        for j in range(0,numvars):
          activation=activation+(weights[j]*exp[0][j])
        activation=activation+b
        #check if wrong
        if (exp[1]*activation)<=0.0:
          #change all weights
          for w in range(0,numvars):
            new=weights[w]+(exp[1]*exp[0][w])
            weights[w]=new
          b=b+exp[1]
          #print(b)
        else:
          correct=correct+1
      #i=i+1
      if correct == total_examples:
        print("All correctly classified")
        break
      else:
        print(i, end=" ")
        print(len(data)-correct)
      i=i+1
      #randomize order for next iteration
      random.shuffle(data)

    return (weights,b)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  # Process command line arguments.
  # (You shouldn't need to change this.)
  if (len(argv) != 3):
    print ('Usage: perceptron.py <train> <test> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  modelfile = argv[2]

  # Train model
  (w,b) = train_perceptron(train)

  # Write model file
  # (You shouldn't need to change this.)
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    #print(x)
    activation = 0.0 # <-- YOUR CODE HERE
    for j in range(len(x)-1):
      activation=activation+(w[j]*x[j])
    activation=activation+b
    #print (activation)
    if activation * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print ("Accuracy: ", end=" ")
  print(acc)

if __name__ == "__main__":
  main(sys.argv[1:])
