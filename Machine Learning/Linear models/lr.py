#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Mohsina Zaman - CIS 572
# Program 2 - Logistic Regression
#
# 
#
import sys
import re
import math
from math import log
from math import exp
from math import sqrt
import numpy as np

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
    data.append( (x,y) )
  return (data, varnames)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0
  bias_gradient = 0.0
  sub_term = [0.0]*len(data)
  weight_gradient = [0.0] * len(w)
  #print(data[0])
  
  iterations = 0
  while iterations<=MAX_ITERS:
    loss = 0

  #calculate weight gradient
    for i in range(0,numvars):
      w_gr = 0.0
      dp = 0
      for exp in data:
        x = exp[0]
        y = exp[1]
        # save the result of terms that will be reused in array
        if i ==0:
          interm = (calculate_gr_sub(w,exp,b))
          sub_term[dp] = interm
          w_gr += (interm * x[i])
        else:
          w_gr += (sub_term[dp] * x[i])
        dp+=1
      w_gr += l2_reg_weight *w[i] # Add the regularizer
      weight_gradient[i] = w_gr #Update the weight
     

    #calculate bias gradient
    bias_sum = 0.0
    for exp in sub_term:
      bias_sum += exp
    bias_gradient = bias_sum
    
    #calculate loss
    for dp in data:
      x=dp[0]
      y=dp[1]
      #dot_product1 = np.dot(w,x)
      #e_term = ((-1)*y) * (dot_product1+b)
      #log_term = (1 + exp(e_term))
      #loss = (-1) * log(log_term)

      dot_product = np.dot(w,x)
      term = ((-1)*y)*(dot_product+b)
      log_cal = (1+ math.exp(term))
      loss +=  log(log_cal)

    #calculate gradient magnitude
    mag_sum = 0.0
    for g in weight_gradient:
      mag_sum += pow(g,2)
    mag_sum += pow(bias_gradient,2)
    magnitude = sqrt(mag_sum)
    print(iterations, end=" ")
    print("Loss: ", loss, end="\t\t")
    print("Gradient Magnitude: ", magnitude)

    #check
    if magnitude < 0.00001:
      print("Gradient low")
      break
    

    #update weights 
    for j in range(0,numvars):
      w[j]=w[j]+(eta*weight_gradient[j])
    #update bias
    b=b+(eta*bias_gradient)
    iterations+=1




  #
  # YOUR CODE HERE
  #

  return (w,b)

def calculate_gr_sub(w,data_point,bias): 
  x = data_point[0]
  y= data_point[1] 
  dot_product = np.dot(w,x)
  term = y*(dot_product+bias)
  result = (1/(1+exp(term)))*y
  return result




# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print ('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = 0.5 # <-- YOUR CODE HERE
    dot_product = np.dot(w,x)
    #term = ((-1)*y)*(dot_product+b)
    term = ((-1))*(dot_product+b)
    log_cal = (1+exp(term))
    #prob = -1 * log(log_cal)
    prob = 1/log_cal

    print (prob )
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print ("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
