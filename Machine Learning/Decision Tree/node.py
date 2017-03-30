#!/usr/bin/python
# 
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/20/2017
# You are not obligated to use any of this code, but are free to use
# anything you find helpful when completing your assignment.
#
# You are free to add methods to this Node class, but you may not need to.
#
import sys

class Node:
  """ Node class for a decision tree. """
  def __init__(self, names):
    self.names = names

  def classify(x):
    """ Handled by the subclasses. """
    return None

  def write(self, f, indent):
    """ Handled by the subclasses. """
    return None


class Leaf(Node):
  def __init__(self, names, value):
    Node.__init__(self, names)
    self.value = value

  def classify(self, x):
    return self.value

  def write(self, f, indent):
    f.write(' %d\n' % self.value)


class Split(Node):
  def __init__(self, names, var, left, right):
    Node.__init__(self, names)
    self.var = var
    self.left = left
    self.right = right
 
  def classify(self, x):
    
    index=self.names.index(self.var)
    if x[index] == 0:
      return self.left.classify(x)
    else:
      return self.right.classify(x)
      
  def write(self, f, indent):
    if indent > 0:
      f.write('\n')
    for i in range(0, indent):
      f.write('| ')
    
    index=self.names.index(self.var)
  
    f.write('%s = 0 :' % self.names[index])
    self.left.write(f, indent+1)
    for i in range(0, indent):
      f.write('| ')
    f.write('%s = 1 :' % self.names[index])
    self.right.write(f, indent+1)

# Test code
if __name__ == "__main__":
  n = ['foo', 'bar', 'baz']
  root = Split(n, 0, Split(n, 1, Leaf(n, 0), Leaf(n, 1)), Leaf(n, 0)) 
  root.write(sys.stdout, 0)

  print (root.classify([0,0,0]))
  print (root.classify([0,0,1]))
  print (root.classify([0,1,0]))
  print (root.classify([0,1,1]))
  print (root.classify([1,0,0]))
  print (root.classify([1,0,1]))
  print (root.classify([1,1,0]))
  print (root.classify([1,1,1]))
