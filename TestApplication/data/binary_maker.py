# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:41:08 2020

@author: Corbi
"""

fileo = open('mnist_binary.csv', 'w') 
filei = open('mnist_test.csv', 'r') 
Lines = filei.readlines()
filei.close()
  

for line in Lines: 
    if (line[0] == "0" or line[0] == "1"):
        fileo.write(line)
    
fileo.close() 