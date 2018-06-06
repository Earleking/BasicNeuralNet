from net2 import net
import csv
import random
from mnist import MNIST
import numpy as np
    

# Everything below this is for setting up and using the net
# The first number is the nodes for the input layer
# The last number is the nodes for the output layer
# The other numbers are the nodes for the hidder layers
a = net([784,16,16,10])
#Create the input and output lists
#These lists are 2-d arrays. Each list within the list represets one trial
inputs = []
outputs = []

#load the training data
mndata = MNIST('mnist_data')
images, labels = mndata.load_training()
#Just set variables that will determine how testing will be run
desired_accuracy = 0.15
max_iterations = 500
train_size = len(images)
# Test size is the size of the data that will be tested when determining error rate
# Smaller test sizes allow for faster learning but may give more skewed results
test_size = 150
iterations = 0    

inputs = np.divide(images, 255)

for i in range(train_size):
    outputs.append([0] * 10)
    outputs[i][labels[i]] = 1

a.test(inputs, outputs)

error = a.error(inputs[:test_size], outputs[:test_size])
while error > desired_accuracy:

    iterations += 1
    if(iterations > max_iterations):
        break
  
    a.train(inputs, outputs)
    # print(iterations)
    error = a.error(inputs[:test_size], outputs[:test_size])
    print(error)
#Run it again vs the tests
a.test(inputs, outputs)

images, labels = mndata.load_testing()
inputs = np.divide(images, 255)
outputs = []
for i in range(len(inputs)):
    outputs.append([0] * 10)
    outputs[i][labels[i]] = 1
a.test(inputs, outputs)


