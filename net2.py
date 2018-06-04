import numpy as np
import random
import math
import csv
import xlrd

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def reverseSigmoid(x):
    return math.log(1/x - 1)

class net:
    def __init__(self, layers):
        #Create activation matrix
        # This will be a 2-d one 
        self.activation_matrix = [] #Define inital array
        self.error_matrix = [] 
        self.z_matrix = [] #Activation before it is sigmoided
        for layer_index,layer in enumerate(layers):
            self.activation_matrix.append([]) #for each layer add an additional array
            self.error_matrix.append([])
            self.z_matrix.append([])
            for i in range(0, layer):
                self.activation_matrix[layer_index].append(0)
                self.error_matrix[layer_index].append(0)
                self.z_matrix[layer_index].append(0)
        #Create weight matrix
        # This will be a 3d matrix with each x,y containing a 1-d array of weights
        # These arrays contain the data for getting the activation from the PREVIOUS layer of neurons
        self.weight_matrix = [0] * (len(layers))
        self.bias_matrix = [0] * len(layers)
        for layer_index, layer_size in enumerate(layers):
            
            # create an array for each node
            self.weight_matrix[layer_index] = [0] * layer_size
            self.bias_matrix[layer_index] = [0] * layer_size #This one is only 2-d
            for node_index in range(0, layers[layer_index]):
                # For each node create the weight array of length equal to the next layer
                self.weight_matrix[layer_index][node_index] = [0] * (layers[layer_index - 1])
        
        # Initialize all weights to a random value
        for layer_index, layer in enumerate(self.weight_matrix):
            for node_index, node in enumerate(layer):
                for weight_index, weight in enumerate(node):
                    self.weight_matrix[layer_index][node_index][weight_index] = random.random() - 0.5
                    self.bias_matrix[layer_index][node_index] = random.random() - 0.5
        
    def createActivationMatrix(self):
        self.activation_matrix = [] #Define inital array
        for layer_index,layer in enumerate(self.bias_matrix):
            self.activation_matrix.append([]) #for each layer add an additional array
            for i in range(0, len(self.bias_matrix[layer_index])):
                self.activation_matrix[layer_index].append(0)

    def activate(self, inputs):
        self.createActivationMatrix()#reset activations
        # Check for correct number of inputs
        if(len(inputs) != len(self.activation_matrix[0])):
            return -1

        self.activation_matrix[0] = inputs
        for layer_index, layer in enumerate(self.weight_matrix):
            if(layer_index == 0):
                continue #skip input layer
            for node_index, node in enumerate(layer):
                #Compute sum of weights
                self.activation_matrix[layer_index][node_index] = np.matmul(self.activation_matrix[layer_index - 1], self.weight_matrix[layer_index][node_index])
                #Add the bias
                self.activation_matrix[layer_index][node_index] += self.bias_matrix[layer_index][node_index]
                #Assign z matrix
                self.z_matrix[layer_index][node_index] = self.activation_matrix[layer_index][node_index]
                #Normalize
                self.activation_matrix[layer_index][node_index] = sigmoid(self.activation_matrix[layer_index][node_index])
                #print(value)
        return self.activation_matrix
        
    def print(self):
        print(self.weight_matrix)
        
    def train(self, inputs, expected_outputs):
        if(len(inputs) != len(expected_outputs)):
            return -1
        #get cost function
        running_total = 0
        for case_number, case_input in enumerate(inputs):
            #Assign for future ease
            outputs = self.activate(case_input)[-1]
            #get the error function output
            case_output = np.power(np.abs(np.subtract(outputs, expected_outputs[case_number])), 2)
            case_output = case_output / 2

            #Calculate the error for the output layer
            deltaC = np.subtract(expected_outputs[case_number], outputs)
            dSigmoid = np.subtract(outputs, np.multiply(outputs, outputs))
            error = np.multiply(deltaC, dSigmoid)
            #Apply errors
            # For bias
            self.bias_matrix[-1] = np.add(self.bias_matrix[-1], self.error_matrix[-1])
            # For weights
            #Assign output layer error
            self.error_matrix[-1] = error
            weight_change = []
            for node in range(0, len(self.error_matrix[-1])):
                weight_change.append(np.multiply(self.activation_matrix[-2], self.error_matrix[-1][node])) 
            #weight_change = np.multiply(weight_change, 5)
            self.weight_matrix[-1] = np.add(self.weight_matrix[-1], weight_change)
            #do rest of errors
            for layer_index in range(len(self.activation_matrix) - 2, 0, -1):
                error = self.error_matrix[layer_index + 1]
                activations = self.activation_matrix[layer_index]
                first_part = np.matmul(np.transpose(self.weight_matrix[layer_index + 1]), error)
                dSigmoid = np.subtract(activations, np.multiply(activations, activations))
                self.error_matrix[layer_index] = np.multiply(first_part, dSigmoid)

            #Now start effecting values
            for layer_index in range(len(self.activation_matrix) - 2, 0, -1):
                # Bias first
                self.bias_matrix[layer_index] = np.add(self.bias_matrix[layer_index], self.error_matrix[layer_index])
                # Now weights
                weight_change = []
                for node in range(0, len(self.error_matrix[layer_index])):
                    weight_change.append(np.multiply(self.activation_matrix[layer_index - 1], self.error_matrix[layer_index][node])) 
                #weight_change = np.multiply(weight_change, 5)
                self.weight_matrix[layer_index] = np.add(self.weight_matrix[layer_index], weight_change)
            
            #apply transformations every 10 items
            if layer_index % 10 == 0:

        
        
a = net([4,7,3])
# print(a.activate([1]))
# a.train([[1], [1], [1], [1], [1], [1], [1], [1]], [[1], [1], [1], [1], [1], [1], [1], [1]])
# print(a.activate([1]))

inputs = []
outputs = []
with open("sincos.csv", newline='') as book:
    spamreader = csv.reader(book, delimiter=' ', quotechar='|')
    for y,row in enumerate(spamreader):
        if(y == 0):
            continue
        inputs.append([float(row[0].split(",")[0])/6, float(row[0].split(",")[1])/3, float(row[0].split(",")[2])/3.76, float(row[0].split(",")[3])/1.2])
        if row[0].split(",")[4] == "Iris-setosa":
            outputs.append([1, 0, 0])
        elif row[0].split(",")[4] == "Iris-versicolor":
            outputs.append([0, 1, 0])
        else:
            outputs.append([0, 0, 1])
        #outputs.append([float(row[0].split(",")[7])])

#shuffle inputs and outputs
for i in range(0, len(inputs)):
    t = random.randint(0, len(inputs) - 1)
    inputs[i], inputs[t] = inputs[t], inputs[i]
    outputs[i], outputs[t] = outputs[t], outputs[i]
# a.print()
# print(inputs[0][0])
print(a.activate([5.1,3.5,1.4,0.2]))
a.train(inputs, outputs)  
print(a.activate([5.1,3.5,1.4,0.2]))
print(a.activate([5.5,2.3,4.0,1.3]))
print(a.activate([6.3,2.9,5.6,1.8]))


# print(outputs)
