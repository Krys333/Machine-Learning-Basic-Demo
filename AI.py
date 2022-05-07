import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)) #this is for sigmoidal function. Makes everything neat and tiy so we do not have to work with extreme numbers

def sigmoid_derivative(x): #rate of change of the sigmoid. Which allow us to modify the weights, depending on the rate of change
    return x*(1-x)

training_inputs = np.array([[0,0,0,1],
                            [0,0,1,0],
                            [0,0,1,1],
                            [0,1,0,0],
                            [0,1,0,1]])

print(training_inputs)

training_outputs = np.array([[1,0,1,0,1]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((4, 1)) -1 #we as for 3 random numbers, up to the value of 1. We then subtract 1
#print("starting synaptic weights: ")
#print(synaptic_weights)


#training
for iteration in range(10000):

    input_layer = training_inputs
    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights)) #what we do here is what happens in the graph. We look for a product of a weight and an input. What we did in the diagram is multiply input and the weight

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)


print("Synaptic weights after training: ")
print(synaptic_weights)

print("outputs after training: ")
print(outputs)
#print(training_outputs)

#recognise
p = [1,0,1,0]
decision = sigmoid(np.dot(p, synaptic_weights))
print("predction of ", p," is odd", decision)



