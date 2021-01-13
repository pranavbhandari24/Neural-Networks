# Name      :   Pranav Bhandari
# Student ID:   1001551132
# Date      :   09/28/2020

import sys, random, math, numpy as np

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        print("Oops: Sigmoid Failed!")
        return 0

def convertToOneHotVectors(data):
    # The following code maps class lables to indices, which is then used to
    # create one hot vectors for each training input
    #
    # The training inputs are then returned as data, and the target outputs 
    # for each training input is returned as one hot vectors

    unique = np.unique(data)
    indices = [i for i in range(len(unique))]
    mapping = dict(zip(unique, indices))
    reverse_mapping = dict(zip(indices, unique))
    target_output = []
    for i in range(len(data)):
        vector = [0.0 for j in range(len(unique))]
        vector[mapping.get(data[i])] = 1.0
        target_output.append(vector)
    return target_output, len(unique), reverse_mapping

def dataPreprocessing(filename):
    # This function reads the file and normalizes the input
    # It also creates one hot vectors for all the outputs

    file = open(filename, "r")
    data = []
    output = []
    max_value = sys.float_info.min
    for row in file:
        temp = row.split()
        length = len(temp)
        intermediate = []
        for i in range(length-1):
            intermediate.append(float(temp[i]))
            if float(temp[i]) > max_value:
                max_value = float(temp[i])
        output.append(temp[length-1])
        data.append(intermediate)

    # Normalizing the attribute values with the MAXIMUM ABSOLUTE value over all attributes over all training objects for that dataset. 
    data = [[float(num/max_value) for num in row] for row in data]
    return data, output

def training(training_file, layers, units_per_layer, rounds):
    # Index starts from 0 and not 1 for both units and layers

    # The training file is read and the associated target values are converted to one-hot vectors
    training_input, output = dataPreprocessing(training_file)
    training_output, num_classes, reverse_mapping = convertToOneHotVectors(output)
    num_attributes = len(training_input[0])    

    # The following lines initialize the weights b and w to small random float values between -0.5 and 0.5
    # The b and w values don't exist for the first layer and hence are not initialized
    unitsInEachLayer = [units_per_layer for i in range(layers)]
    unitsInEachLayer[0] = num_attributes
    unitsInEachLayer[layers-1] = num_classes

    b = [[] for i in range(layers)]
    w = [[] for i in range(layers)]
    for l in range(1, layers):
        b[l] = [random.uniform(-0.05, 0.05) for i in range(unitsInEachLayer[l])]
        w[l] = [[random.uniform(-0.05, 0.05) for j in range(unitsInEachLayer[l-1])] for i in range(unitsInEachLayer[l])]

    learning_rate = 1.0
    for r in range(rounds):
        for n in range(len(training_input)):
            z = [[] for i in range(layers)]
            a = [[] for i in range(layers)]

            z[0] = [0 for i in range(num_attributes)]
            for i in range(num_attributes):
                z[0][i] = training_input[n][i]
                
            for l in range(1, layers):
                a[l] = [0.0 for i in range(unitsInEachLayer[l])]
                z[l] = [0.0 for i in range(unitsInEachLayer[l])]
                for i in range(unitsInEachLayer[l]):
                    weighted_sum = 0.0
                    for j in range(unitsInEachLayer[l-1]):
                        weighted_sum += (w[l][i][j] * z[l-1][j])
                    a[l][i] = b[l][i] + weighted_sum
                    z[l][i] = sigmoid(a[l][i])

            delta =[[] for i in range(layers)]
            delta[layers-1] = [0 for i in range(num_classes)]

            for i in range(num_classes):
                delta[layers-1][i] = (z[layers-1][i] - training_output[n][i]) * z[layers-1][i] * (1.0-z[layers-1][i])
            
            for l in range(layers-2, 0, -1):
                delta[l] = [0 for i in range(unitsInEachLayer[l])]
                for i in range(unitsInEachLayer[l]):
                    sum = 0.0
                    for k in range(unitsInEachLayer[l+1]):
                        sum += (delta[l+1][k] * w[l+1][k][i])
                    delta[l][i] = sum * z[l][i] * (1 - z[l][i])

            for l in range(1, layers):
                for i in range(unitsInEachLayer[l]):
                    b[l][i] -= (learning_rate * delta[l][i])
                    for j in range(unitsInEachLayer[l-1]):
                        w[l][i][j] -= (learning_rate * delta[l][i] * z[l-1][j])
        learning_rate *= 0.98
    return b, w, reverse_mapping, num_classes, unitsInEachLayer

def testing(test_file, layers, b, w, reverse_mapping, num_classes, unitsInEachLayer):
    # Index starts from 0 and not 1 for both units and layers

    test_input, test_output = dataPreprocessing(test_file)
    
    num_attributes = len(test_input[0])
    accuracy = 0.0
    for n in range(len(test_input)):
        z = [[] for i in range(layers)]
        a = [[] for i in range(layers)]

        z[0] = [0.0 for i in range(num_attributes)]
        for i in range(num_attributes):
            z[0][i] = test_input[n][i]
        
        for l in range(1, layers):
            a[l] = [0.0 for i in range(unitsInEachLayer[l])]
            z[l] = [0.0 for i in range(unitsInEachLayer[l])]
            for i in range(unitsInEachLayer[l]):
                weighted_sum = 0.0
                for j in range(unitsInEachLayer[l-1]):
                    weighted_sum += (w[l][i][j] * z[l-1][j])
                a[l][i] = b[l][i] + weighted_sum
                z[l][i] = sigmoid(a[l][i])

        argmax = []
        max_value = -1
        for i in range(num_classes):
            if z[layers-1][i] > max_value:
                max_value = z[layers-1][i]
                argmax.clear()
                argmax.append(i)
            elif z[layers-1][i] == max_value:
                argmax.append(i)
    
        predicted = [reverse_mapping.get(n) for n in argmax]
        true = test_output[n]
        actual_predicted = predicted[0]
        if len(predicted)==1 and int(predicted[0]) == int(true):
            curr_accuracy = 1.0
        else:
            try:
                index = predicted.index(true)
                actual_predicted = predicted[index]
                curr_accuracy = float(1.0/len(predicted))
            except ValueError:
                curr_accuracy = 0.0
        accuracy += curr_accuracy
        print('ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}'.format(n+1, int(actual_predicted), int(true), curr_accuracy))
    print('classification accuracy={:6.4f}'.format(accuracy/len(test_input)))

def neural_network(training_file, test_file, layers, units_per_layer, rounds):
    b, w, reverse_mapping, num_classes, unitsInEachLayer = training(training_file, layers, units_per_layer, rounds)
    testing(test_file, layers, b, w, reverse_mapping, num_classes, unitsInEachLayer)

if __name__ == '__main__':
    neural_network(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))