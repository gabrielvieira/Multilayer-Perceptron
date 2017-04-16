import collections
import csv
import random
import numpy as np
import math

try:
    # PYPY hasn't got scipy
    from scipy.special import expit
except:
    expit = lambda x: 1.0 / (1 + np.exp(-x))

def sum_squared_error( outputs, targets, derivative=False ):
    if derivative:
        return outputs - targets 
    else:
        return 0.5 * np.mean(np.sum( np.power(outputs - targets,2), axis = 1 ))
#end cost function
def sigmoid_function( signal, derivative=False ):
    # Prevent overflow.
    signal = np.clip( signal, -500, 500 )
    #signal
    signal = expit( signal )
    
    if derivative:
        return np.multiply(signal, 1 - signal)
    else:
        return signal

def dropout( X, p = 0. ):
    if p != 0:
        retain_p = 1 - p
        X = X * np.random.binomial(1,retain_p,size = X.shape)
        X /= retain_p
    return X
#end  
def check_network_structure( network, cost_function ):
    if network.layers[-1][1]:
        print "checked" 

def verify_dataset_shape_and_modify( network, dataset ):   
    assert dataset[0].features.shape[0] == network.n_inputs, \
        "input error"
    assert dataset[0].targets.shape[0]  == network.layers[-1][0], \
        "output error"
    
    data              = np.array( [instance.features for instance in dataset ] )
    targets           = np.array( [instance.targets  for instance in dataset ] )
    
    return data, targets 
#end
def add_bias(A):
    #
    return np.hstack(( np.ones((A.shape[0],1)), A ))

class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, target = None ):
        self.features = np.array(features)
        
        if target != None:
            self.targets  = np.array(target)
        else:
            self.targets  = None

class NeuralNet:
    def __init__(self, settings ):

        default_settings = {
            # Optional settings
            "weights_low"           : -0.1,     # Lower bound on initial weight range
            "weights_high"          : 0.1,      # Upper bound on initial weight range
            "initial_bias_value"    : 0.01,
        }

        self.__dict__.update( default_settings )
        self.__dict__.update( settings )
        
        # Count the required number of weights. This will speed up the random number generation when initializing weights
        self.n_weights = (self.n_inputs + 1) * self.layers[0][0] +\
                         sum( (self.layers[i][0] + 1) * layer[0] for i, layer in enumerate( self.layers[1:] ) )
        
        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights( self.weights_low, self.weights_high ) )
        
        # Initalize the bias to 
        for index in xrange(len(self.layers)):
            self.weights[index][:1,:] = self.initial_bias_value
    #end
    
    
    def generate_weights(self, low = -0.1, high = 0.1):
        # Generate new random weights for all the connections in the network
        return np.random.uniform(low, high, size=(self.n_weights,))
    #end
    
    
    def set_weights(self, weight_list ):

        start, stop         = 0, 0
        self.weights        = [ ]
        previous_shape      = self.n_inputs + 1 # +1 because of the bias
        
        for n_neurons, activation_function in self.layers:
            stop           += previous_shape * n_neurons
            self.weights.append( weight_list[ start:stop ].reshape( previous_shape, n_neurons ))
            
            previous_shape  = n_neurons + 1     # +1 because of the bias
            start           = stop
    #end
    
    
    def get_weights(self, ):
        # 
        return [w for l in self.weights for w in l.flat]
    #end
    
    def error(self, weight_vector, training_data, training_targets, cost_function ):
        
        self.set_weights( np.array(weight_vector) )
        # generate output signal
        out = self.update( training_data )
        # use cost function to calulate error of output
        return cost_function(out, training_targets )
    #end
    
    def update(self, input_values, trace=False ):
        # This is a forward operation in the network. This is how we 
        # calculate the network output from a set of input signals.
        output          = input_values
        
        if trace: 
            derivatives = [ ]        # collection of the derivatives of the act functions
            outputs     = [ output ] # passed through act. func.
        
        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            signal      = np.dot( output, weight_layer[1:,:] ) + weight_layer[0:1,:] # implicit bias
            output      = self.layers[i][1]( signal )
            
            if trace: 
                outputs.append( output )
                derivatives.append( self.layers[i][1]( signal, derivative = True ).T ) # the derivative used for weight update
        
        if trace: 
            return outputs, derivatives
        
        return output
    #end
    
    
    def predict(self, predict_set ):

        predict_data           = np.array( [instance.features for instance in predict_set ] )
        return self.update( predict_data )
    #end
#end class
def backpropagation(network, trainingset, testset, cost_function, calculate_dW, evaluation_function = None, ERROR_LIMIT = 1e-3, max_iterations = (), batch_size = 0, input_layer_dropout = 0.0, hidden_layer_dropout = 0.0, print_rate = 1000, save_trained_network = False, **kwargs):
    check_network_structure( network, cost_function ) # check for special case topology requirements, such as softmax
    
    training_data, training_targets = verify_dataset_shape_and_modify( network, trainingset )
    test_data, test_targets    = verify_dataset_shape_and_modify( network, testset )
    
    calculate_print_error = cost_function
    
    #batchs = instancias

    batch_size                 = batch_size if batch_size != 0 else training_data.shape[0] 
    batch_training_data        = np.array_split(training_data, math.ceil(1.0 * training_data.shape[0] / batch_size))
    batch_training_targets     = np.array_split(training_targets, math.ceil(1.0 * training_targets.shape[0] / batch_size))
    batch_indices              = range(len(batch_training_data))       # fast reference to batches
    
    error                      = calculate_print_error(network.update( test_data ), test_targets )
    reversed_layer_indexes     = range( len(network.layers) )[::-1]
    
    epoch                      = 0

    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1
        
        random.shuffle(batch_indices)
        
        for batch_index in batch_indices:

            batch_data                 = batch_training_data[    batch_index ]
            batch_targets              = batch_training_targets[ batch_index ]
            batch_size                 = float( batch_data.shape[0] )
            
            input_signals, derivatives = network.update( batch_data, trace=True )
            out                        = input_signals[-1]

            cost_derivative            = cost_function( out, batch_targets, derivative=True ).T
            delta                      = cost_derivative * derivatives[-1]
            
            for i in reversed_layer_indexes:
                #reverse calculate the deltas
            
                #dropout
                dropped = dropout( 
                            input_signals[i], 
                            # dropout probability
                            hidden_layer_dropout if i > 0 else input_layer_dropout
                        )
            
                # calculate the weight change
                dX = (np.dot( delta, add_bias(dropped) )/batch_size).T
                dW = calculate_dW( i, dX )
                
                if i != 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skip the bias weight
                    weight_delta = np.dot( network.weights[ i ][1:,:], delta )
    
                    # Calculate the delta for the subsequent layer
                    delta = weight_delta * derivatives[i-1]
                
                # Update the weights with Momentum
                network.weights[ i ] += dW
            #end weight adjustment loop
        
        error = calculate_print_error(network.update( test_data ), test_targets )
        
        if epoch%print_rate==0:
            # Show the current training status
            print "Current error:", error, "\tEpoch:", epoch
    
    print "end"
    print "error %.4g." % ( error )
    print "interacoes %d." % epoch
    
    return error , epoch
# end backprop
def backpropagation_momentum(network, trainingset, testset, cost_function, momentum_factor = 0.9, **kwargs  ):
    
    default_configuration = {
        'ERROR_LIMIT'           : 0.01, 
        'learning_rate'         : 0.03, 
        'batch_size'            : 1, 
        'print_rate'            : 1000, 
        'save_trained_network'  : False,
        'input_layer_dropout'   : 0.0,
        'hidden_layer_dropout'  : 0.0, 
        'evaluation_function'   : None,
        'max_iterations'        : ()
    }

    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    momentum = collections.defaultdict( int )
    
    def calculate_dW( layer_index, dX ):
        dW = -learning_rate * dX + momentum_factor * momentum[ layer_index ]
        momentum[ layer_index ] = dW 
        return dW
    #end
    
    return backpropagation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )
#end

#open dataset
csvfile = open('iris/iris_shuffled.csv', 'rt')
lines = csv.reader(csvfile)
dataset = list(lines)

#params for especific data set
num_network_inputs = 4
num_network_output = 3

max_neuron = 2 * max([ num_network_inputs, num_network_output ])

class_mapping = {

    0 : [1,0,0],
    1 : [0,1,0],
    2 : [0,0,1]
}

confusion_matrix  = [[0 for x in range(num_network_output + 1)] for y in range(num_network_output + 1)] 

output_layer = (num_network_output, sigmoid_function)


best_layer =  [(8 , sigmoid_function),(8 , sigmoid_function),  output_layer] 

output_table_lines = []
error_array = []
interation_array = []
accuracy_array = []

for x in range(1):

    rangeSize =  round ( len(dataset) / 10)

    x = 0
        
    leftRange = x * rangeSize
    rightRange = leftRange + rangeSize
    #define data sets
    testSet = dataset[int(leftRange) : int(rightRange)]
    trainingSet = dataset[:]
    del trainingSet[int(leftRange) : int(rightRange)]

    test_instances = []
    training_instances = []

    for item in testSet:
        test_instances.append( Instance( map(float, item[0:num_network_inputs]), class_mapping[int(item[-1])] ) )

    for item in trainingSet:
        training_instances.append( Instance( map(float,item[0:num_network_inputs]), class_mapping[int(item[-1])] ) )


    cost_function       = sum_squared_error

    settings            = {
        # Required settings
        "n_inputs"              : num_network_inputs,
        "layers"                : best_layer,

        # Optional settings
        "initial_bias_value"    : 1.0,
        "weights_low"           : -1,     # Lower bound on the initial weight value
        "weights_high"          : 1,      # Upper bound on the initial weight value
    }


    # initialize the neural network
    network             = NeuralNet( settings )

    test_error , interations = backpropagation_momentum(
            network,                            # the network to train
            training_instances,                 # specify the training set
            test_instances,                     # specify the test set
            cost_function,                      # specify the cost function to calculate error
            
            ERROR_LIMIT             = 0.01,     # define an acceptable error limit   
            learning_rate           = 0.01,     # learning rate
            momentum_factor         = 0.0,      # momentum
            # max_iterations         = 1000, 
            )


    predicted_data = network.predict( test_instances ) 

    total_correct = 0
    total_errors = 0

    for index, val in enumerate(predicted_data):

        predicted = val
        test_data = test_instances[index].targets

        test_index = np.where( test_data == np.amax(test_data) )[0][0]
        predict_index = np.where( predicted == np.amax(predicted) )[0][0]

        confusion_matrix[ test_index + 1 ][predict_index + 1] += 1

        if np.where( predicted == np.amax(predicted) ) == np.where( test_data == np.amax(test_data) ):
            total_correct +=1
            print predicted
            print test_data
            print "correto"
        else:
            print "errado"
            total_errors+=1

    error_array.append(test_error)
    interation_array.append(interations)
    accuracy_array.append( total_correct / rangeSize )



classes = ["Iris-setosa","Iris-versicolor","Iris-virginica"]

for z in range(num_network_output):
    confusion_matrix[0][z + 1] = classes[z]
    confusion_matrix[z + 1][0] = classes[z]


# fl = open('adult/confusion_matrix.csv', 'w')

# writer = csv.writer(fl)
# #writer.writerow(['label1', 'label2', 'label3']) #if needed
# for values in confusion_matrix:
#     writer.writerow(values)

# fl.close()    

print np.mean(interation_array)

# accuracy = np.mean(accuracy_array)
# interations_result = np.mean(interation_array)
# absolute_error = np.mean(error_array)
# stamdard_error = np.std(error_array)
# output_table_lines.append( [layers_array[y][0], accuracy  , absolute_error , stamdard_error , interations_result] )


