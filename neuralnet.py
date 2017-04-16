import numpy as np
import collections
import random
import math



def dropout( X, p = 0. ):
    if p != 0:
        retain_p = 1 - p
        X = X * np.random.binomial(1,retain_p,size = X.shape)
        X /= retain_p
    return X
#end  

def check_network_structure( network, cost_function ):
    assert softmax_function != network.layers[-1][1] or cost_function == softmax_neg_loss,\
        "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
    assert cost_function != softmax_neg_loss or softmax_function == network.layers[-1][1],\
        "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
#end



def verify_dataset_shape_and_modify( network, dataset ):   
    assert dataset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert dataset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    data              = np.array( [instance.features for instance in dataset ] )
    targets           = np.array( [instance.targets  for instance in dataset ] )
    
    return data, targets 
#end


def add_bias(A):
    # Add a bias value of 1. The value of the bias is adjusted through
    # weights rather than modifying the input signal.
    return np.hstack(( np.ones((A.shape[0],1)), A ))
#end addBias


def confirm( promt='Do you want to continue?' ):
    prompt = '%s [%s|%s]: ' % (promt,'y','n')
    while True:
        ans = raw_input(prompt).lower()
        if ans in ['y','yes']:
            return True
        if ans in ['n','no']:
            return False
        print "Please enter y or n."
#end

def backpropagation_foundation(network, trainingset, testset, cost_function, calculate_dW, evaluation_function = None, ERROR_LIMIT = 1e-3, max_iterations = (), batch_size = 0, input_layer_dropout = 0.0, hidden_layer_dropout = 0.0, print_rate = 1000, save_trained_network = False, **kwargs):
    check_network_structure( network, cost_function ) # check for special case topology requirements, such as softmax
    
    training_data, training_targets = verify_dataset_shape_and_modify( network, trainingset )
    test_data, test_targets    = verify_dataset_shape_and_modify( network, testset )
    
    
    # Whether to use another function for printing the dataset error than the cost function. 
    # This is useful if you train the network with the MSE cost function, but are going to 
    # classify rather than regress on your data.
    if evaluation_function != None:
        calculate_print_error = evaluation_function
    else:
        calculate_print_error = cost_function
    
    batch_size                 = batch_size if batch_size != 0 else training_data.shape[0] 
    batch_training_data        = np.array_split(training_data, math.ceil(1.0 * training_data.shape[0] / batch_size))
    batch_training_targets     = np.array_split(training_targets, math.ceil(1.0 * training_targets.shape[0] / batch_size))
    batch_indices              = range(len(batch_training_data))       # fast reference to batches
    
    error                      = calculate_print_error(network.update( test_data ), test_targets )
    reversed_layer_indexes     = range( len(network.layers) )[::-1]
    
    epoch                      = 0
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1
        
        random.shuffle(batch_indices) # Shuffle the order in which the batches are processed between the iterations
        
        for batch_index in batch_indices:
            batch_data                 = batch_training_data[    batch_index ]
            batch_targets              = batch_training_targets[ batch_index ]
            batch_size                 = float( batch_data.shape[0] )
            
            input_signals, derivatives = network.update( batch_data, trace=True )
            out                        = input_signals[-1]
            cost_derivative            = cost_function( out, batch_targets, derivative=True ).T
            delta                      = cost_derivative * derivatives[-1]
            
            for i in reversed_layer_indexes:
                # Loop over the weight layers in reversed order to calculate the deltas
            
                # perform dropout
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
                
                # Update the weights with Nestrov Momentum
                network.weights[ i ] += dW
            #end weight adjustment loop
        
        error = calculate_print_error(network.update( test_data ), test_targets )
        
        if epoch%print_rate==0:
            # Show the current training status
            print "[training] Current error:", error, "\tEpoch:", epoch
    
    print "[training] Finished:"
    print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, error )
    print "[training]   Measured quality: %.4g" % network.measure_quality( training_data, training_targets, cost_function )
    print "[training]   Trained for %d epochs." % epoch
    
    return error , epoch

    if save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_network_to_file()
# end backprop


def classical_momentum(network, trainingset, testset, cost_function, momentum_factor = 0.9, **kwargs  ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    momentum = collections.defaultdict( int )
    
    def calculate_dW( layer_index, dX ):
        dW = -learning_rate * dX + momentum_factor * momentum[ layer_index ]
        momentum[ layer_index ] = dW 
        return dW
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )
#end







#neural net

default_settings = {
    # Optional settings
    "weights_low"           : -0.1,     # Lower bound on initial weight range
    "weights_high"          : 0.1,      # Upper bound on initial weight range
    "initial_bias_value"    : 0.01,
}

class NeuralNet:
    def __init__(self, settings ):
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
        # This is a helper method for setting the network weights to a previously defined list
        # as it's useful for loading a previously optimized neural network weight set.
        # The method creates a list of weight matrices. Each list entry correspond to the 
        # connection between two layers.
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
        # This will stack all the weights in the network on a list, which may be saved to the disk.
        return [w for l in self.weights for w in l.flat]
    #end
    
    
    def error(self, weight_vector, training_data, training_targets, cost_function ):
        # assign the weight_vector as the network topology
        self.set_weights( np.array(weight_vector) )
        # perform a forward operation to calculate the output signal
        out = self.update( training_data )
        # evaluate the output signal with the cost function
        return cost_function(out, training_targets )
    #end
    
    
    def measure_quality(self, training_data, training_targets, cost_function ):
        # perform a forward operation to calculate the output signal
        out = self.update( training_data )
        # calculate the mean error on the data classification
        mean_error = cost_function( out, training_targets ) / float(training_data.shape[0])
        # calculate the numeric range between the minimum and maximum output value
        range_of_predicted_values = np.max(out) - np.min(out)
        # return the measured quality 
        return 1 - (mean_error / range_of_predicted_values)
    #end
    
    
    def gradient(self, weight_vector, training_data, training_targets, cost_function ):
        # assign the weight_vector as the network topology
        self.set_weights( np.array(weight_vector) )
        
        input_signals, derivatives  = self.update( training_data, trace=True )                  
        out                         = input_signals[-1]
        cost_derivative             = cost_function(out, training_targets, derivative=True).T
        delta                       = cost_derivative * derivatives[-1]
        
        layer_indexes               = range( len(self.layers) )[::-1]    # reversed
        n_samples                   = float(training_data.shape[0])
        deltas_by_layer             = []
        
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
            deltas_by_layer.append(list((np.dot( delta, add_bias(input_signals[i]) )/n_samples).T.flat))
            
            if i!= 0:
                # i!= 0 because we don't want calculate the delta unnecessarily.
                weight_delta        = np.dot( self.weights[ i ][1:,:], delta ) # Skip the bias weight
    
                # Calculate the delta for the subsequent layer
                delta               = weight_delta * derivatives[i-1]
        #end weight adjustment loop
        
        return np.hstack( reversed(deltas_by_layer) )
    # end gradient
    
    
    def check_gradient(self, trainingset, cost_function, epsilon = 1e-4 ):
        check_network_structure( self, cost_function ) # check for special case topology requirements, such as softmax
    
        training_data, training_targets = verify_dataset_shape_and_modify( self, trainingset )
        
        # assign the weight_vector as the network topology
        initial_weights         = np.array(self.get_weights())
        numeric_gradient        = np.zeros( initial_weights.shape )
        perturbed               = np.zeros( initial_weights.shape )
        n_samples               = float(training_data.shape[0])
        
        print "[gradient check] Running gradient check..."
        
        for i in xrange( self.n_weights ):
            perturbed[i]        = epsilon
            right_side          = self.error( initial_weights + perturbed, training_data, training_targets, cost_function )
            left_side           = self.error( initial_weights - perturbed, training_data, training_targets, cost_function )
            numeric_gradient[i] = (right_side - left_side) / (2 * epsilon)
            perturbed[i]        = 0
        #end loop
        
        # Reset the weights
        self.set_weights( initial_weights )
        
        # Calculate the analytic gradient
        analytic_gradient       = self.gradient( self.get_weights(), training_data, training_targets, cost_function )
        
        # Compare the numeric and the analytic gradient
        ratio                   = np.linalg.norm(analytic_gradient - numeric_gradient) / np.linalg.norm(analytic_gradient + numeric_gradient)
        
        if not ratio < 1e-6:
            print "[gradient check] WARNING: The numeric gradient check failed! Analytical gradient differed by %g from the numerical." % ratio
            if not confirm("[gradient check] Do you want to continue?"):
                print "[gradient check] Exiting."
                import sys
                sys.exit(2)
        else:
            print "[gradient check] Passed!"
        
        return ratio
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
        """
        This method accepts a list of Instances
        
        Eg: list_of_inputs = [ Instance([0.12, 0.54, 0.84]), Instance([0.15, 0.29, 0.49]) ]
        """
        predict_data           = np.array( [instance.features for instance in predict_set ] )
        
        return self.update( predict_data )
    #end
    
    def save_network_to_file(self, filename = "network0.pkl" ):
        import cPickle, os, re
        """
        This save method pickles the parameters of the current network into a 
        binary file for persistant storage.
        """
    
        if filename == "network0.pkl":
            while os.path.exists( os.path.join(os.getcwd(), filename )):
                filename = re.sub('\d(?!\d)', lambda x: str(int(x.group(0)) + 1), filename)
    
        with open( filename , 'wb') as file:
            store_dict = {
                "n_inputs"             : self.n_inputs,
                "layers"               : self.layers,
                "n_weights"            : self.n_weights,
                "weights"              : self.weights,
            }
            cPickle.dump( store_dict, file, 2 )
    #end

    @staticmethod
    def load_network_from_file( filename ):
        import cPickle
        """
        Load the complete configuration of a previously stored network.
        """
        network = NeuralNet( {"n_inputs":1, "layers":[[0,None]]} )
    
        with open( filename , 'rb') as file:
            store_dict                   = cPickle.load(file)
        
            network.n_inputs             = store_dict["n_inputs"]            
            network.n_weights            = store_dict["n_weights"]           
            network.layers               = store_dict["layers"]
            network.weights              = store_dict["weights"]             
    
        return network
    #end
#end class