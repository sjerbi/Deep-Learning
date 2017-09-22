import time
import random
import numpy as np
from utils import *
from transfer_functions import *
from PIL import Image

class DeepNeuralNetwork(object):
    
    def __init__(self, input_layer_size, first_layer_size, second_layer_size, output_layer_size, iterations=50, learning_rate = 1, transfer_function = sigmoid):
        """
        input: number of input neurons
        first: number of hidden neurons in first hidden layer
        second: number of hidden neurons in second hidden layer
        output: number of output neurons
        iterations: how many iterations
        learning_rate: initial learning rate
        transfer_function: explicit
        dtransfer_function: derivative of transfer function
        """
       
        # initialize parameters
        self.iterations = iterations   #iterations
        self.learning_rate = learning_rate
        self.transfer_function = transfer_function
        if (transfer_function == sigmoid):
            self.dtransfer_function = dsigmoid
        elif (transfer_function == tanh):
            self.dtransfer_function = dtanh
     
        
        # initialize arrays
        self.input = input_layer_size+1  # +1 for the bias node in the input Layer
        self.first = first_layer_size+1 #+1 for the bias node in the hidden layer 
        self.second = second_layer_size+1 #+1 for the bias node in the hidden layer 
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_first = np.ones(self.first)
        self.a_second = np.ones(self.second)
        self.a_out = np.ones(self.output)
        
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_first = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.first-1))
        self.W_first_to_second = np.random.uniform(size = (self.first, self.second-1)) / np.sqrt(self.first)
        self.W_second_to_output = np.random.uniform(size = (self.second, self.output)) / np.sqrt(self.second)
       
        
    def weights_initialisation(self,wi,wo):
        self.W_input_to_first=wi # weights between input and first hidden layers
        self.W_first_to_second=wh # weights between first hidden and second hidden layers
        self.W_second_to_output=wo # weights between second hidden and output layers
   

       
        
    #========================Begin implementation section 1============================================="    
    
    def feedForward(self, inputs):
        
        self.W_input_to_first = np.matrix(self.W_input_to_first)
        self.W_first_to_second = np.matrix(self.W_first_to_second)
        self.W_second_to_output = np.matrix(self.W_second_to_output)

        inputs = np.append(inputs,[1])
        self.a_input = np.matrix(inputs)
        # Compute input activations
        activation_input = self.a_input*self.W_input_to_first
        
        #Compute first hidden activations
        output_first = self.transfer_function(activation_input)
        output_first = np.concatenate((output_first,np.matrix([1])), axis=1)
        self.a_first = output_first
        activation_first = output_first*self.W_first_to_second
        
        #Compute second hidden activations
        output_second = self.transfer_function(activation_first)
        output_second = np.concatenate((output_second,np.matrix([1])), axis=1)
        self.a_second = output_second
        activation_second = output_second*self.W_second_to_output

        
        # Compute output activations
        output_last = self.transfer_function(activation_second)
        self.a_out = output_last
        return(output_last)

       
     #========================End implementation section 1==============================================="   
        
        
        
        
     #========================Begin implementation section 2=============================================#    

    def backPropagate(self, targets):
        
        # calculate error terms for output
        targets = np.matrix(targets)
        error_last = self.a_out - targets
        error_output = np.multiply(error_last,self.dtransfer_function(self.a_out))
        # calculate error terms for second hidden
        error_second = np.multiply((self.W_second_to_output*error_output.T).T,self.dtransfer_function(self.a_second))
        error_second = np.matrix(np.array(error_second)[0][:-1])
        # calculate error terms for first hidden
        error_first = np.multiply((self.W_first_to_second*error_second.T).T,self.dtransfer_function(self.a_first))
        error_first = np.matrix(np.array(error_first)[0][:-1])
        
        # update output weights
        self.W_second_to_output=self.W_second_to_output-self.learning_rate*self.a_second.T*error_output
        # update intermediate weights
        self.W_first_to_second=self.W_first_to_second-self.learning_rate*self.a_first.T*error_second
        # update input weights
        self.W_input_to_first=self.W_input_to_first-self.learning_rate*self.a_input.T*error_first
        # calculate error
        error=0.5*np.dot(np.transpose(error_last),error_last)
        return(np.array(error)[0][0])
        
     #========================End implementation section 2 =================================================="   

    
    
    
    def train(self, data,validation_data,test_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
      
        for it in range(self.iterations):
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            
            error=0.0 
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input)
                error+=self.backPropagate(Target)
            Training_accuracies.append(self.predict(data))
            
            error=error/len(data)
            errors.append(error)
            
           
            print("Iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -Test_Accuracy  %-2.2f  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data)/len(data))*100, self.predict(test_data)/100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        plot_curve(range(1,self.iterations+1),errors, "Error")
        plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")
       
        
     

    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0] ) )
            count = count + 1 if (answer - prediction) == 0 else count 
            count= count 
        return count 
    
    def guess(self, filename):
        filename='Images_test/'+filename
        img = Image.open(filename).convert('1').resize((28,28),Image.ANTIALIAS)
        input = np.array(img).flatten()
        prediction = np.argmax(self.feedForward(input))
        return(prediction)
        
    
    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'wi':self.W_input_to_hidden, 'wo':self.W_hidden_to_output}, f )
        
        
    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden=data['wi']
        self.W_hidden_to_output = data['wo']
 



    
    
   