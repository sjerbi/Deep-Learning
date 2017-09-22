import time
import random
import numpy as np
from utils import *
from transfer_functions import *
from PIL import Image


class NeuralNetwork(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, iterations=50, learning_rate = 1, transfer_function = sigmoid):
        """
        input: number of input neurons
        hidden: number of hidden neurons
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
        self.hidden = hidden_layer_size+1 #+1 for the bias node in the hidden layer 
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_out = np.ones(self.output)
        
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
       
        
    def weights_initialisation(self,wi,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_output=wo # weights between hidden and output layers
   

       
        
    #========================Begin implementation section 1============================================="    
    
    def feedForward(self, inputs):
        
        self.W_input_to_hidden = np.matrix(self.W_input_to_hidden)
        self.W_hidden_to_output = np.matrix(self.W_hidden_to_output)

        inputs = np.append(inputs,[1])
        self.a_input = np.matrix(inputs)
        # Compute input activations
        activation_first = self.a_input*self.W_input_to_hidden
        
        #Compute  hidden activations
        output_hidden = self.transfer_function(activation_first)
        output_hidden = np.concatenate((output_hidden,np.matrix([1])), axis=1)
        self.a_hidden = output_hidden
        activation_hidden = output_hidden*self.W_hidden_to_output

        
        # Compute output activations
        output_last = self.transfer_function(activation_hidden)
        self.a_out = output_last
        return(output_last)

       
     #========================End implementation section 1==============================================="   
        
        
        
        
     #========================Begin implementation section 2=============================================#    

    def backPropagate(self, targets):
        
        # calculate error terms for output
        targets = np.matrix(targets)
        error_last = self.a_out - targets
        error_output = np.multiply(error_last,self.dtransfer_function(self.a_out))
        # calculate error terms for hidden
        error_hidden = np.multiply((self.W_hidden_to_output*error_output.T).T,self.dtransfer_function(self.a_hidden))
        error_hidden = np.matrix(np.array(error_hidden)[0][:-1])
        
        # update output weights
        self.W_hidden_to_output=self.W_hidden_to_output-self.learning_rate*self.a_hidden.T*error_output
        # update input weights
        self.W_input_to_hidden=self.W_input_to_hidden-self.learning_rate*self.a_input.T*error_hidden
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

