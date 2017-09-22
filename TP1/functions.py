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
        