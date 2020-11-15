classdef Layer < handle
    properties
        weights;  % matrix
        biases;  % vector
        activation;  % inline function
        activation_prime;
        
        % help properties for use by backpropagation algorithm
        nablaW;
        nablaB;
        out;
        out_prime;
    end
    
    methods
        function y = feedforward(self, x)
            y = self.activation(self.weights * x + self.biases);
        end
        
        function y = feedforward_prime(self, x)
            y = self.activation_prime(self.weights * x + self.biases);
        end
        
        function init_nablas(self)
            self.nablaW = zeros(size(self.weights));
            self.nablaB = zeros(size(self.biases));
        end
        
        function n = outNeurons(self)
            n = size(self.biases, 2);
        end
    end
end
