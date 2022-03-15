classdef NeuralNetwork < handle

    properties
        layers@cell;
    end

    methods

        function addLayer(self, neurons_in, neurons_out)
            layer = Layer();
            layer.weights = randn(neurons_out, neurons_in);
            layer.biases = randn(neurons_out, 1);
            layer.activation = @(x) 1 ./ (1 + exp(-x)); % sigmoid function
            layer.activation_prime = @(x) layer.activation(x) .* (1 - layer.activation(x));
            self.layers{numel(self.layers) + 1} = layer;
        end

        function prediction = feedforward(self, data)
            X = data';

            for k = 1:numel(self.layers)
                X = self.layers{k}.feedforward(X);
            end

            prediction = X';
        end

        function backpropagate(self, data, results)
            % store all activations
            X = data';

            for k = 1:numel(self.layers)
                self.layers{k}.out = self.layers{k}.feedforward(X);
                self.layers{k}.out_prime = self.layers{k}.feedforward_prime(X);
                X = self.layers{k}.out;
            end

            last_layer = self.layers{numel(self.layers)};
            delta = self.cost_prime(last_layer.out', results)' .* last_layer.out_prime;
            last_layer.nablaB = last_layer.nablaB + delta;
            last_layer.nablaW = last_layer.nablaW + delta * self.layers{numel(self.layers) - 1}.out';

            for l = numel(self.layers) - 1:-1:1
                layer = self.layers{l};
                delta = (self.layers{l + 1}.weights' * delta) .* layer.out_prime;
                layer.nablaB = layer.nablaB + delta;

                if l > 1
                    prev_out = self.layers{l - 1}.out';
                else
                    prev_out = data;
                end

                layer.nablaW = layer.nablaW + delta * prev_out;
            end

        end

        function cost = train_step(self, train_data, trainY, learn_rate)

            for k = 1:numel(self.layers)
                self.layers{k}.init_nablas();
            end

            for k = 1:size(train_data, 1)
                self.backpropagate(train_data(k, :), trainY(k, :));
            end

            for k = 1:numel(self.layers)
                layer = self.layers{k};
                layer.weights = layer.weights - learn_rate * layer.nablaW;
                layer.biases = layer.biases - learn_rate * layer.nablaB;
            end

            prediction = self.feedforward(train_data);
            cost = self.cost(prediction, trainY);
        end

        function train(self, train_data, trainY, steps, learn_rate, ...
                test_data, test_labels)

            for n = 1:steps
                cost = self.train_step(train_data, trainY, learn_rate);
                accuracy = self.test(test_data, test_labels);
                fprintf('Run %d/%d: cost %f, accuracy: %.1f%%\n', n, steps, cost, accuracy * 100);
            end

        end

        function train_smartly(self, train_data, trainY, bsize, epochs, ...
                learn_rate_f, test_data, test_labels)

            for epoch = 1:epochs
                learn_rate = learn_rate_f(epoch, epochs);
                td_rand_indices = randperm(size(train_data, 1));
                td = train_data(td_rand_indices, :);
                tdy = trainY(td_rand_indices, :);

                for n = 1:floor(size(train_data, 1) / bsize) - 1
                    batch = td(n * bsize:(n + 1) * bsize, :);
                    batch_y = tdy(n * bsize:(n + 1) * bsize, :);
                    cost = self.train_step(batch, batch_y, learn_rate);
                end

                accuracy = self.test(test_data, test_labels);
                fprintf('Epoch %d/%d: cost %f, accuracy: %.1f%%\n', ...
                    epoch, epochs, cost, accuracy * 100);
            end

        end

        function accuracy = test(self, test_data, test_labels)
            prediction = self.feedforward(test_data);
            accuracy = nnz(self.classifyPrediction(prediction) == test_labels) / numel(test_labels);
        end

        function digit = classify(self, data)
            digit = self.classifyPrediction(self.feedforward(data));
        end

        function M = confusion_matrix(self, test_data, test_labels)
            M = zeros(10, 10);
            classi = self.classify(test_data);

            for k = 1:numel(test_labels)
                M(classi(k) + 1, test_labels(k) + 1) = M(classi(k) + 1, test_labels(k) + 1) + 1;
            end

            figure();
            image(M);
        end

    end

    methods (Static)

        function digit = classifyPrediction(prediction)
            [~, digit] = max(prediction, [], 2);
            digit = digit - 1;
        end

        function trainY = vectorizeLabels(train_labels)
            trainY = zeros(size(train_labels, 1), 10);
            trainY(sub2ind(size(trainY), 1:size(train_labels, 1), train_labels' + 1)) = 1;
        end

        function cost = cost(prediction, results)
            cost = sum((prediction - results).^2, 'all');
        end

        function cost_prime = cost_prime(prediction, results)
            cost_prime = 2 .* (prediction - results);
        end

        function display_image(data)
            figure();
            image(reshape(data .* 255, 28, 28)');
        end

    end

end
