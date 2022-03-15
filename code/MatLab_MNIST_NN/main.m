% -------------------------------------------------
% Neural Network for classifying handwritten digits
% Project by Peter Waldert 2019
% Have fun!
% -------------------------------------------------

% Load training data from CSV file
% train_data will be a 60000x785 size matrix
if ~exist('train_data', 'var')
    disp('Loading train data');
    table = load('mnist_train.csv');
    train_data = table(:, 2:end) ./ 255;
    train_labels = table(:, 1);
    trainY = NeuralNetwork.vectorizeLabels(train_labels);

    disp('Loading test data');
    table = load('mnist_test.csv');
    table = table(1:1000, :);
    test_data = table(:, 2:end) ./ 255;
    test_labels = table(:, 1);
    testY = NeuralNetwork.vectorizeLabels(test_labels);
    disp('Finished loading train- and test data!');
end

% Initialize NeuralNetwork
net = NeuralNetwork();

% Define the network's dimensions
net.addLayer(784, 20);
net.addLayer(20, 20);
net.addLayer(20, 10);

% Set Parameters!
train_size_total = 8000; % of 60000 images
batch_size = 50;
epochs = 100;
% try neural network weight decay:
%learn_rate_f = @(ep, epmax) -atan(-20+ep/epmax*18) * 0.005;
learn_rate_f = @(ep, epmax) (-atan(-50 + ep / epmax * 49) + atan(-1)) * 1/0.77 * 0.012;
% or just use constant learning_rate:
% learn_rate_f = @(ep, epmax) 0.01;

% Train the network!
net.train_smartly( ...
train_data(1:train_size_total, :), ...
    trainY(1:train_size_total, :), ...
    batch_size, ...
    epochs, ...
    learn_rate_f, ...
    test_data, ...
    test_labels ...
);

% Evaluate the network's accuracy
accuracy = net.test(test_data, test_labels);
fprintf('Network accuracy: %.1f%%\n', accuracy * 100);

% Display the Confusion Matrix
CM = net.confusion_matrix(test_data, test_labels);

% Display an image, should be a three according to train_labels(1227) = 3
net.display_image(train_data(1227, :));
