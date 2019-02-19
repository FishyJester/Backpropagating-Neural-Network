%% Vanishing gradient problem

[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(2);
clear xValid xTest tTest tValid
%% Fixing data
n_patters = length(xTrain);
shift_factor = sum(mean(xTrain))/n_patters;

xTrain = xTrain - shift_factor;
xTrain = [-ones(1,n_patters);xTrain];
%% Not used now
xValid = xValid - shift_factor;
xValid = [-ones(1,length(xValid));xValid];

xTest = xTest - shift_factor;
xTest = [-ones(1,length(xTest));xTest];

%% Start training

learning_rate = 0.003;

% Setup of weights for layers
output_weights = normrnd(0,1/sqrt(30), 10, 30);
output_weights = [zeros(10,1), output_weights];

layer4_weights = normrnd(0, 1/sqrt(30), 30, 30);
layer4_weights = [zeros(30,1), layer4_weights];
layer4_weights = [zeros(1,31); layer4_weights];

layer3_weights = normrnd(0, 1/sqrt(30), 30, 30);
layer3_weights = [zeros(30,1), layer3_weights];
layer3_weights = [zeros(1,31); layer3_weights];

layer2_weights = normrnd(0, 1/sqrt(30), 30, 30);
layer2_weights = [zeros(30, 1), layer2_weights];
layer2_weights = [zeros(1, 31); layer2_weights];

layer1_weights = normrnd(0,1/sqrt(784), 30, 784);
layer1_weights = [zeros(30,1), layer1_weights];
layer1_weights = [zeros(1, 785); layer1_weights];

train_energy = zeros(1,50);
output_speed = zeros(1,50);
layer4_speed = zeros(1,50);
layer3_speed = zeros(1,50);
layer2_speed = zeros(1,50);
layer1_speed = zeros(1,50);



for epoch = 1:50
    
    %Speed of learning
    
    %Training checkout
    layer1_B_training = layer1_weights * xTrain;
    layer1_training_output = 1./(1+exp(-layer1_B_training));
    layer1_training_output(1,:) = -ones(1,length(layer1_training_output));
    clear layer1_B_training
    
    layer2_B_training = layer2_weights * layer1_training_output;
    layer2_training_output = 1./(1+exp(-layer2_B_training));
    layer2_training_output(1,:) = -ones(1, length(layer2_training_output));
    clear layer2_B_training layer1_training_output
    
    layer3_B_training = layer3_weights * layer2_training_output;
    layer3_training_output = 1./(1+exp(-layer3_B_training));
    layer3_training_output(1,:) = -ones(1, length(layer3_training_output));
    clear layer3_B_training layer2_training_output
    
    layer4_B_training = layer4_weights * layer3_training_output;
    layer4_training_output = 1./(1+exp(-layer4_B_training));
    layer4_training_output(1,:) = -ones(1, length(layer4_training_output));
    clear layer4_B_training layer3_training_output
          
    B_training = output_weights * layer4_training_output;
    training_output = 1./(1+exp(-B_training));
    clear B_training layer4_training_output
    
    train_energy(epoch) = 0.5*(sum(sum((tTrain - training_output).^2)));
    
    %Actual start of epoch
    epoch_perm = randperm(n_patters);
    epoch_xTrain = xTrain(:, epoch_perm);
    epoch_tTrain = tTrain(:, epoch_perm);
    
    for i = 1:10:length(xTrain)
        
        mini_batch_xTrain = epoch_xTrain(:, i:(i+9));
        mini_batch_tTrain = epoch_tTrain(:, i:(i+9));
        
%Starting forward classification
        layer1_B = layer1_weights*mini_batch_xTrain;
        layer1_output = 1./(1+exp(-layer1_B));
        layer1_output(1,:) = -ones(1, 10);

        layer2_B = layer2_weights*layer1_output;
        layer2_output = 1./(1+exp(-layer2_B));
        layer2_output(1,:) = -ones(1, 10);
        
        layer3_B = layer3_weights*layer2_output;
        layer3_output = 1./(1+exp(-layer3_B));
        layer3_output(1,:) = -ones(1, 10);
        
        layer4_B = layer4_weights*layer3_output;
        layer4_output = 1./(1+exp(-layer4_B));
        layer4_output(1,:) = -ones(1, 10);
    
        output_B = output_weights*layer4_output;
        output = 1./(1+exp(-output_B));

% Backpropagation commence!
        output_deriv = output.*(1-output);
        output_error = (mini_batch_tTrain - output).*output_deriv;
        output_incr = learning_rate*(output_error*layer4_output.');
        
        layer4_deriv = layer4_output.*(1-layer4_output);
        layer4_error = (output_error.'*output_weights).*layer4_deriv.';
        layer4_incr = learning_rate * (layer4_error.' * layer3_output.');
        
        layer3_deriv = layer3_output.*(1-layer3_output);
        layer3_error = (layer4_error * layer4_weights).*layer3_deriv.';
        layer3_incr = learning_rate * (layer3_error.' * layer2_output.');

        layer2_deriv = layer2_output.*(1-layer2_output);
        layer2_error = (layer3_error * layer3_weights).*layer2_deriv.';
        layer2_incr = learning_rate * (layer2_error.' * layer1_output.');
    
        layer1_deriv = layer1_output.*(1-layer1_output);
        layer1_error = (layer2_error * layer2_weights).*layer1_deriv.';
        layer1_incr = learning_rate * (layer1_error.' * mini_batch_xTrain.');
    
        output_weights = output_weights + output_incr;
        layer4_weights = layer4_weights + layer4_incr;
        layer3_weights = layer3_weights + layer3_incr;
        layer2_weights = layer2_weights + layer2_incr;
        layer1_weights = layer1_weights + layer1_incr;
     
        % Reset upper row to zero to handle matlab rounding errors
        layer1_weights(1,:) = zeros(1,length(layer1_weights));
        layer2_weights(1,:) = zeros(1, length(layer2_weights));
        layer3_weights(1,:) = zeros(1,length(layer3_weights));
        layer4_weights(1,:) = zeros(1,length(layer4_weights));
        
    end
    %Learning speeds at end of epoch
        tmp = sum(-output_error,2);
        output_speed(epoch) = sqrt(tmp.' * tmp);
        
        tmp = sum(-layer4_error);
        layer4_speed(epoch) = sqrt(tmp * tmp.');
        
        tmp = sum(-layer3_error);
        layer3_speed(epoch) = sqrt(tmp * tmp.');
        
        tmp = sum(-layer2_error);
        layer2_speed(epoch) = sqrt(tmp * tmp.');
        
        tmp = sum(-layer1_error);
        layer1_speed(epoch) = sqrt(tmp * tmp.');
end

%% PLotting
q = plot(1:epoch, train_energy);
q.LineWidth = 2;
%%
p = plot(1:epoch, log(output_speed), 1:epoch, log(layer4_speed), 1:epoch, log(layer3_speed), 1:epoch, log(layer2_speed), 1:epoch, log(layer1_speed));
