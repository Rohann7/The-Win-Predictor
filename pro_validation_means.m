clear ; close all; clc

input_layer_size  = 12; 
hidden_layer_size = 6;  
val=1;


initial_Theta1 = pro_rand_init(input_layer_size, hidden_layer_size);
initial_Theta2 = pro_rand_init(hidden_layer_size, 1);




inn_params = [initial_Theta1(:) ; initial_Theta2(:)];

x1= load('feautures_data.txt');
y1 = load('y_data.txt');


X=[x1 y1];

X= X(randperm(size(X,1)),:);

XC=X(241:320,1:((size(X,2)-1)));
YC=X(241:320,size(X,2));

XT=X(321:400,1:((size(X,2)-1)));
YT=X(321:400,size(X,2));

Y=X(1:240,size(X,2));

X=X(1:240,1:((size(X,2)-1)));
	
options = optimset('MaxIter', 50);

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i=1:length(lambda_vec),

lambda=lambda_vec(i);



% Create "short hand" for the cost function to be minimized
costFunction = @(p) pro_cost(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   val, X, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction,inn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 val, (hidden_layer_size + 1));
				 
[pred,pred2,pred3,JC,JT,J] = predict(Theta1, Theta2,X,Y,XC,YC,XT,YT,lambda);

error_val(i)= JC;
error_train(i) = J;

end

pro_validation(X,Y,XC,YC,error_train,error_val,lambda_vec);


				 
				 
				 