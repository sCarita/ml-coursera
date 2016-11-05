function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost Calculation

cte = 1/(m);
x = theta'*X';

for n = 1:m
   J = J + ( -y(n)*log(sigmoid(x(n))) - (1-y(n))*log(1-sigmoid(x(n))) );
end
J = cte*J;

cte = lambda/(2*m);
reg_factor = 0;
[w,~] = size(theta);
for n = 2:w
    reg_factor = reg_factor + theta(n)^2;
end

J = J + cte*reg_factor;

% Gradients Calculation
cte = 1/(m);
[w,~] = size(theta);

for i = 1:m
    grad(1) = grad(1) + (sigmoid(x(i)) - y(i))*X(i,1);
end
grad(1) = cte*grad(1);

for j = 2:w
    grad(j) = 0;
    for i = 1:m
        grad(j) = grad(j) + (sigmoid(x(i)) - y(i))*X(i,j);
    end
    grad(j) = cte*grad(j) + (lambda/m)*theta(j);
end

% =============================================================

end
