function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  
%   for i = 1: m
%       thetaX = sum(bsxfun(@times, theta, X(:,i)),1);
%       thetaX = [thetaX, 0];%attention theta is 9 not 10, so default_theta = 0
%       ex_thetaX = exp(thetaX);
%       dome = sum(ex_thetaX); 
%       f =  f + ex_thetaX(y(i))/dome;
%       indicator = zeros(9,1);
%       indicator(y(i)) = 1;
%       temp = indicator - ex_thetaX;
%       g =  g - bsxfun(@times, X(:,i), )
%   end
A = zeros(num_classes,m);
y1 = sub2ind(size(A), y, 1:size(A,2));
A(y1) = 1;
  
h = exp(theta'*X);
h = [h;ones(1,m)];
s = sum(h,1);
for i = 1:m
    h(:,i) = h(:,i)/s(i); % Obtain hypothesis 
end
   
f = -sum(sum(A.*log(h)));

g = - X*(A-h)';

g(:,end) = [];

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  
  g=g(:); % make gradient a vector for minFunc

