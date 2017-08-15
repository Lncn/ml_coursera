function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

least_error = 0;

% These are sane attempts at choosing a C and sigma.
tryme = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30; 100; 300];

for ii = 1:length(tryme)
    for jj = 1:length(tryme)
        
        C_test = tryme(ii);
        sigma_test = tryme(jj);
        
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); 
        
        error = mean(double(svmPredict(model, Xval) ~= yval));


        if (ii == 1 && jj == 1) || (error < least_error)
            least_error = error;
            C = C_test;
            sigma = sigma_test;
        end
    end
end

% =========================================================================

end
