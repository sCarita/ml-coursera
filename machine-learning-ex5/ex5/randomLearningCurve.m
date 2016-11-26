function [error_train, error_val] = randomLearningCurve(X, y, Xval, yval, lambda)
%RANDOMLEARNINGCURVE 

    m = size(X,1);
    error_train = zeros(1,m);
    error_val = zeros(1,m);

    loops = 50;
    
    for l = 1:loops
        for i = 1:m
            sel = randperm(m);
			sel = sel(1:i);
            
            % create a matrix consisting only of the randomly selected rows from X and y
			X_sel = X(sel,:);
			y_sel = y(sel,:);

			% (:2:) learn parameters theta using the randomly chose training set
			theta = trainLinearReg(X_sel, y_sel, lambda);
	
			% (:3a:) evaluate the parameters theta on the randomly chosen training set
			[J, grad] = linearRegCostFunction(X_sel, y_sel, theta, 0);

			% accumulate errors for i-training examples
			error_train(i) = error_train(i) + J;

			% ---
			% cross validation set
			% ---
			% (:1b:) ... and i examples from the cross validation set
			sel = randperm(size(Xval, 1));
			sel = sel(1:i);
			X_sel = Xval(sel,:);
			y_sel = yval(sel,:);
			% (:3b:) ... and cross validation set
			[J, grad_val] = linearRegCostFunction(X_sel, y_sel, theta, 0);

			error_val(i) = error_val(i) + J;
            
        end
    end

    error_train = error_train ./ loops;
	error_val = error_val ./ loops;
    
    	% least but not last, do some plotting to visualise our results
	plot(1:m, error_train, 1:m, error_val);
	xlabel('Number of training examples');
	ylabel('Error');
	axis([0 13 0 100]);
	legend('Train', 'Cross Validation');
end

