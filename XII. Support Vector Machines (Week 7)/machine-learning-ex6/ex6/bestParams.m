function [C, sigma] = bestParams(X, y, Xval, yval)
C = 1;
sigma = 0.3;

C_init = 0.01;
sigma_init = 0.01;

step = 3;
steps = 8;

C = C_init;

best_error = 1;
best_C = C_init;
best_sigma = sigma_init;

for cs = 1:steps
    sigma = sigma_init;
    for ss = 1:steps
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predicted = svmPredict(model, Xval);
        current_error = mean(double(predicted ~= yval));
        if (current_error < best_error) 
            best_error = current_error;
            best_C = C;
            best_sigma = sigma;
        end
        sigma *= step;
    end
    C *= step;
end 

C = best_C;
sigma = best_sigma;

end
