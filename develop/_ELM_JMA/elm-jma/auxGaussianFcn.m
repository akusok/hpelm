function out = auxGaussianFcn(X, C, sig)
% 	out = exp(sum((X - ones(size(X,1),1) * C).^2, 2) / sig^2);
	out = exp(-mean(abs((X - ones(size(X,1),1) * C)).^2, 2) / sig^2);
end