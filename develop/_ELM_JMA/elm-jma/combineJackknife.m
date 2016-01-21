function weights = combineJackknife(residuals, minThresh, criterion, numBestModels)
%
% Jackknife model averaging with some thresholding. Original Jackknife is 
% based on loo residuals.
%
% Optional arguments are focused on condisering only several models,
% i.e. taking best models based on some criterion. Minimization is assumed,
% that is, the smaller the value in this vector the better the model is.
%
% Input:
%   residuals           Residuals of the models. Matrix NxK.
%   minThresh           Minimum threshold for weights to be included. Default is 0.
%   criterion           Some criterion for models. Vector Kx1.
%   numBestModels       Number of best models to choose (based on above criterion).
%
% Output:
%   weights             Ensemble weights for all K models. Those that are
%                       not considered at all have 0 weight.
%

%
% Reference:
%   Hansen and Racine. 'Jackknife model averaging',
%   Journal of Econometrics 167():38-46, 2012
%
	
	numModels = size(residuals, 2);
	if (nargin < 2 || isempty(minThresh))
		minThresh = 0;
	end
	if (nargin < 3 || isempty(criterion))
		idx = 1:numModels;
	else
		[~,idx] = sort(criterion);
	end
	if (nargin < 4 || isempty(numBestModels))
		numBestModels = numModels;
	end

	numBest = min(max(1, numBestModels), numModels);
	idx = idx(1:numBest);
	sel = false(1,numModels);
	sel(idx) = true;
	weights = zeros(numModels,1);

	% quadratic problem solver
	S = (residuals(:,sel)' * residuals(:,sel)) / size(residuals,1);
	S = (S + S') / 2;
	w = quadprog(2*S, zeros(1,numBest), [], [], ones(1,numBest), 1, zeros(numBest,1), [], [],...
		optimset('Algorithm', 'interior-point-convex', 'Display', 'off'));
	
	list = w < minThresh;
	w(list) = 0;
	w(~list) = w(~list) ./ sum(w(~list));
	weights(sel) = w;
end

