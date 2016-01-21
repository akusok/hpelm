function model = elmTrain_loo(x, y, activationFcns, numHiddenNeurons, rndseed)
%
% Extreme Learning Machine training phase with LOO as criterion.
% Train neuron by neuron, and ensemble with Jackknife model averaging
%
% Input:
%   x
%   y
%   humHiddenNeurons
%
% Output:
%   model                   model which contains weights for all possible
%                           weight averaging combinations
% 

	[N,d] = size(x);
	numHiddenNeurons = double(numHiddenNeurons);
	
	if (exist('rndseed', 'var') && ~isempty(rndseed))
		elmStream = RandStream.create('mt19937ar', 'NumStreams', 1, 'Seed', rndseed);
	else
		elmStream = RandStream.getDefaultStream;
	end
	
	%% initialization of kernels
	H = auxBuildKernelMatrix(x, activationFcns, numHiddenNeurons, elmStream);
	maxNeurons = min(N - 5, size(H.value,2));   % N-5 because of kicc, keep this since I am going to compare to kicc criterion
	perm = randperm(elmStream, maxNeurons);
	H.value = H.value(:,perm);                  % permute to avoid stacked same functions
	H.function = H.function(perm);
	H.param.p1 = H.param.p1(:,perm);
	H.param.p2 = H.param.p2(perm);
	
	%% go through possible solutions (adding 1 neuron at a time)
	weights = zeros(maxNeurons+1, maxNeurons);
	yh = zeros(N,maxNeurons);
	residuals = zeros(N,maxNeurons);
% 	yhloo = zeros(N,maxNeurons);
	residualsloo = zeros(N,maxNeurons);
	loo = zeros(1,maxNeurons);
	
%	timer = CTimer();
%	timer.start();
	for i = 1:maxNeurons
		Ht = [H.value(:,1:i) ones(N,1)];
		[Q R] = qr(Ht, 0);
		opts.UT = true;
		w = linsolve(R, Q' * y, opts);

		s = R \ eye(size(R,1));
		P = Ht * (s * s');
		mydiag = dot(P, Ht, 2);
		yh(:,i) = Ht * w;
		residuals(:,i) = y - yh(:,i);
		residualsloo(:,i) = residuals(:,i) ./ (1 - mydiag);
% 		yhloo(:,i) = y - residualsloo(:,i);
		
		weights([1:i maxNeurons+1],i) = w;
		
		loo(i) = mean(residualsloo(:,i).^2);
	end
	elm_train_time = timer.getElapsedSeconds();
%	timer.stop();

	[~, idxloo] = min(loo);
	ww = weights(:,idxloo);
	% check gradient as well (TO DO)


	%% ensembling
	MIN_THRESH = 1e-3;
%	timer.start();
	jack = combineJackknife(residualsloo, MIN_THRESH);
	ensemble_time = timer.getElapsedSeconds();
%	timer.stop();


	%% build output structure
	model.KM = H;
	model.outWeights = ww;
	model.allWeights = weights;
	model.loo = loo;
	model.bestloo = idxloo;
	model.jackWeights = jack;
	model.times.train = elm_train_time;
	model.times.ensemble = ensemble_time;

end
