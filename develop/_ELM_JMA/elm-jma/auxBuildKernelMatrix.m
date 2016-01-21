function KM = auxBuildKernelMatrix(x, activationFcns, numHiddenNeurons, rndstream)
	[N,d] = size(x);
	KM.value = [];
    KM.function = [];
    KM.param.p1 = [];
    KM.param.p2 = [];
	
	for i = 1:length(activationFcns)
		switch lower(activationFcns{i})
			case {'linear', 'l', 'lin'}
				KM.value = [KM.value x];
				KM.function = [KM.function repmat({'lin'}, 1, d)];
				KM.param.p1 = [KM.param.p1 zeros(d,d)];
				KM.param.p2 = [KM.param.p2 1:d];
				
			case {'sigmoid', 's', 'sig'}
				tt = rand(rndstream, d, numHiddenNeurons(i)) * 5 - 2.5;
				bb = rand(rndstream, 1, numHiddenNeurons(i)) * 2 - 1;
				h = x * tt + ones(N, 1) * bb;
				KM.value = [KM.value  1 ./ (1 + exp(-h))];
				KM.function=[KM.function repmat({'sig'}, 1, numHiddenNeurons(i))];
				KM.param.p1=[KM.param.p1 tt];
				KM.param.p2=[KM.param.p2 bb];
				
			case {'tanh', 't'}
				tt = rand(rndstream, d, numHiddenNeurons(i)) * 5 - 2.5;
				bb = rand(rndstream, 1, numHiddenNeurons(i)) * 2 - 1;

				h = x * tt + ones(N, 1) * bb;
				KM.value = [KM.value tanh(h)];
				KM.function=[KM.function repmat({'tanh'}, 1, numHiddenNeurons(i))];
				KM.param.p1=[KM.param.p1 tt];
				KM.param.p2=[KM.param.p2 bb];
				
			case {'sin', 'sine'}
				tt = rand(rndstream, d, numHiddenNeurons(i)) * 5 - 2.5;
				bb = rand(rndstream, 1, numHiddenNeurons(i)) * 2 - 1;
				h = x * tt + ones(N, 1) * bb;
				KM.value = [KM.value sin(h)];
				KM.function = [KM.function repmat({'sine'}, 1, numHiddenNeurons(i))];
				KM.param.p1=[KM.param.p1 tt];
				KM.param.p2=[KM.param.p2 bb];
				
			case {'rbf'}
				mm = minmax(x');
				W1 = zeros(numHiddenNeurons(i), d);
				for j = 1:numHiddenNeurons(i)
					W1(j,:) = transpose(rand(1, d)) .* (mm(:,2) - mm(:,1)) + mm(:,1);
				end
				if (N > 2000)
					Y = pdist(x(randperm(rndstream, 2000),:));
				else
					Y = pdist(x);
				end
				a20 = prctile(Y, 20);
				a60 = prctile(Y, 60);
				W10 = rand(rndstream, 1, numHiddenNeurons(i)) * (a60-a20) + a20;
				vi = zeros(N, numHiddenNeurons(i));
				for j = 1:numHiddenNeurons(i)
					vi(:,j) = auxGaussianFcn(x, W1(j,:), W10(j));
				end
				KM.value = [KM.value vi];
				KM.function = [KM.function repmat({'rbf'}, 1, numHiddenNeurons(i))];
				KM.param.p1 = [KM.param.p1 W1'];
				KM.param.p2 = [KM.param.p2 W10];
				clear a20 a60 W1 W10 Y vi mm
				
			case {'gauss', 'g', 'gaussian'}
				if (N > 2000)
					Y = pdist(x(randperm(rndstream, 2000),:));
				else
					Y = pdist(x);
				end
				plower = prctile(Y,20);
				pupper = prctile(Y,60);
				if (numHiddenNeurons(i) <= N)
					MP = randperm(rndstream, N);
					MP = MP(1:numHiddenNeurons(i));
				else
					MP = ceil(rand(rndstream, 1, numHiddenNeurons(i)) * N);
				end
				W1 = x(MP,:);
				W10 = rand(rndstream, 1, numHiddenNeurons(i)) * (pupper-plower) + plower;
				
				vi = zeros(N, numHiddenNeurons(i));
				for j = 1:numHiddenNeurons(i)
					vi(:,j) = auxGaussianFcn(x, W1(j,:), W10(j));
% 					vi(:,j) = exp(-sum((x - ones(N,1) * W1(j,:)).^2, 2) / W10(j)^2);
				end
				KM.value = [KM.value vi];
				KM.function = [KM.function repmat({'gauss'}, 1, numHiddenNeurons(i))];
				KM.param.p1 = [KM.param.p1 W1'];
				KM.param.p2 = [KM.param.p2 W10];
				clear W1 W10 pupper plower Y MP
		end
	end
end
