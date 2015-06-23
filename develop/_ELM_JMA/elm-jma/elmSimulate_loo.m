function [err yh] = elmSimulate_loo(model, x, y)

	if (nargin < 3 || isempty(y))
		y = ones(size(x,1),1) * NaN;
	end

	KM = model.KM;
	[N,d] = size(x);
	
	H = zeros(size(x,1), length(KM.function));
	for i = 1:length(KM.function)
		switch KM.function{i}
			case 'lin'
				H(:,i) = x(:,KM.param.p2(i));
				
			case 'sig'
				tt = x * KM.param.p1(:,i) + ones(N,1) * KM.param.p2(:,i);
				H(:,i) = 1 ./ (1 + exp(-tt));
				
			case 'tanh'
				tt = x * KM.param.p1(:,i) + ones(N,1) * KM.param.p2(:,i);
				H(:,i) = tanh(tt);
				
			case 'sine'
				tt = x * KM.param.p1(:,i) + ones(N,1) * KM.param.p2(:,i);
				H(:,i) = sin(tt);
				
			case {'rbf', 'gauss'}
% 				H(:,i) = exp(sum((x - ones(N,1) * KM.param.p1(:,i)').^2, 2) / KM.param.p2(:,i));
				H(:,i) = auxGaussianFcn(x, KM.param.p1(:,i)', KM.param.p2(:,i));
		end
	end
	
	yh.loo = [H(:,1:model.bestloo) ones(N,1)] * model.outWeights([1:model.bestloo end]);
	err.loo = mean((y - yh.loo).^2);
	
	ll = find(model.jackWeights > 0);
	yh.jack = zeros(N, length(model.jackWeights));
	for i = 1:length(ll)
		yh.jack(:,ll(i)) = [H(:,1:ll(i)) ones(N,1)] * model.allWeights([1:ll(i) size(KM.value,2)+1], ll(i));
	end
	yh.jack = yh.jack * model.jackWeights;
	err.jack = mean((y - yh.jack).^2);
end
