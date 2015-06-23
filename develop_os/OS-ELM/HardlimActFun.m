function H = HardlimActFun(P,IW,Bias);

%%%%%%%% Feedforward neural network using hardlim activation function
V=P*IW'; ind=ones(1,size(P,1));
BiasMatrix=Bias(ind,:);      
V=V+BiasMatrix;
H = hardlim(V);