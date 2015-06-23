function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = OSELM_VARY(TrainingData_File, TestingData_File, Elm_Type, nHiddenNeurons, ActivationFunction, N0, Block_Range)

% Usage: OSELM_VARY(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, N0, Block_Range)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = OSELM_VARY(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, N0, Block_Range)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% nHiddenNeurons        - Number of hidden neurons assigned to the OSELM
% ActivationFunction    - Type of activation function:
%                           'rbf' for radial basis function, G(a,b,x) = exp(-b||x-a||^2)
%                           'sig' for sigmoidal function, G(a,b,x) = 1/(1+exp(-(ax+b)))
%                           'sin' for sine function, G(a,b,x) = sin(ax+b)
%                           'hardlim' for hardlim function, G(a,b,x) = hardlim(ax+b)
% N0                    - Number of initial training data used in the initial phase of OSLEM, which is not less than the number of hidden neurons
% Block_Range           - Range from wich the size of data block randomly generated in each iteration of sequential learning phase
%
% Output: 
% TrainingTime          - Time (seconds) spent on training OSELM
% TestingTime           - Time (seconds) spent on predicting all testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classifcation
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classifcation
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: OSELM_VARY('mpg_train', 'mpg_test', 0, 25, 'rbf', 75, [10,30]);
% Sample2 classification: OSELM_VARY('segment_train', 'segment_test', 1, 180, 'sig', 280, [10,30]);

    %%%%    Authors:    
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      
    %%%%    WEBSITE:    
    %%%%    DATE:       

%%%%%%%%%%% Macro definition
REGRESSION=0; 
CLASSIFICATION=1;

%%%%%%%%%%% Load dataset
train_data=load(TrainingData_File); test_data=load(TestingData_File);
T=train_data(:,1); P=train_data(:,2:size(train_data,2));
TV.T=test_data(:,1); TV.P=test_data(:,2:size(test_data,2));
clear train_data test_data;

nTrainingData=size(P,1); 
nTestingData=size(TV.P,1);
nInputNeurons=size(P,2);

%%%%%%%%%%%% Preprocessing T in the case of CLASSIFICATION 
if Elm_Type==CLASSIFICATION
    sorted_target=sort(cat(1,T,TV.T),1);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(nTrainingData+nTestingData)
        if sorted_target(i,1) ~= label(j,1)
            j=j+1;
            label(j,1) = sorted_target(i,1);
        end
    end
    nClass=j;
    nOutputNeurons=nClass;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(nTrainingData,nClass);
    for i = 1:nTrainingData
        for j = 1:nClass
            if label(j,1) == T(i,1)
                break; 
            end
        end
        temp_T(i,j)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(nTestingData,nClass);
    for i = 1:nTestingData
        for j = 1:nClass
            if label(j,1) == TV.T(i,1)
                break; 
            end
        end
        temp_TV_T(i,j)=1;
    end
    TV.T=temp_TV_T*2-1;
end
clear temp_T temp_TV_T sorted_target

start_time_train=cputime;
%%%%%%%%%%% step 1 Initialization Phase
P0=P(1:N0,:); 
T0=T(1:N0,:);

IW = rand(nHiddenNeurons,nInputNeurons)*2-1;
switch lower(ActivationFunction)
    case{'rbf'}
        Bias = rand(1,nHiddenNeurons);
%        Bias = rand(1,nHiddenNeurons)*1/3+1/11;     %%%%%%%%%%%%% for the cases of Image Segment and Satellite Image
%        Bias = rand(1,nHiddenNeurons)*1/20+1/60;    %%%%%%%%%%%%% for the case of DNA
        H0 = RBFun(P0,IW,Bias);
    case{'sig'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = SigActFun(P0,IW,Bias);
    case{'sin'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = SinActFun(P0,IW,Bias);
    case{'hardlim'}
        Bias = rand(1,nHiddenNeurons)*2-1;
        H0 = HardlimActFun(P0,IW,Bias);
        H0 = double(H0);
end

M = pinv(H0' * H0);
beta = pinv(H0) * T0;
clear P0 T0 H0;

%%%%%%%%%%%%% step 2 Sequential Learning Phase
n = N0 + 1;
while n <= nTrainingData
    Block = randint(1,1,Block_Range);    
    if (n+Block-1) > nTrainingData
        Pn = P(n:nTrainingData,:);    Tn = T(n:nTrainingData,:);
        Block = size(Pn,1);             %%%% correct the block size
    else
        Pn = P(n:(n+Block-1),:);    Tn = T(n:(n+Block-1),:);
    end 
    switch lower(ActivationFunction)
        case{'rbf'}
            H = RBFun(Pn,IW,Bias);
        case{'sig'}
            H = SigActFun(Pn,IW,Bias);
        case{'sin'}
            H = SinActFun(Pn,IW,Bias);
        case{'hardlim'}
            H = HardlimActFun(Pn,IW,Bias);
    end    
    M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M; 
    beta = beta + M * H' * (Tn - H * beta);
    n = n + Block;
end
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train        
clear Pn Tn H M;

switch lower(ActivationFunction)
    case{'rbf'}
        HTrain = RBFun(P, IW, Bias);
    case{'sig'}
        HTrain = SigActFun(P, IW, Bias);
    case{'sin'}
        HTrain = SinActFun(P, IW, Bias);
    case{'hardlim'}
        HTrain = HardlimActFun(P, IW, Bias);
end
Y=HTrain * beta;
clear HTrain;

%%%%%%%%%%% Performance Evaluation
start_time_test=cputime; 
switch lower(ActivationFunction)
    case{'rbf'}
        HTest = RBFun(TV.P, IW, Bias);
    case{'sig'}
        HTest = SigActFun(TV.P, IW, Bias);
    case{'sin'}
        HTest = SinActFun(TV.P, IW, Bias);
    case{'hardlim'}
        HTest = HardlimActFun(TV.P, IW, Bias);
end    
TY=HTest * beta;
clear HTest;
end_time_test=cputime;
TestingTime=end_time_test-start_time_test  

if Elm_Type == REGRESSION
    %%%%%%%%%%%%%% Calculate RMSE in the case of REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))               
    TestingAccuracy=sqrt(mse(TV.T - TY))            
elseif Elm_Type == CLASSIFICATION
%%%%%%%%%% Calculate correct classification rate in the case of CLASSIFICATION
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : nTrainingData
        [x, label_index_expected]=max(T(i,:));
        [x, label_index_actual]=max(Y(i,:));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/nTrainingData
    for i = 1 : nTestingData
        [x, label_index_expected]=max(TV.T(i,:));
        [x, label_index_actual]=max(TY(i,:));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/nTestingData  
end