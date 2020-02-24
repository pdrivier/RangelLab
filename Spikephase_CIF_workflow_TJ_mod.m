  function [CIF,regressor_opt,opt_order,VM,time] = Spikephase_CIF_workflow_TJ_mod(spikes, phase, gamma, SPKflag, savename)
%Description: Function designed to determine conditional intensity 
%relationships between spiking data with circular data sets (e.g. local 
%field potential phase). Additionally, provides figures for optimal 
%regressor selection (penalized negative log likelihood versus number of 
%regressors used), Q-Q plot for determining model uncertainty, and histogram 
%showing a rough indication of when spikes occurred with relation to phase. 
%Please note that use of this function requires ksdiscrete.m available at: 
%http://www.neurostat.mit.edu/software/time_rescaling

%Inputs:
%Spikes - 1xN array consisting of either a spike train (consisting of solely
%0s and 1s) or spike indices (specific time points at which spikes occurred, 
%non-negative real numbers) where is either the length of the dataset or 
%the number of spikes depending on the input format. If entering spike 
%indices, spike indices should be on the same time scale as the phase data.
%Phase - Nx1 array consisting of phase values in radians where N is the 
%length of the entire dataset.
%Gamma - The number of L1 penalty terms used in the original ordering of 
%regressors, with 100 being standard. Increasing this value may increase 
%the accuracy of the ultimate conditional intensity function, but will 
%increase the computational time of the model.
%SPKflag - Needs to be set to either ‘st’ for spiketrain or ‘si’ for spike 
%indices. Sets what format the spiking data is being sent into the model. 
%Will error if not set to one of these two values.
%Savename - A string array that will ultimately serve as the name of saved 
%workspace.

%Outputs:
%CIF - 50x1 array representation describing how the conditional intensity 
%of a neuron’s spiking changes across a single phase cycle (-pi to pi).
%regressor_opt - optimal regressor
%opt_order - scalar, optimal model order
%VM - model information, obtained from fitglm step
%Time - Duration of time for workflow computation.
%Figures -
%Figure 1: The normalized negative log-likelihood (NNLL) and the penalized 
%normalized negative log-likelihood (PNNLL) versus the number of regressors 
%used to model the data.
%Figure 2: A quantile-quantile plot describing the goodness of fit of the 
%model for the data being characterized.
%Figure 3: The conditional intensity function (probability of spiking given
%the circular dataset)versus the circular dataset.
%Histogram of the number of spikes occurring within the dataset at each 
%particular phase.
%Savefile - Workspace of the results of the workflow. Saved as a .mat file 
%with the string used for Savename. Workspace is saved automatically to 
%active folder. Depending on the size of the dataset being analyzed, some 
%non-essential components of the workflow may not be saved.

tic
savefilename = [savename '.mat'];
if SPKflag == 'st'
    spikes = spikes;
elseif SPKflag == 'si'
    spikes_temp = zeros(1, numel(phase));
    spikes_temp(round(spikes,1)) = 1;
    spikes = spikes_temp;
else
    error('SPKflag must be set as either the string spikeind or spiketrain')
end
    
kappa = linspace(.01,30,20);
deltamu = linspace(0,2*pi,20);
deltamu = deltamu(1:end-1);
regressor_temp = zeros(numel(spikes), length(deltamu)*length(kappa));
param_hold = zeros(length(deltamu)*length(kappa), 2);
for j = 1:length(deltamu)
    for l = 1:length(kappa)
        %In these loops, the parameters are being stored into an array for
        %easy recall later and the regressor matrix is being created using
        %the mean and variance pairs previously defined.
        param_hold((j-1)*length(kappa)+l,:) = [deltamu(j),kappa(l)];
        regressor_temp(:,(j-1)*length(kappa)+l) = exp(kappa(l).*cos(phase+deltamu(j)))./(2.*pi.*besseli(0,kappa(l)));
    end
end

sprintf('begin lasso glm')
%L1 Regularized GLM step. This is used to determine model ordering.
[B,fitinfo] = lassoglm(regressor_temp, spikes', 'binomial', 'Link', 'logit','NumLambda',gamma);

%Temporarily saves the L1 weights due to how computationally intensive this
%step is.
save('B_temp.mat','B')

%Flips the weight ordering in order to have increasing model complexity.
NNZ_temp = zeros(1,gamma);
C = fliplr(B);

%Clears the workspace to save RAM space.
clear B kappa deltamu

%Following steps are used to determine the number of basis functions being
%used for each gamma value and stores the mean variance pairs as well as 
%the corresponding regressors for each of these gamma values.
for i = 1:gamma
    NNZ_temp(i) = nnz(C(:,i));
end

[uniqueNNZ, ia] = sort(NNZ_temp, 'ascend');

sortorder = ia(find(uniqueNNZ ~= 0));
regressor = {};

for j = 1:numel(sortorder)
    regressor{j} = regressor_temp(:,find(C(:,sortorder(j))~=0));
end

%Using the ordering previously determined by the L1 regularization,
%weights are now created for each increasing model order as well as the
%NNLL and PNNLL of each model order.
x = {};
for i = 1:length(sortorder) 
    VM = fitglm(regressor{i},spikes', 'linear', 'Link','logit', 'Distribution', 'binomial');
    x{i} = VM.Coefficients.Estimate;
    NLL(i) = sum(-spikes'.*([ones(length(regressor_temp),1), regressor{i}]*x{i})+log(1+exp([ones(length(regressor_temp),1), regressor{i}]*x{i})));
    NNLL(i) = NLL(i)/length(spikes);
    [~, d] = size(regressor{i});
    PNNLL(i) = NNLL(i) + d/length(spikes);
    
end

%Removes the 0 model order and plots the lower end of model orders to show
%the PNNLL-model order relationship.
NNZ = uniqueNNZ(2:end);
[Q, UIA] = unique(NNZ);

figure(1)
plot(Q, NNLL(UIA),'k')
hold on
plot(Q,PNNLL(UIA),'r')
xlabel('Model Order')
ylabel('Normalized Negative Log-Likelihood')

%Find the optimal model order by looking for the global minimum. The
%regressors used for this model order are the evaluated to produce the QQ
%plot and the CIF plot
%optorder_temp = find(PNNLL_MDL == min(PNNLL_MDL));
optorder_temp = find(PNNLL == min(PNNLL));

optorder = optorder_temp(1);
modelorder = uniqueNNZ(optorder+1);
morder = modelorder(1);
params_hold = param_hold(find(C(:,sortorder(optorder))~=0),:);

%save the optimal order,regressor, and (positive) log likelihood
regressor_opt = regressor{optorder};
LL_norm = -1*(NNLL); %normalized log likelihood  
opt_order = optorder;

NNLL_plot = NNLL;
NNLL = NNLL(optorder);
VM = fitglm(regressor{optorder},spikes', 'linear', 'Link','logit', 'Distribution', 'binomial')

CIF_temp = exp([ones(length(regressor_temp),1), regressor{optorder}]*x{optorder})./(1+exp([ones(length(regressor_temp),1), regressor{optorder}]*x{optorder}));

%generate Q-Q plots
[rst,rstsort,xks,cb,rstoldsort] = ksdiscrete(CIF_temp, spikes','spiketrain');
figure(2)
plot(xks,rstsort,'k-');
hold on;
plot(xks,xks+cb,'k--',xks,xks-cb,'k--');
axis([0,1,0,1])

%Conditional intensity function produced from a single cycle just to
%represent the relationship for spiking-phase in a simpler form.
phasetemp = linspace(-pi,pi,50);
params_length = size(params_hold,1);
regressorprob = ones(50,params_length+1);
for l = 1:params_length
    regressorprob(:,l+1) = exp(params_hold(l,2).*cos(phasetemp+params_hold(l,1)))./(2.*pi.*besseli(0,params_hold(l,2)));
end
CIF = exp(regressorprob*x{optorder})./(1+exp(regressorprob*x{optorder}));

figure(3)
plot(phasetemp,CIF','r')
xlabel('Phase (rad)')
ylabel('P(Spike|Phase)')
xlim([-pi pi])

figure(4)
histogram(phase(spikes==1),50)
xlim([-pi pi])
ylabel('# of Spikes')
xlabel('Phase (rad)')

figure(5)
subplot(2,2,1)
plot(Q, NNLL_plot(UIA),'k')
hold on
plot(Q,PNNLL(UIA),'r')
xlabel('Model Order')
ylabel('Normalized Negative Log-Likelihood')

subplot(2,2,2)
plot(xks,rstsort,'k-');
hold on;
plot(xks,xks+cb,'k--',xks,xks-cb,'k--');
axis([0,1,0,1])

subplot(2,2,3)
plot(phasetemp,CIF','r')
xlabel('Phase (rad)')
ylabel('P(Spike|Phase)')
xlim([-pi pi])

subplot(2,2,4)
histogram(phase(spikes==1),50)
xlim([-pi pi])
ylabel('# of Spikes')
xlabel('Phase (rad)')

save(savefilename)
time = toc
end