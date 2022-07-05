% Program to find the optimal policy via policy iteration
% sarangpm@mail.uc.edu

clear;
policy = zeros(50,5)
for i=1:length(policy)
    policy(i,:) = policy_iterator(i);
end

plot(policy(:,2),'LineWidth',2)
set(gca,'xtick',[1 5:5:50])
xlabel("R");
ylabel("Optimal action for state 2")
title("Optimal action for state 2. 1 = Stay, 2 = Leave ")
grid on
grid minor
%set(gca,'ytick',[0:50:100])
% A global function to run policy iteration for given R
%%
function [Pi] = policy_iterator(R)
%Environment probabilities
global theta lambda r P V Pi step
theta = 0.01;    %accuracy threshould in Estimation step
lambda = 0.9;
r = [1,2,3,4,R];
P = zeros(5,5,2);
P(:,:,1) = [0.3 0.4 0.2 0.1 0;
    0.2 0.3 0.5 0 0;
    0.1 0 0.8 0.1 0
    0.4 0 0 0.6 0
    0 0 0 0 0];
P(:,:,2) = [0 0 0 0 1;
    0 0 0 0 1;
    0 0 0 0 1;
    0 0 0 0 1;
    0 0 0 0 0];

% 1. Initializing values
%disp("Initialize with arbitrary policy and values")
V = [0 0 0 0 0];
Pi = [1 1 1 1 1];
%disp("Values :")
%disp(V)
%disp("Policy :")
%disp(Pi)


%Run Policy Estimation step and Policy improvement steps iteratively till
%the policy is optimal

step = 1;
PolicyEstimate
while(~PolicyImprovement)
    step = step + 1;
    PolicyEstimate
end
%disp("Optimal policy generated")
end

% 2. Policy estimation
function PolicyEstimate()
global theta lambda r P V Pi step;

%disp(">>> Run " + step);
delta = 1;
count = 0;
while (delta >= theta)
    count = count + 1;
    delta = 0;
    for k=1:length(V)
        v = V(k);
        V(k) = sum(P(k,:,Pi(k)).*(r+lambda*V));
        delta = max(delta,abs(v-V(k)));
    end
end
%disp("Estimation step run for " + count + " times.")
V
delta

end


function policy_stable = PolicyImprovement()
global P V Pi r lambda;
%disp("Improving Policy")
policy_stable = true;
for k=1:length(V)
    %%disp("State " + k)
    old_action = Pi(k);
    for t=1:2
        %%disp("T = " + t)
        estimated_val_for_action(t) = sum(P(k,:,t).*(r+lambda*V));
    end
    [~,Pi(k)] = max(estimated_val_for_action);
    if(Pi(k) ~= old_action)
        policy_stable = false;
    end
    
end
%disp("New Policy : ")
%disp(Pi)
end