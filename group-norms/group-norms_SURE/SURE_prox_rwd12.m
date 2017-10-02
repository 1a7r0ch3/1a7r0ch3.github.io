function [SURE, La] = SURE_prox_rwd12(Yb, Sb2, SYb2, Mu, B, laMin, laMax, nLaMax)
%
%	     [SURE, La] = SURE_prox_rwd12(Yb, Sb2, SYb2, Mu, B, [laMin=0], [laMax=Inf], [nLaMax=Inf])
%
% compute the SURE of the reweighted prox d1,2 estimator
%
%   SURE(Y,la) = sum_{b:||Yb||_d<=lab} ||Yb||_d^2 + 
%                sum_{b:||Yb||_d> lab} ((lab+yb) - sqrt((lab+yb)^2 - 4*lab*lab))^2 / 4
%                     + ( 1 + (sqrt((lab+yb)^2-4*lab*lab) - lab)/||Yb|| )*||Sb||^2*(1-1/|b|)
%                     + ( 1 + (4*lab - (lab+yb))/sqrt((lab+yb)^2-4*lab*lab) )*lab*SYb2/||Yb||_d^3
%                + sum_b (2/|b| - 1) ||Sb||^2 ,
%
% where for all b, lab = la*mub, ||.||_d is the deviation semi-norm, and yb is
% short for ||Yb||_d.
%
% INPUT:
% 	'Yb' - L-by-K array
%	       K observations of L semi-norms of groups of multivariate normal random variables
%	       (work on each column independently)
% 	'Sb2' - L_-by-K_ array
% 		    estimates of the variances of the group semi-norms
%	        L_ is 1 or L; K_ is 1 or K.
%	'SYb2' - L_-by-K_ array
% 		     square semi-norms of the groups weighted by the variances
%	'Mu' - L_-by-K_ array
%	       normalization applied to each group regularization
% 	'laMin' - 1-by-K_ array
%             lower bound on the scaling coefficient
% 	'laMax' - 1-by-K_ array
%             upper bound on the scaling coefficient
% 	'nLaMax' - 1-by-K_ array of positive integers
%              maximum number of tested coefficients
%
% OUTPUT:
%	'SURE' - (L+1)-by-K array
%	         Stein Unbiaised Risk Estimator for the threshold values running throught the the tested observed values and 0
%	'La' - (L+1)-by-K array
%	       all the tested observed values and 0
%
% Hugo Raguet 2016
if nargin<5, laMin = 0; end
if nargin<6, laMax = Inf; end
if nargin<7, nLaMax = Inf; end

[L, K] = size(Yb);
B = repmat(B, [L K]./size(B));
Mu = repmat(Mu, [L K]./size(Mu));
Sb2 = repmat(Sb2, [L K]./size(Sb2));
SYb2 = repmat(SYb2, [L K]./size(SYb2));

% compute threshold values and sort them
Yb = Yb./Mu;
Yb(Mu==0) = 0;
[Yb, idx] = sort(Yb, 1, 'ascend');
for k=1:K
    B(:,k) = B(idx(:,k), k); 
    Mu(:,k) = Mu(idx(:,k), k); 
    Sb2(:,k) = Sb2(idx(:,k), k);
    SYb2(:,k) = SYb2(idx(:,k), k);
end

% precompute SYb2/Yb^3; normalized by Mu
SYb2 = SYb2./(Mu.^2)./(Yb.^3); % overwrite SYb2
SYb2(Yb==0) = 0;

% make sure 0 is considered as a threshold
Yb = padarray(Yb, [1 0], 0, 'pre');
Mu = padarray(Mu, [1 0], 0, 'pre');
Sb2 = padarray(Sb2, [1 0], 0, 'pre');
SYb2 = padarray(SYb2, [1 0], 0, 'pre');
L = L+1;

%% precompute the distance to observation due to thresholding
% sum_{b:||Yb||<=lab} ||Yb||^2
SURE = cumsum((Yb.*Mu).^2);
%% add overall variance correction
% B <- Sb2*(1 - 1./B)
non0 = B>0;
B(non0) = cast(1./B(non0), class(Yb));
B = padarray(B, [1 0], 0, 'pre'); % for threshold 0
B = Sb2.*(1 - B);
% + sum_b (2/|b| - 1) ||Sb||^2 ,
SURE = bsxfun(@plus, SURE, sum(Sb2) - 2*sum(B));

% preselect values of interest
La = Yb;
Lm = 0;
laIdx = zeros(min(L,max(nLaMax)), K, 'int32');
for k=1:K
    % values are ordered in ascending order
    laMinIdx = find(laMin(min(k,end))<Yb(:,k), 1, 'first');
    laMaxIdx = find(laMax(min(k,end))>Yb(:,k), 1, 'last');
    if isempty(laMinIdx)
        error('minimum requested penalization (%f) greater than the maximum of %dth observation (%f)', laMin(min(k,end)), k, La(end,k));
    else
        minIdx = max(laMinIdx - 1, 1);
    end
    if isempty(laMinIdx)
        error('maximum requested penalization (%f) lower than the minimum of %dth observation (%f)', laMax(min(k,end)), k, La(1,k));
    else
        maxIdx = min(laMaxIdx + 1, L);
    end
    Lk = min(nLaMax(min(k,end)), maxIdx - minIdx + 1);
    laIdx(1:Lk,k) = round(linspace(minIdx, maxIdx, Lk));
    laIdx(Lk+1:end,k) = -1;
    SURE(1:Lk,k) = SURE(laIdx(1:Lk,k),k);
    SURE(Lk+1:end,k) = Inf;
    La(1:Lk,k) = La(laIdx(1:Lk,k),k);
    La(Lk+1:end,k) = -1;
    if laMinIdx>1
        La(1,k) = laMin(min(k,end));
    end
    if laMaxIdx<L
        La(Lk,k) = laMax(min(k,end));
    end
    Lm = max(Lm, Lk);
end
laIdx = laIdx(1:Lm,:);
SURE = SURE(1:Lm,:);
La = La(1:Lm,:);
L = Lm;

% compute SURE
% sum_{b:||Yb||_d> lab} ((lab+yb) - sqrt((lab+yb)^2 - 4*lab*lab))^2 / 4
%                     + ( 1 + (sqrt((lab+yb)^2-4*lab*lab) - lab)/||Yb|| )*||Sb||^2*(1-1/|b|)
%                     + ( 1 + (4*lab - (lab+yb))/sqrt((lab+yb)^2-4*lab*lab) )*lab*SYb2/||Yb||_d^3
SURE = SURE_prox_rw12_mex(SURE, La, laIdx-1, Yb, B, SYb2, Mu);

end %SURE_prox_rwd12
