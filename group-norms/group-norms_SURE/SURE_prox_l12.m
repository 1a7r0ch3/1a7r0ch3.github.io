function [SURE, La] = SURE_prox_l12(Yb, Sb2, SYb2, Mu)
%
%	     [SURE, La] = SURE_prox_l12(Yb, Sb2, SYb2, Mu)
%
% compute the SURE of the prox l1,2 estimator
%
%   SURE(Y,la) = sum_{b:||Yb||<=lab} ||Yb||^2 + sum_{b:||Yb||> lab} lab^2 +
%                2 sum_{b:||Yb||> lab} ||Sb||^2 (1 - lab/||Yb||) + lab SYb2/||Yb||^3
%                - sum_b ||Sb||^2 ,
%
% where for all b, lab = la*mub.
%
% INPUT:
% 	'Yb' - L-by-K array
%	       K observations of L norms of groups of multivariate normal random variables
%	       (work on each column independently)
% 	'Sb2' - L_-by-K_ array
% 		    estimates of the variances of the group norms
%	        L_ is 1 or L; K_ is 1 or K.
%	'SYb2' - L_-by-K_ array
% 		     square norms of the groups weighted by the variances
%	'Mu' - L_-by-K_ array
%	       normalization applied to each group regularization
%
% OUTPUT:
%	'SURE' - (L+1)-by-K array
%	         Stein Unbiaised Risk Estimator for the threshold values running throught the the tested observed values and 0
%	'La' - (L+1)-by-K array
%	       all the tested observed values and 0
%
% Hugo Raguet 2014
[L, K] = size(Yb);

% reshape and make sure 0 is considered as a threshold
Yb = padarray(Yb, [1 0], 0, 'pre');
Mu = padarray(repmat(Mu, [L, K]./size(Mu)), [1 0], 0, 'pre');
Sb2 = padarray(repmat(Sb2, [L, K]./size(Sb2)),  [1 0], 0, 'pre');
SYb2 = padarray(repmat(SYb2, [L, K]./size(SYb2)), [1 0], 0, 'pre');

% compute threshold values and sort them
Rb = Yb./Mu;
Rb(Mu==0) = 0;
[Rb, idx] = sort(Rb, 1, 'descend');
for k=1:K
    Yb(:,k) = Yb(idx(:,k), k); 
    Mu(:,k) = Mu(idx(:,k), k); 
    Sb2(:,k) = Sb2(idx(:,k), k);
    SYb2(:,k) = SYb2(idx(:,k), k);
end
clear idx;

% precompute some quantities
Yb2 = Yb.^2;
SumSb_Rb = (Sb2 - SYb2./Yb2)./Rb;
SumSb_Rb(Rb==0) = 0;
SumSb_Rb = cumsum(SumSb_Rb);
SumMu2 = cumsum(Mu.^2);
clear Yb Mu SYb2;

% refine best values of La
% max(r_{b_i}, min(r_{b_{i+1}}, sum_{b:r_b>r_bi} 1/r_b (||Sb||^2 - SYb2/||Yb||^2) / sum_{b:r_b>r_b_i} mub^2))
La = [Rb(1,:); max(padarray(Rb(2:end,:), [1 0], 0, 'post'), min(Rb, SumSb_Rb./SumMu2))];
clear Rb;

% sum_{b:||Yb||<=lab} ||Yb||^2 + sum_{b:||Yb||>lab} lab^2
dif = padarray(cumsum(Yb2(end:-1:1,:)), [1 0], 0, 'pre');
dif = dif(end:-1:1,:) + padarray(SumMu2, [1 0], 0, 'pre').*(La.^2);
clear SumMu2 Yb2;

% sum_{b:||Yb||>lab} ||Sb||^2 - la/Rb (||Sb||^2 - SYb2/||Yb||^2)
dof = padarray(SumSb_Rb, [1 0], 0, 'pre').*La;
dof = padarray(cumsum(Sb2), [1 0], 0, 'pre') - dof;
clear SumSb_Rb;
SURE = dif + 2*dof;
% - sum_b ||Sb||^2
SURE = bsxfun(@minus, SURE(end:-1:1,:), sum(Sb2));
La = La(end:-1:1,:);

end %SURE_prox_l12
