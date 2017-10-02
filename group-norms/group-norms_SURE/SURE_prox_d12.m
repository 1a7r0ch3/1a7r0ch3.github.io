function [SURE, La] = SURE_prox_d12(Yb, Sb2, SYb2, Mu, B)
%
%	     [SURE, La] = SURE_prox_d12(Yb, Sb2, SYb2, Mu, B)
%
% compute the SURE of the prox d1,2 estimator
%	
%   SURE(Y,la) = sum_{b:||Yb||_d<=lab} ||Yb||_d^2 + sum_{b:||Yb||_d> lab} lab^2
%                + 2 sum_{b:||Yb||_d> lab} ||Sb||^2*(1 - lab/||Yb||_d)*(1-1/|b|)
%                                          + lab SYb2/||Yb||_d^3
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
%	'B' - L_-by-K_ array
%	      cardinal of each group
%
% OUTPUT:
%	'SURE' - (L+1)-by-K array
%	         Stein Unbiaised Risk Estimator for the threshold values running
%	         throught the the tested observed values, and 0
%	'La' - (L+1)-by-K array
%	       all the tested observed values, and 0
%
% Hugo Raguet 2014
[L, K] = size(Yb);
% make sure 0 is considered as a threshold
Yb = padarray(Yb, [1 0], 0, 'pre');
Mu = padarray(repmat(Mu, [L, K]./size(Mu)), [1 0], 0, 'pre');
Sb2 = padarray(repmat(Sb2, [L, K]./size(Sb2)),  [1 0], 0, 'pre');
SYb2 = padarray(repmat(SYb2, [L, K]./size(SYb2)), [1 0], 0, 'pre');

non0 = B>0;
B(non0) = 1./B(non0);
B = padarray(repmat(B, [L, K]./size(B)), [1 0], 0, 'pre');
Rb = Yb./Mu;
Rb(Mu==0) = 0;
[Rb, idx] = sort(Rb, 1, 'descend');
for k=1:K
    B(:,k) = B(idx(:,k), k); 
    Yb(:,k) = Yb(idx(:,k), k); 
    Mu(:,k) = Mu(idx(:,k), k); 
    Sb2(:,k) = Sb2(idx(:,k), k);
    SYb2(:,k) = SYb2(idx(:,k), k);
end
clear idx, non0;
Yb2 = Yb.^2;
B = Sb2.*(1 - B);
SumSb_Rb = (B - SYb2./Yb2)./Rb;
SumSb_Rb(Rb==0) = 0;
SumSb_Rb = cumsum(SumSb_Rb);
SumMu2 = cumsum(Mu.^2);
clear Yb Mu SYb2;

% refine best values of La
% max(r_{b_i}, min(r_{b_{i+1}},
%   sum_{b:r_b>r_bi} 1/r_b ((1-1/|b|)*||Sb||^2 - SYb2/||Yb||_d^2) / sum_{b:r_b>r_b_i} mub^2))
La = [Rb(1,:); max(padarray(Rb(2:end,:), [1 0], 0, 'post'), min(Rb, SumSb_Rb ./ SumMu2))];
clear Rb;

% sum_{b:||Yb||_d<=lab} ||Yb||_d^2 + sum_{b:||Yb||_d> lab} lab^2
dif = padarray(cumsum(Yb2(end:-1:1,:)), [1 0], 0, 'pre');
dif = dif(end:-1:1,:) + padarray(SumMu2, [1 0], 0, 'pre').*(La.^2);
clear SumMu2 Yb2;

% sum_{b:||Yb||> lab} ||Sb||^2 (1-1/|b|)
%                     - lab/||Yb|| ((1-1/|b|) ||Sb||^2 - SYb2/||Yb||^2)
dof = padarray(SumSb_Rb, [1 0], 0, 'pre').*La;
dof = padarray(cumsum(B), [1 0], 0, 'pre') - dof;
clear SumSb_Rb;

SURE = dif + 2*dof;
% + sum_b (2/|b| - 1) ||Sb||^2 ,
SURE = bsxfun(@plus, SURE(end:-1:1,:), sum(Sb2) - 2*sum(B));
La = La(end:-1:1,:);

end %SURE_prox_d12
