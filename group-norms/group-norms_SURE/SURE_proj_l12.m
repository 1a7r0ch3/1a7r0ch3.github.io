function [SURE, La] = SURE_proj_l12(Yb, Sb2, SYb2, Mu)
%
%	     [SURE, La] = SURE_proj_l12(Yb, Sb2, SYb2, Mu)
%
% compute the SURE of the proj l1,2 estimator
%
% 	SURE(Y,la) = sum_{b:||Yb||>=lab} (||Yb|| - lab)^2 +
%                2 sum_{b:||Yb||>=lab} lab/||Yb|| (||Sb||^2 - SYb2/||Yb||^2) +
%                2 sum_{b:||Yb||< lab} ||Sb||^2 - sum_b ||Sb||^2 ,
%
% where for all b, lab = la*mub.
%
% INPUT:
% 	'Yb' - L-by-K array
%	       K observations of L norms of groups of multivariate normal random variables (work on each column independently)
% 	'Sb2' - L_-by-K_ array
% 		    estimates of the variances of the group norms
%	        L_ is 1 or L; K_ is 1 or K.
%	'SYb2' - L_-by-K_ array
% 		     square norms of the groups weighted by the variances
%	'Mu' - L_-by-K_ array
%	       normalization applied to each group regularization
%
% OUTPUT:
%	'SURE' - L-by-K array
%	         Stein Unbiaised Risk Estimator for the threshold values running
%	         throught the the tested observed values, and infinity
%	'La' - L-by-K array
%	       all the tested observed values, and infinity
%
% Hugo Raguet 2014
[L, K] = size(Yb);
Mu = repmat(Mu, [L K]./size(Mu)  );
Sb2 = repmat(Sb2, [L K]./size(Sb2) );
SYb2 = repmat(SYb2, [L K]./size(SYb2));

Rb = Yb./Mu;
Rb(Mu==0) = 0;
[Rb, idx] = sort(Rb, 1, 'descend');
for k=1:K
    Yb(:,k) = Yb(idx(:,k), k); 
    Mu(:,k) = Mu(idx(:,k), k); 
    Sb2(:,k) = Sb2(idx(:,k), k);
    SYb2(:,k) = SYb2(idx(:,k),k);
end
clear idx;
Yb2 = Yb.^2;
SumMu2 = cumsum(Mu.^2);
  
% refine best values of La
% max(r_{b_i}, min(r_{b_{i+1}}, sum_{b:r_b>r_bi} 1/r_b (||Yb||^2 + SYb2/||Yb||^2 - ||Sb||^2) / sum_{b:r_b>r_b_i} mub^2))
La = max(padarray(Rb(2:end,:), [1 0], 0, 'post'), min(Rb, cumsum((Yb2 + SYb2./Yb2 - Sb2)./Rb)./SumMu2));

% sum_{b:||Yb||>lab} (||Yb||-lab)^2
dif = cumsum(Yb2) - 2*cumsum(Yb.*Mu).*La + SumMu2.*(La.^2);
dif = padarray(dif, [1 0], 0, 'pre'); % for la = inf;
clear Yb Mu SumMu2;

% sum_{b:||Yb||>lab} lab/||Yb|| (||Sb||^2 - SYb2/||Yb||^2)
dofgt = (Sb2 - SYb2./Yb2)./Rb;
dofgt(Rb==0) = 0;
dofgt = padarray(cumsum(dofgt).*La, [1 0], 0, 'pre');
clear SYb2 Yb2;

% sum_{b:||Yb||<=lab} ||Sb||^2 
doflt = cumsum(Sb2(end:-1:1,:));
doflt = padarray(doflt(end:-1:1,:), [1 0], 0, 'post'); % for la = 0

SURE = dif + 2*(dofgt+doflt);
% - sum_b ||Sb||^2
SURE = bsxfun(@minus, SURE(end:-1:1,:), sum(Sb2,1));
La = padarray(La(end:-1:1,:), [1 0], Inf, 'post');

end %SURE_proj_l12
