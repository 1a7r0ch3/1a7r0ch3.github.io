function [SURE, La] = SURE_proj_d12(Yb, Sb2, SYb2, Mu, B)
%
%	     [SURE, La] = SURE_proj_d12(Yb, Sb2, SYb2, Mu, B)
%
% compute the SURE of the proj d1,2 estimator
%	
% 	SURE(Y,la) = sum_{b:||Yb||_d>=lab} (||Yb||_d - lab)^2
%                + 2 sum_{b:||Yb||>=lab} ||Sb||^2/|b| (1 - lab/||Yb||)
%                                        + lab/||Yb|| (||Sb||^2 - SYb2/||Yb||^2)
%                + 2 sum_{b:||Yb||< lab} ||Sb||^2 - sum_b ||Sb||^2 ,
%
% where for all b, lab = la*mub.
%
% INPUT:
% 	'Yb' - L-by-K array
%	       K observations of L norms of groups of multivariate normal random variables
%          (work on each column independently)
% 	'Sb2' - L_-by-K_ array
% 		    estimates of the variances of the group norms
%	        L_ is 1 or L; K_ is 1 or K.
%	'SYb2' - L_-by-K_ array
% 		     square norms of the groups weighted by the variances
%	'Mu' - L_-by-K_ array
%	       normalization applied to each group regularization
%	'B' - L_-by-K_ array
%	      cardinal of each group
%
% OUTPUT:
%	'SURE' - L-by-K array
%	         Stein Unbiaised Risk Estimator for the threshold values running
%	         throught the the tested observed values, and infinity
%	'La' -  L-by-K array
%	        all the tested observed values, and infinity
%
% Hugo Raguet 2014
[L, K] = size(Yb);
Mu = repmat(Mu, [L K]./size(Mu)  );
Sb2 = repmat(Sb2, [L K]./size(Sb2) );
SYb2 = repmat(SYb2, [L K]./size(SYb2));

non0 = B > 0;
B(non0) = 1./B(non0);
B = repmat(B, [L K]./size(B));

Rb = Yb./Mu;
Rb(Mu==0) = 0;
[Rb, idx] = sort(Rb, 1, 'descend');
for k=1:K
    B(:,k)   = B(idx(:,k),  k); 
    Yb(:,k)   = Yb(idx(:,k),  k); 
    Mu(:,k)   = Mu(idx(:,k),  k); 
    Sb2(:,k)  = Sb2(idx(:,k), k);
    SYb2(:,k) = SYb2(idx(:,k),k);
end
clear idx, non0;
Yb2 = Yb.^2;
SumMu2 = cumsum(Mu.^2);
B  = Sb2.*B;

% refine best values of La
% max(r_{b_i}, min(r_{b_{i+1}}, sum_{b:r_b>r_bi} 1/r_b (||Yb||_d^2 + SYb2/||Yb||_d^2 - (1-1/|B|)*||Sb||^2) / sum_{b:r_b>r_b_i} mub^2))
La = max(padarray(Rb(2:end,:), [1 0], 0, 'post'), min(Rb, cumsum((Yb2+SYb2./Yb2-Sb2-B)./Rb)./SumMu2));

% sum_{b:||Yb||>lab} (||Yb||_d-lab)^2
dif = cumsum(Yb2) - 2*cumsum(Yb.*Mu).*La + SumMu2.*(La.^2);
dif = padarray(dif, [1 0], 0, 'pre'); % for la = inf;
clear Yb Mu SumMu2;

% sum_{b:||Yb||>lab} ||Sb||^2/|b| + lab/||Yb|| (||Sb||^2 (1-1/|b|) - SYb2/||Yb||^2)
dofgt = (Sb2 - B - SYb2./Yb2)./Rb;
dofgt(Rb==0) = 0;
dofgt = padarray(cumsum(B) + cumsum(dofgt).*La, [1 0], 0, 'pre');
clear SYb2 Yb2 B;

% sum_{b:||Yb||<=lab} ||Sb||^2 
doflt = cumsum(Sb2(end:-1:1,:));
doflt = padarray(doflt(end:-1:1,:), [1 0], 0, 'post'); % for la = 0

SURE = dif + 2*(dofgt+doflt);
% - sum_b ||Sb||^2
SURE = bsxfun(@minus, SURE(end:-1:1,:), sum(Sb2, 1));
La = padarray(La(end:-1:1,:), [1 0], Inf, 'post');

end %SURE_proj_d12
