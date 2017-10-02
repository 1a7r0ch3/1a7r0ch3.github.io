function [sIdx, minSURE, pa] = best_estimators_set(SURE, VAR, Pa)
%
%	     [sIdx, minSURE, pa] = best_estimators_set(SURE, [VAR={}], [Pa])
%
% gives the parameter index and corresponding penalization scaling for a sum of simple estimators, based on the average of the SUREs of the individual estimators (when only two inputs), or on the SURE of the average of the individual estimators (when VAR is given)
%
% INPUT:
%   'SURE' - S-long cell array of nPa(s)-by-K values. 'SURE{s}(p,k)' contains the average SURE of 
%            the 's'-th set of estimators with the 'p'-th parameter, on the
%            'k'-th observation.
%   'VAR' - S-long cell array of nPa(s)-by-K values. 'VAR{s}(p,k)' contains
%           the variance accross estimators of the 's'-th set, for
%           the 'p'-th scaling penalization coefficient, on the
%           'k'-th observation. Set to empty cell for ignoring
%           variance. [default={}]
%   'Pa' - S-long cell array of nPa(s)-by-K value. 'Pa{s}(p,k)' contains the 'p'-th
%          scalar-valued parameter the 's'-th set of estimators and
%          the 'k'-th the observation.
% OUTPUT:
%   'sIdx' - 1-by-K array of indices in [1:S] indicating for each observation
%            the set of estimators minimizing the desired SURE value.
%   'minSURE' - 1-by-K array of the corresponding minimal SURE values.
%   'pa' - K-by-1 array of the corresponding best scalar-valued parameters.
%
% Hugo Raguet 2014
if nargin<2, VAR = {}; end

S = length(SURE);
K = size(SURE{1}, 2);
sIdx = zeros(1, K);
minSURE = Inf(1, K);
if nargout>2
    if nargin<3
        error('must provide the parameters array (third argument)');
    else
        pa = zeros(K, 1);
    end
end
for s=1:S
    if isempty(VAR)
		[minSUREs, pa_s] = min(SURE{s}, [], 1);
    else
		[minSUREs, pa_s] = min(SURE{s}-VAR{s}, [], 1);
    end
    k_s = find(minSUREs<minSURE);
    minSURE(k_s) = minSUREs(k_s);
    sIdx(k_s) = s;
    if nargout>2
        pa(k_s) = Pa{s}((k_s-1)*size(Pa{s}, 1) + pa_s(k_s));
    end
end

end %best_estimators_set
