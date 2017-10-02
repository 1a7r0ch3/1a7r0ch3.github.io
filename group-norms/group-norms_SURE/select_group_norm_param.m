function [G, K, I, N, La] = select_group_norm_param(G, sIdx, la, Mu, normalization)
%
%        [G, K, I, N, La] = select_group_norm_param(G, sIdx, la, Mu, normalization)
%
% INPUT:
%   'G' - S-long cell of group structures. 'G{s}' is a 1-by-N{s}
%         cells of nonoverlapping group structures constituing 'G'.
%   'sIdx' - K-long array of indices in [1:S] indicating for each observation
%            the best group structure.
%   'pa' - K-long array of the corresponding best scaling penalization
%          coefficients.
%   'Mu' - S-long cell of N{s}-long cells of K-by-|G{s}{n}| array of
%          normalizing penalization coefficients. 'Mu{s}{n}(k,g)' contains the
%          normalization for the 'g'-th group within the nonoverlapping group
%          structure G{s}{n} and the 'k'-th observation.
%   'normalization' - if nonzero, all penalization coefficients are scaled by
%                     'normalization', and normalized by the number of overlaps
%                     N{s}. This actually assumes that the number of overlaps
%                     is constant along pixels, as for regular grid block
%                     structures.
%
% OUTPUT:
%   'G' - S_-long cell of the group structures in input 'G' selected by at
%         least one observation.
%   'K' - (S_+1)-long array containing the number of observations selecting each
%         group structure. 'K(S_+1)' contains the total number of observations.
%   'I' - 1-by-S_ cell array of 'K(s)'-long vector of the observation indices
%         selecting each group structure.
%   'N' - 1-by-S_ cell array containing the number of nonoverlapping group
%         structures constituing each selected group structures.
%   'La' - 1-by-S_ cell array of 1-by-N{s} cell arrays of K{s}-by-|G{s}{n}|
%          arrays of group penalization coefficients. 'La{s}{n}(k,g)' contains
%          the penalization coefficient for the 'k'-th obervation and the
%          'g'-th group of the 'n'-th nonoverlapping group structure of the 's'
%          group structure.
%
% Hugo Raguet 2014
if nargin < 5, normalization = 0; end

S = length(G);
K = zeros(1, S+1);
I = cell(1, S);
N = zeros(1, S);
La = cell(1, S);
for s=S:-1:1
    idx = find(s == sIdx);
    if isempty(idx)
        G(s) = [];
		K(s) = [];
        I(s) = [];
        N(s) = [];
		La(s) = [];
    else
        K(s) = length(idx);
        I{s} = idx;
        N(s) = length(G{s});
        La{s} = cell(1, N(s));
        lap = la(idx);
        if normalization, lap = lap*(normalization/N(s)); end
        for n=1:N(s)
            La{s}{n} = bsxfun(@times, lap, Mu{s}{n}(idx,:));
        end
    end
end
K(end) = length(la);

end %select_group_norm_param
