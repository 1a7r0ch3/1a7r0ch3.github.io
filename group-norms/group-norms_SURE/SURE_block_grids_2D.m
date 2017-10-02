function [B, La, Mu, SURE, VAR] = SURE_block_grids_2D(Y, ROI, V, szList, spList, estimator, laMin, laMax, nLaMax, verbose)
%
%        [B, La, Mu, SURE, VAR] = SURE_block_grids_2D(Y, ROI, V, szList, [spList=1], [estimator=prox_l12], [laMin=0], [laMax=Inf], [nLaMax=Inf], [verbose=false])
%
% INPUT:
%   'Y' - K-by-P array of K observations of P pixels.
%   'ROI' - H-by-W array of logicals indicating pixels positions;
%           sum(ROI(:)) = P.
%   'V' - 1-by-P or K-by-P array of noise variance for each pixel (and, in the
%         case K-by-P, each observation)
%   'szList' - 1-by-S or 2-by-S array indicating the list of considered 2D block
%              sizes (in pixels) in each grid block structure. If 1-by-S,
%              then blocks are squares.
%   'spList' - 1-by-S_ or 2-by-S_ array indicating the list of considered 2D block
%              separations (in pixels) in each grid block structure. S_ can be
%              1 of S. If 1, all separtions are the same. If 'spList' is
%              1-by-S, then blocks are separated by the same amount in each
%              direction [default=1].
%   'estimator' - ['prox_l12'] | 'proj_l12' | 'prox_d12' | 'proj_d12' | 'prox_rwl12' | 'prox_rwd12'
%                 the considered denoising estimators.
%   'laMin' - minimum value of penalization coefficients [default=0].
%   'laMax' - maximum value of penalization coefficients [default=Inf].
%   'nLaMax' - maximum number of considered penalization coefficients [default=Inf].
%   'verbose' - logical; set to true for displaying information about the
%               process [default=false].
%
% OUPUT:
%   'B' - 1-by-S cell array of grid block structures. 'B{s}' is a 1-by-N{s}
%         cell array of all nonoverlapping grid block structures defined by
%         szList(s) and spList(s).
%   'La' - 1-by-S cell array of nLa(s)-by-K values. 'La{s}(l,k)' contains the 'l'-th
%          scaling penalization coefficient for the 's'-th grid block structure
%          and the 'k'-th observation.
%   'Mu' - 1-by-S cell array of 1-by-N{s} cell arrays of K-by-|B{s}{n}| arrays of
%          normalizing penalization coefficients. 'Mu{s}{n}(k,b)' contains the
%          normalization for the 'b'-th block with the nonoverlapping grid block
%          structure B{s}{n} and the 'k'-th observation.
%   'SURE' - 1-by-S cell array of nLa(s)-by-K values. 'SURE{s}(l,k)' contains the
%            average SURE accross all nonoverlapping grid block structures
%            in B{s}, for the 'l'-th scaling penalization coefficient and the
%            'k'-th observation.
%
% Hugo Raguet 2015
if nargin < 5, spList  = [1 1]; end
if nargin < 6, estimator = 'prox_l12'; end
if nargin < 7, laMin = 0; end
if nargin < 8, laMax = Inf; end
if nargin < 9, nLaMax = Inf; end
if nargin < 10, verbose = false; end

if size(szList, 1) > 2, szList = szList'; end
if size(szList, 1) == 1, szList = [szList; szList]; end
if size(spList, 1) > 2, spList = spList'; end
if size(spList, 1) == 1, spList = [spList; spList]; end
S = size(szList, 2);
if size(spList, 2) == 1, spList = repmat(spList, [1 S]); end

[K, P] = size(Y);
if isnumeric(ROI) && length(ROI)==2
    ROI = true(ROI(1), ROI(2));
end
V = repmat(V, [K, P]./size(V));

eval(sprintf('sure = @SURE_%s;', estimator));
eval(sprintf('vari = @VAR_%s_mex;', estimator));
SURE = cell(1, S);
VAR = cell(1, S);
La = cell(1, S);
Mu = cell(1, S);
B = cell(1, S);
vprintf(verbose, 'SURE of %s on grids:\n', estimator);
for s=1:S
	sz = szList(:,s)';
	sp = spList(:,s)';
    vprintf(verbose, 'block size %dx%d:', sz(1), sz(2));
    vprintf(verbose, '\tblock structures and norms... ');
    B{s} = all_grids_2D_groups(ROI, sz, sp);
    N = length(B{s});
    Yb = cell(1, N);
    Vb = cell(1, N);
    VYb2 = cell(1, N);
    Mu{s} = cell(1, N);
    Bsz = cell(1, N);
	if nargout>4 && N>1, Rb = cell(1, N); end
    for n=1:N
        Vb{n} = group_norms_l1p_mex(V, B{s}{n}, 1);
        Bsz{n} = zeros(1, B{s}{n}{1}(1), class(Y));
        for g=1:B{s}{n}{1}(1)
            Bsz{n}(g) = B{s}{n}{g+1}(1);
        end
        if strfind(estimator,'d12')
            Yb{n} = group_norms_d1p_mex(Y, B{s}{n}, 2);
            VYb2{n} = group_norms_d1p_mex(Y, B{s}{n}, 2, V).^2;
        else
            Yb{n} = group_norms_l1p_mex(Y, B{s}{n}, 2);
            VYb2{n} = group_norms_l1p_mex((Y.^2).*V, B{s}{n}, 1);
        end
        switch estimator
        case {'prox_l12', 'prox_rwl12'}
            Mu{s}{n} = sqrt(Vb{n});
        case 'proj_l12'
            Mu{s}{n} = repmat(sqrt(Bsz{n}), [K, 1]);
        case {'prox_d12', 'prox_rwd12'}
            Mu{s}{n} = sqrt(bsxfun(@times, (Bsz{n} - 1)./Bsz{n}, Vb{n}));
        case 'proj_d12'
            Mu{s}{n} = repmat(sqrt(Bsz{n}-1), [K, 1]);
        otherwise
            error('unknown estimator ''%s''. Estimator must be ''pro(x|j)_(l|d)12'' or ''prox_rw(l|d)12''', estimator);
        end
	    if nargout>4 && N>1
            Rb{n} = Yb{n}./Mu{s}{n};
            Rb{n}(Mu{s}{n}==0) = 0;
        end
    end
    vprintf(verbose, 'done.');
    %%%  pool all grids together  %%%
	Yb = cat(2, Yb{:})';
	Vb = cat(2, Vb{:})';
	VYb2 = cat(2, VYb2{:})';
    Bsz = cat(2, Bsz{:})';
	Mu_ = cat(2, Mu{s}{:})';
    vprintf(verbose, '\taverage SURE... ');
    switch estimator
    case {'prox_l12', 'proj_l12'}
        [SURE{s}, La{s}] = sure(Yb, Vb, VYb2, Mu_);
    case {'prox_d12', 'proj_d12'}
        [SURE{s}, La{s}] = sure(Yb, Vb, VYb2, Mu_, Bsz);
    case 'prox_rwl12'
        [SURE{s}, La{s}] = sure(Yb, Vb, VYb2, Mu_, laMin, laMax, nLaMax);
    case 'prox_rwd12'
        [SURE{s}, La{s}] = sure(Yb, Vb, VYb2, Mu_, Bsz, laMin, laMax, nLaMax);
    end
    clear Yb Vb VYb2 Bsz Mu_
	SURE{s} = SURE{s}/N;
    % keeps only nLaMax values between laMin and laMax
    % set non interesting values to Inf
    nLa = size(La{s}, 1);
    if isempty(strfind(estimator, 'rw')) % already selected in SURE_prox_rw*
        nLam = 0;
        for k=1:K
            % values are ordered in ascending order
            laMinIdx = find(laMin(min(k,end))<La{s}(:,k), 1, 'first');
            laMaxIdx = find(laMax(min(k,end))>La{s}(:,k), 1, 'last');
            if isempty(laMinIdx)
                error('minimum requested penalization (%f) greater than the maximum of the %dth observation (%f)', laMin(min(k,end)), k, La{s}(end,k));
            else
                minIdx = max(laMinIdx - 1, 1);
            end
            if isempty(laMinIdx)
                error('maximum requested penalization (%f) lower than the minimum of the %dth observation (%f)', laMax(min(k,end)), k, La{s}(1,k));
            else
                maxIdx = min(laMaxIdx + 1, nLa);
            end
            nLak = min(nLaMax(min(k,end)), maxIdx - minIdx + 1);
            laIdx = round(linspace(minIdx, maxIdx, nLak));
            SURE{s}(1:nLak,k) = SURE{s}(laIdx,k);
            SURE{s}(nLak+1:nLa,k) = Inf;
            La{s}(1:nLak,k) = La{s}(laIdx,k);
            La{s}(nLak+1:nLa,k) = -1;
            if laMinIdx>1
                La{s}(1,k) = laMin(min(k,end));
            end
            if laMaxIdx<nLa
                La{s}(nLak,k) = laMax(min(k,end));
            end
            nLam = max(nLam, nLak);
        end
        SURE{s} = SURE{s}(1:nLam,:);
        La{s} = La{s}(1:nLam,:);
        nLa = nLam;
    end
    vprintf(verbose, 'done.');
	if nargout>4 && N>1 %actually compute variance
		vprintf(verbose, '\tvariance... '); 
        if strfind(estimator, 'prox') % 0 was added at the beginning of La
            VAR{s} = padarray(vari(Y, B{s}, Rb, La{s}(2:end,:)), [1 0], 'pre');
        else % Inf was added at the end of La
            VAR{s} = padarray(vari(Y, B{s}, Rb, La{s}(1:end-1,:)), [1 0], 'post');
        end
		vprintf(verbose, 'done.'); 
    else
        VAR{s} = 0;
	end
    vprintf(verbose, '\n');
end

end %SURE_block_grids_2D
