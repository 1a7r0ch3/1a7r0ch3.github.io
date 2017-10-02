function [SURE, VAR, W] = SURE_VAR_prox_graph_d1_mex(Y, S2, Mu, La, Eu, Ev, verbose);
%
%        [SURE, VAR, W] = SURE_VAR_prox_graph_d1_mex(Y, S2, Mu, La, Eu, Ev, verbose);
%
% SURE and variance accross edges for graph d1 denoising estimator
%
% Recall: over one edge, with d1(la,(xu,xv)) = la |xu - xv|,
%
% prox_{d1,la}((xu,xv)) = ((xu+xv)/2 + (1 - 2la/|xu - xv|)(xu - xv)/2)   
%                         ((xu+xv)/2 + (1 - 2la/|xu - xv|)(xv - xu)/2),
%                                                  if |xu - xv| >  2 la
%                         ((xu+xv)/2)
%                         ((xu+xv)/2),             if |xu - xv| <= 2 la
%
% SURE(prox_{d1,la}, (xu,xv)) = 2 la^v + su^v + sv^v  if |xu - xv| >  2 la
%                               1/2 |xu - xv|^v       if |xu - xv| <= 2 la
%
% INPUTS: (warning: real numeric type is either single or double, not both)
% Y  - observations, array of length V (real)
% S2 - noise variance on observations, array of length V (real)
% Mu - individual scaling on each edge, array of length E (real)
% La - list of overall scaling to test, array of length L (real)
% Eu - for each edge, index of one vertex, array of length E (int32)
% Ev - for each edge, index of the other vertex, array of length E (int32)
%      Every vertex should belong to at least one edge. If it is not the case, 
%      a workaround is to add an edge from the vertex to itself with a nonzero
%      penalization coefficient.
% verbose - if nonzero, display information on the progress
%
% OUTPUTS:
% SURE - SURE values for different penalization scaling, array of length L
%        (real)
% VAR  - VAR values for different penalization scaling, array of length L
%        (real)
% W    - for each node, inverse of the number of edges involving this node,
%        array of length V (real)
%
% parallel implementation with OpenMP API
%
% Reference: H. Raguet, A Signal Processing Approach to Voltage Sensitive Dye,
% Chapter V: "Risk Estimation for Parameter Selection in Proximal Denoising",
% Ph.D. Thesis, 2014.
%
% Hugo Raguet 2016
