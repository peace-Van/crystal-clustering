function [idx, mst, G, theoT] = crystalcluster(X, W, T, mode, loops, verbose)
% CRYSTALCLUSTER  Crystal(aka thermodynamics) clustering.
%   Cluster on dataset X with weights W at temperature T, 
%   using the algorithm specified by 'mode'
%   running a maximum number of 'loops'.

%   [idx, mst, G, theoT] = crystalcluster(X, W, T, mode, loops, verbose)
%   X - A m-by-n numeric matrix where each row is an observation and each 
%       column is a variable.
%   W - A m-by-1 numeric vector where each entry specifies the weight of
%       the corresponding observation. The weights will be normalized so that
%       they sum up to 1. If the weights are not given in correct format then
%       observations will be treated as equal weight.
%   T - The system temperature. To get valid clustering results it has to
%       be a positive number.
%   mode - One of 'surrogate', 'greedy', 'backtrack', 'random'
%       'surrogate' - try to find global optimum, slow. Requires Global
%                     Optimization Toolbox.
%       'greedy' - simulate solubility equilibrium on small dataset(just dissolve)
%                  may stuck into a local optimum. For research use only
%       'backtrack' - simulate solubility equilibrium on small dataset, also 
%                     greedy but both dissolution and precipitation, more 
%                     likely to find a global optimum, may not converge, 
%                     may need a stopping criterion(loops)
%       'random' - simulate solubility equilibrium on large dataset(dissolve
%                  and precipitate), only to find an acceptable solution, 
%                  require a stopping criterion(loops)
%       'backtrack' is the go-to algorithm.
%   loops - The maximum number of iterations. If not given when running the
%       'backtrack' or 'random' algorithm, the program may not terminate.
%   verbose - Whether to print each action.
%   idx - Cluster indices returned as a m-by-1 numeric column vector. 
%   mst - The resulting spanning forest after system evolution.
%   G - The lowest Gibbs energy reached.
%   theoT - The theoretical temperature.

    sz = size(X, 1);
    try
        assert(sz == size(W, 1) && size(W, 2) == 1 && isnumeric(W));
        W = W * length(W) / sum(W);
    catch
        W = ones(sz, 1);
    end
    
    G = 0;
    
    switch mode
        case 'surrogate'
            mst = my_minspantree(X, W);
            theoT = -sum(mst.Edges.Weight) / log(sz);
            
            opt = optimoptions('surrogateopt', 'Display', 'iter', 'MinSampleDistance', sqrt(2/(sz-1)), 'UseParallel', true, 'PlotFcn', [], 'OutputFcn', @stopFcn);%'MaxFunctionEvaluations', 20*(sz-1));
            [sol, G] = surrogateopt(@(m) objective_func(m, mst, W, T), zeros(sz-1, 1), ones(sz-1, 1), 1:sz-1, opt);
    
            mst = rmedge(mst, find(sol));
        
        case {'greedy', 'backtrack'}
            bt = strcmp(mode, 'backtrack');
            converge = true;

            mst = my_minspantree(X, W);
            theoT = -sum(mst.Edges.Weight) / log(sz);
            delta_H = -adjacency(mst, 'weighted');
            
            delta_S = adjacency(mst);       % initialize
            ed = mst.Edges.EndNodes;
            for i = 1:length(ed)
                % calculate the entropy increase for every bond break
                sub = rmedge(mst, i);
                S = my_entropy(sub, W);
                delta_S(ed(i, 1), ed(i, 2)) = S;
                delta_S(ed(i, 2), ed(i, 1)) = S;
            end
            delta_G = delta_H - T * delta_S;    % Gibbs free energy change for every bond break
            
            loop_cnt = 0;
            while any(delta_G < 0, 'all')
                if bt && loop_cnt >= loops
                    converge = false;
                    break;
                end
                % break/form the bond between row and col, which gives the
                % largest decrease of G
                [M, I] = min(delta_G, [], 'all', 'linear');
                [row, col] = ind2sub([sz sz], I);
                G = G + M;
                
                bins0 = conncomp(mst)';
                t = findedge(mst, row, col);
                affected_nodes = (bins0 == bins0(row) | bins0 == bins0(col));
                affected_edges = find(affected_nodes(ed(:, 1)) | affected_nodes(ed(:, 2)));
                if bt && ~t
                    % if the bond does not exist, form it
                    mst = addedge(mst, row, col, -delta_H(row, col));
                    if verbose
                        fprintf('connect %d - %d\n', col, row)
                    end
                else
                    mst = rmedge(mst, t);
                    if verbose
                        fprintf('break %d - %d\n', col, row)
                    end
                end
                
                if bt
                    delta_H(row, col) = -delta_H(row, col);
                    delta_H(col, row) = -delta_H(col, row);
                else
                    delta_H(row, col) = inf;
                    delta_H(col, row) = inf;
                end
                    
                % refresh the delta_G matrix
                % has to re-calculate the entropy increase for every
                % affected edge's break
                S0 = my_entropy(mst, W);
                for i = affected_edges'
                    t = findedge(mst, ed(i, 1), ed(i, 2));
                    if ~t
                        if bt
                            sub = addedge(mst, ed(i, 1), ed(i, 2), -delta_H(ed(i, 1), ed(i, 2)));
                        else
                            continue;
                        end
                    else
                        sub = rmedge(mst, t);                       
                    end
                    S = my_entropy(sub, W);
                    delta_S(ed(i, 1), ed(i, 2)) = S - S0;
                    delta_S(ed(i, 2), ed(i, 1)) = S - S0;
                end
                delta_G = delta_H - T * delta_S;

                loop_cnt = loop_cnt + 1;
            end
            if converge
                fprintf('Crystal clustering converged at Action %d.\n', loop_cnt)
            else
                fprintf('Crystal clustering did not converge within %d actions.\n', loops)
            end
            
        case 'random'
            mst = graph(speye(sz), 'omitselfloops');
            theoT = [];    % without calculating the MST this algorithm cannot give theoT
            G = -T * my_entropy(mst, W);
            for i = 1:loops
                p = randperm(sz, 2);
                p1 = p(1); p2 = p(2);
                to_break = findedge(mst, p1, p2);
                
                S0 = my_entropy(mst, W);
                if to_break > 0
                    sub = rmedge(mst, to_break);
                else
                    sub = addedge(mst, p1, p2, 1);
                end
                S = my_entropy(sub, W);
                dS = S - S0;
                if dS == 0
                    continue;
                end
                
                dH = -W(p1) * W(p2) / norm(X(p1, :) - X(p2, :));
                if to_break > 0
                    dH = -dH;
                end
                
                dG = dH - T * dS;
                if dG < 0
                    mst = sub;
                    G = G + dG;
                end
            end
            
    end
    
    idx = conncomp(mst);
    idx = idx';
end

function stop = stopFcn(~, optimValues, ~)
    stop = (strcmp(optimValues.flag, 'adaptive') && strcmp(optimValues.currentFlag, 'random'));
end


