function disconnected_bump = bump_detection_disconnectedRegion(x, y, lb, ub)

nx = size(x, 2);
aa = primparams();
[prim, ~] = primbuild(x, -y, aa);


disconnected_bump = false;
if length(prim.boxes) > 2
    
    region1 = prim.boxes{1};
    region2 = prim.boxes{2};
    
    if ~isempty(region2.min) %in case regions2 is empty in output field
        
        for ii = 1:2
            mentioned_var = prim.boxes{ii}.vars;
            kk = 1;
            for jj = 1:nx % varible indicator
                if ismember(jj, mentioned_var)
                    % deal with nan
                    if isnan(prim.boxes{ii}.min(kk))
                        bump_lb = lb(jj);
                    else
                        bump_lb = prim.boxes{ii}.min(kk);
                    end
                    
                    % deal with nan
                    if isnan(prim.boxes{ii}.max(kk))
                        bump_ub = ub(jj);
                    else
                        bump_ub = prim.boxes{ii}.max(kk);
                    end
                    
                    % assign range
                    region_lb(ii, jj) = bump_lb;
                    region_ub(ii, jj) = bump_ub;
                    
                    % move pointer
                    kk = kk + 1;
                else
                    region_lb(ii, jj) = lb(jj);
                    region_ub(ii, jj) = ub(jj);
                end
            end
        end
      
       % first region enclose the second region
       a1 = (region_lb(1, :) - region_ub(2, :)) >= 0;
       b1 = (region_lb(2, :) - region_ub(1, :)) >= 0;
       if sum(a1) == nx || sum(b1) == nx
           disconnected_bump = true;
       end
       
      
    end
end

% 
end




function params = primparams(alphaPeel, alphaPaste, minSupport, maxBoxes, terminateBelowMean)
% primparams
% Creates configuration for the PRIM algorithm. The output structure is for
% further use with function primbuild.
%
% Call:
%   params = primparams(alphaPeel, alphaPaste, minSupport, maxBoxes, terminateBelowMean)
%
% All the input arguments of this function are optional. Empty values are
% also accepted (the corresponding default values will be used).
%
% Input:
%   alphaPeel     : Peeling quantile. Usually chosen somewhere between 0.05
%                   and 0.1 (default value = 0.1) (Friedman & Fisher,
%                   1999). This defines the fraction of the data in the
%                   current box that is removed at each iteration. Too
%                   large values make the peeling crude. Too small values
%                   make the peeling sensitive to noise.
%   alphaPaste    : Pasting quantile. Set to 0 to disable pasting. (default
%                   value = 0.05)
%   minSupport    : The peeling of a box continues until its support (i.e.,
%                   the fraction of data covered by the box) drops below
%                   minSupport. Friedman & Fisher, 1999, call this
%                   parameter "beta0". Values lower than or equal to 1 are
%                   treated as percent of the whole data. Values larger
%                   than 1 are treated as a number of data observations.
%                   Default value = 0.1, i.e., minimum support for a box is
%                   10% of the whole data.
%   maxBoxes      : The maximum number of boxes to build. Default value =
%                   Inf. Usually the most interesting boxes are only the
%                   few first ones. Note that actually the result will
%                   always contain one box more than specified here because
%                   the last box will always be the 'leftovers' box that
%                   contains all the data observations not covered by any
%                   previous boxes.
%   terminateBelowMean : Whether to stop building boxes when the mean value
%                   of Y for a completed new box drops below the global
%                   mean of Y. Default value = true.
%
% Output:
%   params        : A structure of parameters for further use with function
%                   primbuild containing the provided values (or default
%                   ones, if not provided).

% =========================================================================
% Bump hunting using Patient Rule Induction Method for Matlab/Octave
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2015  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

% Last update: November 9, 2015

if (nargin < 1) || isempty(alphaPeel)
    params.alphaPeel = 0.1;
else
    params.alphaPeel = max(0, alphaPeel);
end

if (nargin < 2) || isempty(alphaPaste)
    params.alphaPaste = 0.05;
else
    params.alphaPaste = max(0, alphaPaste);
end

if (nargin < 3) || isempty(minSupport)
    params.minSupport = 0.1;
else
    params.minSupport = max(0, minSupport);
end

if (nargin < 4) || isempty(maxBoxes)
    params.maxBoxes = Inf;
else
    params.maxBoxes = max(1, maxBoxes);
end

if (nargin < 5) || isempty(terminateBelowMean)
    params.terminateBelowMean = true;
else
    params.terminateBelowMean = terminateBelowMean;
end
end

function [prim, trajectory] = primbuild(X, Y, params, isBinCat, Xt, Yt)
% primbuild
% Builds a sequence of boxes using PRIM.
%
% Call:
%   [prim, trajectory] = primbuild(X, Y, params, isBinCat, Xt, Yt)
%
% All the input arguments, except the first three, are optional. Empty
% values are also accepted (the corresponding default values will be used).
%
% Input:
%   X, Y          : Training data. X is a matrix with rows corresponding to
%                   observations and columns corresponding to input
%                   variables. Y is a column vector of response values.
%                   Input variables can be continuous, binary, as well as
%                   categorical (indicate using isBinCat). All values must
%                   be numeric. Missing values in X must be indicated as
%                   NaN.
%                   primbuild searches for maxima. If you are interested in
%                   minima, replace Y by -Y.
%                   For binary classification, negative class should be
%                   coded with label 0 and positive class with label 1 so
%                   that all the calculated mean values can be interpreted
%                   as the probabilities of response being equal to 1.
%                   For more than two classes, the problem cannot be
%                   handled directly. However, it can be reformulated as a
%                   collection of binary problems. Code one class with
%                   label 1 and the rest with label 0 and use primbuild for
%                   each class in turn.
%   params        : A structure of parameters for the PRIM algorithm (see
%                   function primparams for details).
%   isBinCat      : A vector of flags indicating type of each input
%                   variable - either continuous (false) or categorical
%                   (true) with any number of categories, including binary.
%                   The vector should be of the same length as the number
%                   of columns in X. primbuild then detects all the
%                   actually possible values for categorical variables from
%                   the training data. Any new values detected later, i.e.,
%                   during prediction, will be treated as NaN. By default,
%                   the vector is created with all values equal to false,
%                   meaning that all the variables are treated as
%                   continuous.
%   Xt, Yt        : Test data for implementation of the "cross-validation"
%                   stopping rule. Typically the test data set is half the
%                   size of the training data set (Friedman & Fisher,
%                   1999). If Xt,Yt are provided, the "pasting" phase in
%                   PRIM is omitted.
%                   The number of columns in Xt and X should be equal.
%
% Output:
%   prim          : A structure of the built boxes together with some
%                   additional information. The structure has the following
%                   fields:
%     params      : A structure of parameters for the PRIM algorithm.
%     info        : Information about the training data set.
%     boxes       : A cell array of the built boxes. Each element contains
%                   definition of one box together with some additional
%                   information about the box.
%   trajectory    : "Peeling trajectory" (Friedman & Fisher, 1999) for the
%                   first box in the list of boxes. First column is box
%                   support (i.e., the fraction of data covered by the
%                   box). Second column is box mean. If Xt,Yt are provided,
%                   third and fourth columns are box support and box mean
%                   in the test data set.

% =========================================================================
% Bump hunting using Patient Rule Induction Method for Matlab/Octave
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2015  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

% Citing the toolbox:
% Jekabsons G., Bump hunting using Patient Rule Induction Method for
% Matlab/Octave, 2015, available at http://www.cs.rtu.lv/jekabsons/

% Last update: November 9, 2015

if nargin < 3
    error('Not enough input arguments.');
end

if isempty(X) || isempty(Y)
    error('X,Y is empty.');
end
[n, m] = size(X);
if size(Y,1) ~= n
    error('The number of rows in X and Y should be equal.');
end
if size(Y,2) ~= 1
    error('Y should have one column.');
end

params.alphaPeel = max(0, params.alphaPeel);
params.alphaPaste = max(0, params.alphaPaste);
params.maxBoxes = max(1, params.maxBoxes);

params.minSupport = max(0, params.minSupport);
if params.minSupport <= 1
    minSupportN = max(1, round(n * params.minSupport));
else
    minSupportN = round(params.minSupport);
end

if (nargin < 4) || isempty(isBinCat)
    isBinCat = false(1,m);
else
    isBinCat = isBinCat(:)'; % force row vector
    if length(isBinCat) ~= m
        error('The number of elements in isBinCat should be equal to the number of columns in X.');
    end
end
if (nargin < 6) || isempty(Xt) || isempty(Yt)
    Xt = [];
    Yt = [];
    testDataSupplied = false;
else
    params.alphaPaste = 0;
    testDataSupplied = true;
    Yt = Yt(:);
    if size(Yt,1) ~= size(Xt,1)
        error('The number of rows in Xt and Yt should be equal.');
    end
    if size(Yt,2) ~= 1
        error('Yt should have one column.');
    end
    if size(X,2) ~= size(Xt,2)
        error('The number of columns in X and Xt should be equal.');
    end
end

prim.params = params;

catVals = cell(1,m);
hasNan = false(1,m);
for k = 1 : m
    if isBinCat(k)
        XX = X(:,k);
        nans = isnan(XX);
        if any(nans)
            u = unique(XX(~nans));
            hasNan(k) = true;
        else
            u = unique(XX);
        end
        catVals{k} = u;
        if size(u,1) >= 50
            disp(['Warning: Categorical variable #' num2str(k) ' has ' num2str(size(u,1)) ' unique values.']);
        end
    else
        hasNan(k) = any(isnan(X(:,k)));
    end
end
prim.info.isBinCat = isBinCat;
prim.info.catVals = catVals;
prim.info.hasNan = hasNan;
prim.info.minVals = min(X);
prim.info.maxVals = max(X);
prim.info.meanGlobal = mean(Y);
prim.info.n = n;

prim.boxes = {};
trajectory = [];
totalN = n;
if testDataSupplied
    totalNTest = size(Xt,1);
    % Values previously unseen in the training data are interpreted as unknown in test data
    for k = 1 : m
        if isBinCat(k)
            which = ~ismember(Xt(:,k), catVals{k});
            Xt(which,k) = NaN;
        end
    end
end

while true % loop for adding boxes
    n = size(Y,1);
    if testDataSupplied
        nTest = size(Yt,1);
    end
    
    % default box
    box.vars = [];
    box.supportN = n;
    box.support = n / totalN;
    if testDataSupplied
        box.supportNTest = nTest;
        box.supportTest = nTest / totalNTest;
    end
    box.mean = mean(Y);
    if testDataSupplied
        box.meanTest = mean(Yt);
    end
    box.min = [];
    box.max = [];
    box.cats = [];
    box.nan = [];
    if length(prim.boxes) >= params.maxBoxes
        prim.boxes{end+1,1} = box;
        break;
    end
    
    Xfull = X;
    Yfull = Y;
    if testDataSupplied
        XfullT = Xt;
        YfullT = Yt;
    end
    catValsfull = catVals;
    hasNanfull = hasNan;
    
    % PEEL
    varNum = 0;
    boxvars = [];
    boxmin = [];
    boxmax = [];
    boxcats = {};
    boxnan = [];
    while true % loop for finding one "best" box to add next
        boxsupportSAVE = 0;
        boxmeanSAVE = -realmax;
        for k = 1 : m
            if ~isBinCat(k)
                % Try peeling the "left side"
                boxminNEW = prctile(X(:,k), 100 * params.alphaPeel);
                if hasNan(k)
                    YY = Y((X(:,k) >= boxminNEW) | isnan(X(:,k)));
                else
                    YY = Y(X(:,k) >= boxminNEW);
                end
                boxsupportNEW = size(YY,1);
                if boxsupportNEW == size(X,1) % workaround for when too many values are identical (allow bigger steps)
                    val = min(X(:,k));
                    boxminNEW = min(X(X(:,k) > val,k)); % take the second smallest value
                    if boxminNEW ~= max(X(:,k))
                        %if ~isempty(boxminNEW)
                        if hasNan(k)
                            YY = Y((X(:,k) >= boxminNEW) | isnan(X(:,k)));
                        else
                            YY = Y(X(:,k) >= boxminNEW);
                        end
                        boxsupportNEW = size(YY,1);
                    end
                end
                % Check if mean increased and save the box if it did
                % but do it only if the box size decreased and is still not
                % lower than minSupportN.
                if (boxsupportNEW < size(X,1)) && (boxsupportNEW >= minSupportN) % ja variants sanaak mazaaks par minSupport, tad neapskatam
                    boxmeanNEW = mean(YY);
                    if (boxmeanNEW > boxmeanSAVE) || ...
                            ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE)) % break ties by choosing box with largest support
                        kSAVE = k;
                        boxsupportSAVE = boxsupportNEW;
                        boxmeanSAVE = boxmeanNEW;
                        boxminSAVE = boxminNEW;
                        boxmaxSAVE = [];
                        boxcatsSAVE = [];
                        boxnanSAVE = [];
                    end
                end
                % Try peeling the "right side"
                boxmaxNEW = prctile(X(:,k), 100 * (1 - params.alphaPeel));
                if hasNan(k)
                    YY = Y((X(:,k) <= boxmaxNEW) | isnan(X(:,k)));
                else
                    YY = Y(X(:,k) <= boxmaxNEW);
                end
                boxsupportNEW = size(YY,1);
                if boxsupportNEW == size(X,1) % workaround for when too many values are identical (allow bigger steps)
                    val = max(X(:,k));
                    boxmaxNEW = max(X(X(:,k) < val,k)); % take the second largest value
                    if boxmaxNEW ~= min(X(:,k))
                        %if ~isempty(boxmaxNEW)
                        if hasNan(k)
                            YY = Y((X(:,k) <= boxmaxNEW) | isnan(X(:,k)));
                        else
                            YY = Y(X(:,k) <= boxmaxNEW);
                        end
                        boxsupportNEW = size(YY,1);
                    end
                end
                % Check if mean increased and save the box if it did
                % but do it only if the box size decreased and is still not
                % lower than minSupportN.
                if (boxsupportNEW < size(X,1)) && (boxsupportNEW >= minSupportN)
                    boxmeanNEW = mean(YY);
                    if (boxmeanNEW > boxmeanSAVE) || ...
                            ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE))
                        kSAVE = k;
                        boxsupportSAVE = boxsupportNEW;
                        boxmeanSAVE = boxmeanNEW;
                        boxminSAVE = [];
                        boxmaxSAVE = boxmaxNEW;
                        boxcatsSAVE = [];
                        boxnanSAVE = [];
                    end
                end
            else
                % Consider removing a category if there are more than one
                % or if the data has NaNs and there are one category left.
                if (length(catVals{k}) > 1) || (hasNan(k) && (length(catVals{k}) == 1))
                    for c = catVals{k}'
                        YY = Y(X(:,k) ~= c);
                        % Check if mean increased and save the box if it did
                        % but do it only if the box size decreased and is still not
                        % lower than minSupportN.
                        boxsupportNEW = size(YY,1);
                        if (boxsupportNEW < size(X,1)) && (boxsupportNEW >= minSupportN)
                            boxmeanNEW = mean(YY);
                            if (boxmeanNEW > boxmeanSAVE) || ...
                                    ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE))
                                kSAVE = k;
                                boxsupportSAVE = boxsupportNEW;
                                boxmeanSAVE = boxmeanNEW;
                                boxminSAVE = [];
                                boxmaxSAVE = [];
                                boxcatsSAVE = c; % category to exclude
                                boxnanSAVE = [];
                            end
                        end
                    end
                end
            end
            
            % Consider removing NaNs for continuous variables or for
            % categorical variables if there are at least one category left
            if hasNan(k) && ((~isBinCat(k)) || (~isempty(catVals{k})))
                YY = Y(~isnan(X(:,k)));
                % Check if mean increased and save the box if it did
                % but do it only if the box size is still not lower than minSupportN.
                boxsupportNEW = size(YY,1);
                if (boxsupportNEW >= minSupportN)
                    boxmeanNEW = mean(YY);
                    if (boxmeanNEW > boxmeanSAVE) || ...
                            ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE))
                        kSAVE = k;
                        boxsupportSAVE = boxsupportNEW;
                        boxmeanSAVE = boxmeanNEW;
                        boxminSAVE = [];
                        boxmaxSAVE = [];
                        boxcatsSAVE = [];
                        boxnanSAVE = false;
                    end
                end
            end
        end % end of for k = 1 : m
        
        % If successfuly "peeled" the box
        if boxsupportSAVE > 0
            if varNum > 0
                boxvarsExists = boxvars == kSAVE; % box already has this variable defined
            end
            if (varNum > 0) && any(boxvarsExists)
                boundIdx = find(boxvarsExists, 1); % box already has this variable defined
            else
                % we are introducing a new variable to the definition of the box
                varNum = varNum + 1;
                boundIdx = varNum;
                boxmin(boundIdx) = NaN;
                boxmax(boundIdx) = NaN;
                boxcats{boundIdx} = catVals{kSAVE};
                boxnan(boundIdx) = hasNan(kSAVE);
            end
            boxvars(boundIdx) = kSAVE;
            if ~isempty(boxminSAVE)
                % boxminSAVE contains the new minimum for the variable
                boxmin(boundIdx) = boxminSAVE;
                if hasNan(kSAVE)
                    which = (X(:,kSAVE) >= boxminSAVE) | isnan(X(:,kSAVE));
                    if testDataSupplied
                        whichT = (Xt(:,kSAVE) >= boxminSAVE) | isnan(Xt(:,kSAVE));
                    end
                else
                    which = X(:,kSAVE) >= boxminSAVE;
                    if testDataSupplied
                        whichT = Xt(:,kSAVE) >= boxminSAVE;
                    end
                end
            elseif ~isempty(boxmaxSAVE)
                % boxminSAVE contains the new maximum for the variable
                boxmax(boundIdx) = boxmaxSAVE;
                if hasNan(kSAVE)
                    which = (X(:,kSAVE) <= boxmaxSAVE) | isnan(X(:,kSAVE));
                    if testDataSupplied
                        whichT = (Xt(:,kSAVE) <= boxmaxSAVE) | isnan(Xt(:,kSAVE));
                    end
                else
                    which = X(:,kSAVE) <= boxmaxSAVE;
                    if testDataSupplied
                        whichT = Xt(:,kSAVE) <= boxmaxSAVE;
                    end
                end
            elseif ~isempty(boxcatsSAVE)
                % boxcatsSAVE contains category to exclude
                catValsK = catVals{kSAVE};
                boxcats{boundIdx} = catValsK(catValsK ~= boxcatsSAVE);
                catVals{kSAVE} = boxcats{boundIdx};
                which = X(:,kSAVE) ~= boxcatsSAVE;
                if testDataSupplied
                    whichT = Xt(:,kSAVE) ~= boxcatsSAVE;
                end
            elseif ~isempty(boxnanSAVE)
                % non-empty boxnanSAVE is always false (excludes NaNs)
                boxnan(boundIdx) = boxnanSAVE;
                which = ~isnan(X(:,kSAVE));
                if testDataSupplied
                    whichT = ~isnan(Xt(:,kSAVE));
                end
                hasNan(kSAVE) = false;
            end
            % Update X,Y according to the reduced box
            X = X(which,:);
            Y = Y(which);
            if testDataSupplied
                Xt = Xt(whichT,:);
                Yt = Yt(whichT);
                boxmeanSAVETEST = mean(Yt); % for empty Yt, result is NaN
            end
            
            % Here we save the overall best box (the one with the smallest mean)
            % This is the actual structure to be added to the list of boxes
            if ((~testDataSupplied) && (boxmeanSAVE > box.mean)) || ...
                    (testDataSupplied && (boxmeanSAVETEST > box.meanTest))
                box.vars = boxvars;
                box.supportN = boxsupportSAVE;
                box.support = boxsupportSAVE / totalN;
                if testDataSupplied
                    box.supportNTest = size(Yt,1);
                    box.supportTest = size(Yt,1) / totalNTest;
                end
                box.mean = boxmeanSAVE;
                if testDataSupplied
                    box.meanTest = boxmeanSAVETEST;
                end
                box.min = boxmin;
                box.max = boxmax;
                box.cats = boxcats;
                box.nan = boxnan;
                % Shrink the box so that all its edges have at least one data point on them
                % This does not change the box's actual definition,
                % just sometimes shrinks it so that it's not bigger than
                % the data it's covering.
                for i = 1 : length(box.vars)
                    k = box.vars(i);
                    if ~isBinCat(k)
                        val = min(X(:,k));
                        if (val > prim.info.minVals(k)) && ...
                                (isnan(box.min(i)) || (val > box.min(i)))
                            box.min(i) = val;
                        end
                        val = max(X(:,k));
                        if (val < prim.info.maxVals(k)) && ...
                                (isnan(box.max(i)) || (val < box.max(i)))
                            box.max(i) = val;
                        end
                    end
                end
            end
            
            % For the very first box, save the "peeling trajectory"
            if isempty(prim.boxes)
                trajectory(end+1,1) = boxsupportSAVE / totalN;
                trajectory(end,2) = boxmeanSAVE;
                if testDataSupplied
                    trajectory(end,3) = size(Yt,1) / totalNTest;
                    trajectory(end,4) = boxmeanSAVETEST;
                end
            end
            
            if boxsupportSAVE <= minSupportN
                break; % we reached smallest allowed support (minSupportN)
            end
        else
            break; % we reached smallest allowed support (minSupportN)
        end
    end
    
    % If the box is still the default one, terminate
    if isempty(box.vars)
        prim.boxes{end+1,1} = box; % add the box to the list
        break;
    end
    
    X = Xfull;
    Y = Yfull;
    if testDataSupplied
        Xt = XfullT;
        Yt = YfullT;
    end
    catVals = catValsfull;
    hasNan = hasNanfull;
    
    % Remove the data covered by the new box
    which = false(size(X,1),1);
    if testDataSupplied
        whichT = false(size(Xt,1),1);
    end
    for i = 1 : length(box.vars)
        k = box.vars(i);
        % Continuous variables
        if (~isnan(box.min(i))) || (~isnan(box.max(i)))
            if box.nan(i)
                which = which | (X(:,k) < box.min(i)) | (X(:,k) > box.max(i));
                if testDataSupplied
                    whichT = whichT | (Xt(:,k) < box.min(i)) | (Xt(:,k) > box.max(i));
                end
            else
                which = which | (X(:,k) < box.min(i)) | (X(:,k) > box.max(i)) | isnan(X(:,k));
                if testDataSupplied
                    whichT = whichT | (Xt(:,k) < box.min(i)) | (Xt(:,k) > box.max(i)) | isnan(Xt(:,k));
                end
            end
        end
        % Categorical variables
        if isBinCat(k)
            catsNop = setdiff(catVals{k}, box.cats{i});
            if box.nan(i)
                which = which | ismember(X(:,k), catsNop);
                if testDataSupplied
                    whichT = whichT | ismember(Xt(:,k), catsNop);
                end
            else
                which = which | ismember(X(:,k), catsNop) | isnan(X(:,k));
                if testDataSupplied
                    whichT = whichT | ismember(Xt(:,k), catsNop) | isnan(Xt(:,k));
                end
            end
        end
    end
    % The inside of the box
    if params.alphaPaste > 0
        Ybox = Y(~which);
    end
    % The outside of the box
    X = X(which,:);
    Y = Y(which);
    if testDataSupplied
        Xt = Xt(whichT,:);
        Yt = Yt(whichT);
    end
    
    % PASTE
    if params.alphaPaste > 0
        while true % loop as long as the one box can be increased in size
            boxsupportSAVE = box.supportN;
            boxmeanSAVE = box.mean;
            for i = 1 : length(box.vars) % loop through all variables that define the box
                k = box.vars(i);
                if ~isBinCat(k)
                    numAdd = max(1, round(size(X,1) * params.alphaPaste));
                    
                    % Try "pasting" one the "left side"
                    if ~isnan(box.min(i))
                        [XX, YY, idxXY] = createSubbox(X, Y, box, i, true, numAdd);
                        if ~isempty(YY)
                            YY = [YY; Ybox];
                            % Check if mean increased and save the box if it did
                            boxsupportNEW = size(YY,1);
                            boxmeanNEW = mean(YY);
                            if (boxmeanNEW > boxmeanSAVE) || ...
                                    ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE)) % break ties by choosing box with largest support
                                kSAVE = k;
                                boxsupportSAVE = boxsupportNEW;
                                boxmeanSAVE = boxmeanNEW;
                                boxminSAVE = min(XX(:,k));
                                boxmaxSAVE = [];
                                boxcatsSAVE = [];
                                boxnanSAVE = [];
                                boxIdxXY = idxXY;
                            end
                        end
                    end
                    
                    % Try "pasting" one the "right side"
                    if ~isnan(box.max(i))
                        [XX, YY, idxXY] = createSubbox(X, Y, box, i, false, numAdd);
                        if ~isempty(YY)
                            YY = [YY; Ybox];
                            % Check if mean increased and save the box if it did
                            boxsupportNEW = size(YY,1);
                            boxmeanNEW = mean(YY);
                            if (boxmeanNEW > boxmeanSAVE) || ...
                                    ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE))
                                kSAVE = k;
                                boxsupportSAVE = boxsupportNEW;
                                boxmeanSAVE = boxmeanNEW;
                                boxminSAVE = [];
                                boxmaxSAVE = max(XX(:,k));
                                boxcatsSAVE = [];
                                boxnanSAVE = [];
                                boxIdxXY = idxXY;
                            end
                        end
                    end
                else
                    % Try adding a category
                    if ~isempty(catVals{k})
                        for c = catVals{k}'
                            [YY, idxXY] = createSubboxCat(X, Y, box, isBinCat, i, c);
                            if ~isempty(YY)
                                YY = [YY; Ybox];
                                % Check if mean increased and save the box if it did
                                boxsupportNEW = size(YY,1);
                                boxmeanNEW = mean(YY);
                                if (boxmeanNEW > boxmeanSAVE) || ...
                                        ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE))
                                    kSAVE = k;
                                    boxsupportSAVE = boxsupportNEW;
                                    boxmeanSAVE = boxmeanNEW;
                                    boxminSAVE = [];
                                    boxmaxSAVE = [];
                                    boxcatsSAVE = [box.cats{k}; c]; % all categories for the box
                                    boxnanSAVE = [];
                                    boxIdxXY = idxXY;
                                end
                            end
                        end
                    end
                end
                % Try adding NaNs
                if hasNan(k)
                    [YY, idxXY] = createSubboxNaN(X, Y, box, isBinCat, i);
                    if ~isempty(YY)
                        YY = [YY; Ybox];
                        % Check if mean increased and save the box if it did
                        boxsupportNEW = size(YY,1);
                        boxmeanNEW = mean(YY);
                        if (boxmeanNEW > boxmeanSAVE) || ...
                                ((boxmeanNEW == boxmeanSAVE) && (boxsupportNEW > boxsupportSAVE))
                            kSAVE = k;
                            boxsupportSAVE = boxsupportNEW;
                            boxmeanSAVE = boxmeanNEW;
                            boxminSAVE = [];
                            boxmaxSAVE = [];
                            boxcatsSAVE = [];
                            boxnanSAVE = true;
                            boxIdxXY = idxXY;
                        end
                    end
                end
            end
            
            % If the box increased in size, modify the actual structure to be
            % added to the list of boxes.
            if boxsupportSAVE > box.supportN
                box.supportN = boxsupportSAVE;
                box.support = boxsupportSAVE / totalN;
                box.mean = boxmeanSAVE;
                boundIdx = find(box.vars == kSAVE, 1);
                if ~isempty(boxminSAVE)
                    box.min(boundIdx) = boxminSAVE;
                    Ybox = [Ybox; Y(boxIdxXY)];
                    X(boxIdxXY,:) = [];
                    Y(boxIdxXY) = [];
                elseif ~isempty(boxmaxSAVE)
                    box.max(boundIdx) = boxmaxSAVE;
                    Ybox = [Ybox; Y(boxIdxXY)];
                    X(boxIdxXY,:) = [];
                    Y(boxIdxXY) = [];
                elseif ~isempty(boxcatsSAVE)
                    box.cats{boundIdx} = boxcatsSAVE;
                    Ybox = [Ybox; Y(boxIdxXY)];
                    X(boxIdxXY,:) = [];
                    Y(boxIdxXY) = [];
                elseif ~isempty(boxnanSAVE)
                    % non-empty boxnanSAVE is always true (includes NaNs)
                    box.nan(boundIdx) = boxnanSAVE;
                    Ybox = [Ybox; Y(boxIdxXY)];
                    X(boxIdxXY,:) = [];
                    Y(boxIdxXY) = [];
                end
            else
                break; % terminate if could not increase the size of the box
            end
        end
    end
    
    % If the next box (the one after the current one) would be too small or
    % if the current box's mean is under global mean, just combine them
    % (i.e., just create the default 'leftovers' box) and terminate
    if (n - box.supportN < minSupportN) || ...
            (params.terminateBelowMean && (box.mean < prim.info.meanGlobal))
        box.vars = [];
        box.supportN = n;
        box.support = n / totalN;
        if testDataSupplied
            box.supportNTest = nTest;
            box.supportTest = nTest / totalNTest;
        end
        box.mean = mean(Yfull);
        if testDataSupplied
            box.meanTest = mean(YfullT);
        end
        box.min = [];
        box.max = [];
        box.cats = [];
        box.nan = [];
        prim.boxes{end+1,1} = box; % add the box to the list
        break;
    end
    
    % finally add the box to the list
    prim.boxes{end+1,1} = box;
end
end




function [X, Y, idxXY] = createSubbox(X, Y, box, varIdx, leftSide, takeNearestNum)
% Function creates a subbox (using continuous variables) for "pasting" to the current box
which = true(size(X,1),1);
for i = 1 : length(box.vars)
    k = box.vars(i);
    % Continuous variables
    if (~isnan(box.min(i))) && ((i ~= varIdx) || (~leftSide))
        if box.nan(i)
            which = which & ((X(:,k) >= box.min(i)) | isnan(X(:,k)));
        else
            which = which & (X(:,k) >= box.min(i));
        end
    end
    if (~isnan(box.max(i))) && ((i ~= varIdx) || leftSide)
        if box.nan(i)
            which = which & ((X(:,k) <= box.max(i)) | isnan(X(:,k)));
        else
            which = which & (X(:,k) <= box.max(i));
        end
    end
    % Categorical variables
    if ~isempty(box.cats{i})
        if box.nan(i)
            which = which & (ismember(X(:,k), box.cats{i}) | isnan(X(:,k)));
        else
            which = which & ismember(X(:,k), box.cats{i});
        end
    end
end
idxXY = 1:size(X,1);
idxXY = idxXY(which);
X = X(which,:);
Y = Y(which);
% If we need less than available, sort the values and take only the ones
% neareast to the existing box.
if (takeNearestNum > 0) && (takeNearestNum < size(X,1))
    if leftSide
        [Xsorted, idx] = sort(X(:,box.vars(varIdx)), 1, 'descend');
    else
        [Xsorted, idx] = sort(X(:,box.vars(varIdx)));
    end
    % Check if some values are identical. Take all of those.
    while takeNearestNum < length(idx)
        if Xsorted(takeNearestNum) == Xsorted(takeNearestNum+1)
            takeNearestNum = takeNearestNum + 1;
        else
            break;
        end
    end
    idx = idx(1:takeNearestNum);
    idxXY = idxXY(idx);
    X = X(idx,:);
    Y = Y(idx);
end
end

function [Y, idxXY] = createSubboxCat(X, Y, box, isBinCat, varIdx, c)
% Function creates a subbox (using categorical variables) for "pasting" to the current box
which = true(size(X,1),1);
for i = 1 : length(box.vars)
    k = box.vars(i);
    % Continuous variables
    if ~isnan(box.min(i))
        if box.nan(i)
            which = which & ((X(:,k) >= box.min(i)) | isnan(X(:,k)));
        else
            which = which & (X(:,k) >= box.min(i));
        end
    end
    if ~isnan(box.max(i))
        if box.nan(i)
            which = which & ((X(:,k) <= box.max(i)) | isnan(X(:,k)));
        else
            which = which & (X(:,k) <= box.max(i));
        end
    end
    % Categorical variables
    if isBinCat(k)
        if (i == varIdx)
            if box.nan(i)
                which = which & (ismember(X(:,k), [box.cats{i}; c]) | isnan(X(:,k)));
            else
                which = which & ismember(X(:,k), [box.cats{i}; c]);
            end
        else
            if box.nan(i)
                which = which & (ismember(X(:,k), box.cats{i}) | isnan(X(:,k)));
            else
                which = which & ismember(X(:,k), box.cats{i});
            end
        end
    end
end
idxXY = 1:size(X,1);
idxXY = idxXY(which);
Y = Y(which);
end

function [Y, idxXY] = createSubboxNaN(X, Y, box, isBinCat, varIdx)
% Function creates a subbox (using NaNs) for "pasting" to the current box
which = true(size(X,1),1);
for i = 1 : length(box.vars)
    k = box.vars(i);
    % Continuous variables
    if ~isnan(box.min(i))
        if box.nan(i) || (i == varIdx)
            which = which & ((X(:,k) >= box.min(i)) | isnan(X(:,k)));
        else
            which = which & (X(:,k) >= box.min(i));
        end
    end
    if ~isnan(box.max(i))
        if box.nan(i) || (i == varIdx)
            which = which & ((X(:,k) <= box.max(i)) | isnan(X(:,k)));
        else
            which = which & (X(:,k) <= box.max(i));
        end
    end
    % Categorical variables
    if isBinCat(k)
        if box.nan(i) || (i == varIdx)
            which = which & (ismember(X(:,k), box.cats{i}) | isnan(X(:,k)));
        else
            which = which & ismember(X(:,k), box.cats{i});
        end
    end
end
idxXY = 1:size(X,1);
idxXY = idxXY(which);
Y = Y(which);
end