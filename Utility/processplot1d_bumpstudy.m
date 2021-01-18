function[] = processplot1d_bumpstudy(fighn, trainx, trainy, krg, prob, initx, before, daceflag)


clf(fighn);
% (1) create test data for 1d
testdata = linspace(prob.xl_bl, prob.xl_bu, 100);
testdata = testdata';

[freal, ~]= prob.evaluate_l([], testdata);
minf = min(freal);

% surfc(x1, x2, f); hold on;
plot(testdata, freal); hold on;

scatter(trainx, trainy, 20, 'ro', 'filled');


% plot bump hunting
prim =  bump_detection_4test(trainx, trainy, prob.xl_bl, prob.xl_bu);
nboxes = length(prim.boxes);
nx = size(trainx, 2);
fprintf('%d interesting boxes \n', nboxes);
% 
for ii = 1:nboxes
  region = prim.boxes{ii};
  
  fprintf('box %d, mean %f, support %d \n', ii, prim.boxes{ii}.mean, prim.boxes{ii}.supportN);
  % adjust boundary
  mentioned_var = prim.boxes{ii}.vars;
  bump_lb = prob.xl_bl;
  bump_ub = prob.xl_bu;
  
  nb = length(mentioned_var);
  
  if ~isempty(mentioned_var)
      for jj = 1:nb % varible indicator
          if ~isnan(prim.boxes{ii}.min(jj))
              bump_lb(mentioned_var(jj)) =  prim.boxes{ii}.min(jj);
          end
          
          if ~isnan(prim.boxes{ii}.max(jj))
              bump_ub(mentioned_var(jj)) =  prim.boxes{ii}.max(jj);
          end
          
      end
  else
      continue;
  end
  
  
  % find corresponding y
  bump_ly = prob.evaluate_l([], bump_lb);
  bump_uy = prob.evaluate_l([], bump_ub);
  
  if bump_ly > bump_uy
      tmp = bump_uy;
      bump_uy = bump_ly;
      bump_ly = tmp;
  end
   
  xx = [bump_lb, bump_ub,bump_ub, bump_lb];
  yy = [minf, minf, bump_uy, bump_uy];
  
  
  % 
  fill(xx, yy, 'k', 'FaceAlpha', 0.1, 'EdgeColor','k');
  pause(1);
end


end

function crosscheck(krg, trainx, trainy, daceflag)
if daceflag
    [yn, ~] = predictor(trainx, krg);
else
    [yn, ~] = predict(krg, trainx);
end

y =  denormzscore(trainy, yn);
b = max(abs(y - trainy)) - 0.01;
disp(b);
fprintf('stability check should be negative: %f\n',b);
% save('sig_dace', 'sig_dace');

a = unique(round(trainx, 3));
% size(a)
% size(trainx)
end