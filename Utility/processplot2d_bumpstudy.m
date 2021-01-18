
function[] = processplot2d_bumpstudy(fighn, trainx, trainy, krg, prob, initx, before, daceflag)


clf(fighn);
% (1) create test data for 2d
lb1 = prob.xl_bl(1);
ub1 = prob.xl_bu(1);

lb2 = prob.xl_bl(2);
ub2 = prob.xl_bu(2);

num_points = 101;
x1 = linspace(lb1, ub1, num_points);
x2 = linspace(lb2, ub2, num_points);
[x1, x2] = meshgrid(x1, x2);
testdata = zeros(num_points, num_points);


for i = 1:num_points
    for j = 1:num_points
        [f(i, j), ~] = predict(krg{1}, [x1(i, j), x2(i, j)]);
        f(i, j) = denormzscore( trainy,  f(i, j));
    end
end

surfc(x1, x2, f); hold on;
% contour(x1, x2, f); hold on;
colormap parula
shading interp

scatter(trainx(:, 1), trainx(:, 2), 20,  'go', 'filled');

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
  
  
  % draw this box on plot
  xx = [bump_lb(1), bump_ub(1), fliplr([bump_lb(1), bump_ub(1)])];
  yy = [bump_lb(2), bump_lb(2), bump_ub(2), bump_ub(2)];
  
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