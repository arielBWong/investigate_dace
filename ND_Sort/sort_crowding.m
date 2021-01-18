

%% Crowding distance method
function [ranks, dist] = sort_crowding(f_all1, front_f)
f_all = f_all1;
L = length(front_f);
if L == 1
	ranks = front_f;
	dist = Inf;
else
	dist = zeros(L, 1);
	nf = size(f_all, 2);
	
	for i = 1:nf
		f = f_all(front_f, i);		% get ith objective
		[tmp, I] = sort(f);
		scale = f(I(L)) - f(I(1));
		dist(I(1)) = Inf;
		for j = 2:L-1
			id = I(j);
			id1 = front_f(I(j-1));
			id2 = front_f(I(j+1));
			if scale > 0
				dist(id) = dist(id) + (f_all(id2,i)-f_all(id1,i)) / scale;
			end
		end
	end
	dist = dist / nf;
	[tmp, I] = sort(dist, 'descend');
	ranks = front_f(I)';
end
return

