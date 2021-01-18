function[f,g] = Welded_beam_MO(x)
if nargin == 0
	prob.nx = 4;
	prob.nf = 2;
	prob.ng = 4;
	prob.bounds(1,:) = [0.125,5.0];
	prob.bounds(2,:) = [0.125,5.0];
	prob.bounds(3,:) = [0.1,10.0];
	prob.bounds(4,:) = [0.1,10.0];
	f = prob;
else
	[f,g] = welded_beam2_true(x);
end
return


function [obj,g] = welded_beam2_true(x)

h = x(:,1);
l = x(:,2);
t = x(:,3);
b = x(:,4);

P = 6000;	 		% lb
L = 14;				% in
E = 30e6;			% psi
G = 12e6;			% psi
tau_max = 13600;	% psi
sigma_max = 30000;	% psi
delta_max = 0.25;	% in

tau_1 = P ./ (sqrt(2)*h.*l);
M = P * (L + l/2);
R = sqrt(l.^2/4 + ((h+t)/2).^2);
J = 2 * (h.*l/sqrt(2) .* (l.^2/12 + ((h+t)/2).^2));
tau_2 = M .* R ./ J;
tau = sqrt(tau_1.^2 + 2*tau_1.*tau_2.*l./(2*R) + tau_2.^2);
sigma = 6*P*L ./ (b.*t.^2);
delta = 4*P*L^3 ./ (E*t.^3.*b);
Pc = 4.013*sqrt(E*G*t.^2.*b.^6/36)/L^2 .* (1 - t.*sqrt(E/(4*G))/(2*L));
obj(:,1) = 1.10471*h.^2.*l + 0.04811*t.*b.*(14.0+l);
obj(:,2) = delta;
g(:,1) = -(tau_max - tau);
g(:,2) = -(sigma_max - sigma);
g(:,3) = -(b - h);
g(:,4) = -(Pc - P);
return