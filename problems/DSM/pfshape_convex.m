function p1 = pfshape_convex(xu, r)
 p1(:, 1) = (1+ r - cos(pi * xu(:, 1))) - r * cos(pi * xu(:, 1)) ;
 p1(:, 2) = (1+ r - sin(pi * xu(:, 1))) - r * sin( pi * xu(:, 1)) ;
end