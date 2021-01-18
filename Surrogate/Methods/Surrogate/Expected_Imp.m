function EI=Expected_Imp(F,SF,f_ref)
EI=[];
for i=1:size(F,1)
    u = (f_ref - F(i));
    z = u./SF(i);
    f1a = u.*normcdf(z,0,1);
    f1a(SF(i)==0,1) = 0;
    f1b = SF(i).*normpdf(z,0,1);
    EI= [EI;(f1a+f1b)];
end
return