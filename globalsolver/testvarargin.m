%%
% p{1} = 'one input';
% p{2} = '2 inputs';
testv( 'one input', '2 inputs');



function testv(varargin)
if length(varargin) > 1
    disp(varargin{2});
else
    disp(varargin{1});
end
end