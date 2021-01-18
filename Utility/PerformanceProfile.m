%% Performance Profile Plot
function PerformanceProfile(Data1,Algos,varargin)
if ~isempty(varargin)
    M = varargin{1};
else
    M = [];
end

%Handling NaN Values
if sum(sum(isnan(Data1),1),2)>0
    for jj = 1:size(Data1,2)
        if sum(isnan(Data1(:,jj)))==size(Data1,1)
            Data1(:,jj) = Data1(:,jj);
        else
            minvalue = min(Data1(:,jj));
            epsilon = abs(minvalue./100);
            id = find(isnan(Data1(:,jj)));
            Data1(id,jj) = max(Data1(:,jj)) + epsilon;
        end
    end
else
    Data1 = Data1;
end

%Normalizing Data to ensure no negative values
for ii = 1:size(Data1,2)
    eps = 1e-3;%eps = abs(min(Data1(:,ii))./1.e6);
    if min(Data1(:,ii)) == max(Data1(:,ii))
        Data1(:,ii) = ones(size(Data1,1),1);
    else
        Data1(:,ii) = eps + (Data1(:,ii) - min(Data1(:,ii))) * (1-eps) / (max(Data1(:,ii)) - min(Data1(:,ii)));
    end
    %         x_normal(:,i) = x(:,i);
end
% rows-> algos columns-> problems
rows = size(Data1,1);
% cols = size(Data,2);

a = Data1./repmat(min(Data1,[],1),rows,1);

linestyles = cellstr(char('-',':','-.','--','-',':','-.','--','-',':','-',':',...
    '-.','--','-',':','-.','--','-',':','-.','-',':','-.','--','-',':','-.','--'));
MarkerEdgeColors=colormap('lines');

if ~isempty(M)
    mSize = 5*ones(1,10);%[9 11 10];
else
    mSize = 5*ones(1,10);%[7 9 8];
end
for i = 1:rows
    A = a(i,:);
    [tmpf,tmpv,~,~] = ecdf(A);
    f = tmpf;
    v = tmpv;
    if i == 1
        v_max = max(v);
    else
        if max(v) > v_max
            v_max = max(v);
        end
    end
    %stairs((v),(f),'DisplayName',strcat('algo',num2str(Algos(i))),'color',MarkerEdgeColors(i,:),'linestyle',linestyles{i},'LineWidth',2); hold on;grid on;
    stairs((v),(f),'DisplayName',Algos{i},'color',MarkerEdgeColors(i,:),'linestyle',linestyles{i},'LineWidth',2); hold on;grid on;
end
hold off;
set(gca, 'Xlim', [0 ceil(v_max)]);%'XScale', 'log' ,

legend('Location','southeast');
legend('show')
% xlabel('Total FEs/Total FEs_{Best}');
xlabel('\tau');
ylabel('\rho_s(\tau)');
set(gca,'FontSize',16);
end