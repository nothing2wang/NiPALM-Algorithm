%       SNMF experiment on ORL data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
Data = load('ORL64');
A  = Data.fea;
A = A';
[n, d] = size(A);
% normalize the data as suggested
for i = 1:d
    A(:, i) = A(:, i)./norm(A(:, i)); 
end
y = A;
maxNumCompThreads(1);
[n, d] = size(y);
% minibatch subsampling ratio = 1/sr = 1/20 = 5%
sr = 20;
n_epochs = 250; % number of total epochs
tau = round(n/4);% sparsity constraint
% number of basis image to be extracted
r  = 25;

load('init_snmf_orl_s');

% NiPALM
[ Aout01, xt01, error01, time01 ] = SNMF_NiPALM(y,n_epochs, tau, r, Ain, xin);
%
bound = 7777;%2.3;
linewidth = 1;
axesFontSize = 6;
labelFontSize = 11;
legendFontSize = 8;
resolution = 108; % output resolution
output_size = resolution *[12, 12]; % output size
%%%%%% %%%%%% %%%%%% %%%%%% %%%%%%
figure(101), clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-1.525 -1.3125 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[2.65 2.195]);
im = eigenfaces(Aout01);
imagesc(abs(im)); % colormap gray
set(gca,'XTick',[], 'YTick', [])
% pdfname = sprintf('snmf_orl_s_eigenface.pdf');
% print(pdfname, '-dpdf');
%%%%%% %%%%%% %%%%%% %%%%%% %%%%%%
epochs = n_epochs;
figure(101), clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.15 0.6]);
p1 = plot(0:1:n_epochs, min(bound,log10(error01(1:end))), 'x-','LineWidth',1.5, 'color', [0,0,1], 'MarkerIndices', 1:epochs/5:epochs,'MarkerSize',10);
set(gca,'FontSize', 12);
grid on;
lg = legend(p1,'NiPALM');
legend('boxoff');
set(lg, 'Location', 'NorthEast');
set(lg, 'FontSize', 12);
ylb = ylabel({'$\mathrm{log}(\Psi(X_k, Y_k))$'},'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 16);
set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
xlb = xlabel({'$\#~ of~iterations$'}, 'FontSize', 14,'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);







