function [ Aout, xt, error, time] = SNMF_NiPALM(y,n_epochs, tau, r, Ain, xin)
% Implement generalized inertial PALM for sparse non-negative matrix factorization
%      argmin_{A,X} \|Y - AX\|_F^2 
%      s.t. \|A_k\|_0 <= tau \forall k, A_{i,j} >=0,  X_{i,j} >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n , d] =   size(y);
xi     =    zeros(r,d); 
error = zeros(n_epochs, 1);
pn = 5; % number of power iterations
% initialization
A = Ain;
xi = xin;
time = zeros(n_epochs, 1);
t_total = 0;
e0 = 0.5 * ( norm( A * xi - y ,'fro') )^2 ;
xi_a = xi;
A_a = A;
xi_b = xi;
A_b = A;
md = zeros(1,r);
for k = 1 : n_epochs
    tic;
    ss=1; 
    a = ss*(k-1)/(k+1); % inertial parameter
    b = ss*(k-1)/(k+1);
    pp=1; 
    xi_t = xi + a * (xi - xi_a);
    xi_a = xi;
    L_A = power_method(A_b, pn);
    u  = pp/L_A; 
    grad   =   A_b'*(A_b*xi_t - y); % gradient calculation
    xi     =   xi_t - u*(grad);
    xi(xi < 0) = 0;
    xi_b = xi + 0.7 * (xi - xi_b); %a
    L_x = power_method(xi_b, pn);
    uy = pp/L_x; 
    A_t = A + b * (A - A_a);
    A_a = A;
    A = A_t - uy * ((A_t*xi_b - y)*xi_b');
    B = sort(abs(A), 1, 'descend');
    md = B(tau,:);
    for q = 1:r % hard - tresholding
        A(:,q) = wthresh(A(:,q),'h',md(q));
    end
    A(A<0) = 0;
    A_b = A + 0.7 * (A - A_b); %b
    t1 = toc;
    t_total = t_total + t1;
    time(k) = t_total;
    error(k) = 0.5 * ( norm( A * xi - y ,'fro') )^2 ;
end
xt = xi; % output
Aout = A;
error = [e0; error];
end









