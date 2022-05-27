function [h,k,error] = heat_CN_nrbc(m)
%
% heat_CN_nrbc.m
% **This code uses legpts in the Chebfun package**
%
% Solve u_t = u_{xx} on [ax,bx] with NRBC,
% using the Crank-Nicolson method with m interior points.
% This is a modification of the code heat_CN.m
% from  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)

clf              % clear graphics
hold on          % Put all plots on the same graph (comment out if desired)

ax = 0;
bx = 1;
kappa = 1;                 % heat conduction coefficient = 1
tfinal = 1;                % final time T

h = (bx-ax)/(m+1);         % h = delta x
x = linspace(ax,bx,m+2)';  % note x(1)=0 and x(m+2)=1
                           % u(1)=g0 and u(m+2)=g1 are known from BC's
k = h;                  % time step

nsteps = round(tfinal / k);    % number of time steps
% nplot = 100;      % plot solution every nplot time steps
                 % (set nplot=2 to plot every 2 time steps, etc.)
nplot = nsteps;  % only plot at final time

% true solution for comparison:
% For Gaussian initial conditions u(x,0) = exp(-(x-0.5)^2 /(4*t0))
t0 = 0.001;
utrue = @(x,t) exp(-(x-0.5).^2 / (4*(t+t0))) / sqrt(4*pi*(t+t0));
% initial conditions:
u0 = utrue(x,0);


% Each time step we solve MOL system U' = AU + g using the Trapezoidal method

%% set up matrices: **(Incorporate I_local here)**
% use second-order, one-sided approximation for u_x at bdy 
r = (1/2) * kappa* k/(h^2);
q = sqrt(k/pi)/3/h; %factor in front of I_local 

e = ones(m+2,1);
A = spdiags([e -2*e e], [-1 0 1], m+2, m+2);
A1 = eye(m+2) - r * A;
A1(1,:) = [1+6*q, -8*q, 2*q, zeros(1, m-1)]; %bdy condition is also implicit
A1(m+2,:) = [zeros(1, m-1), 2*q, -8*q, 1+6*q];
A2 = eye(m+2) + r * A;
A2(1,:) = [-3*q, 4*q, -q, zeros(1, m-1)];
A2(m+2,:) = [zeros(1, m-1), -3*q, 4*q, -q];


% initial data on fine grid for plotting:
xfine = linspace(ax,bx,1001);
ufine = utrue(xfine,0);

% initialize u and plot:
tn = 0;
u = u0;

plot(x,u,'k.-', xfine,ufine,'k')
legend('computed','true')
title('Initial data at time = 0')

input('Hit <return> to continue  ');


%% initialize I_history calculation

% I_history at initial time 
% in the first time-step, the integral is from 0 to 0
Ihist0 = 0 ; Ihist1 = 0; 

% x derivative at each bdy (initial data)
ux0 = (-3*u(1)+4*u(2)-u(3))/(2*h);
ux1 = (3*u(m+2)-4*u(m+1)+u(m))/(2*h);

% setup for sum of exponentials nodes and weights
epsilon = 10^(-15); %This gives machine precision
n0 = ceil(0.565*log10(10/epsilon));
Lmin = floor(log2(sqrt(n0/tfinal)));
Lmax = floor(log2(sqrt(log10(1/epsilon)/k)));
n1 = ceil(1/3*log2(12*(Lmax-Lmin+1)/epsilon));
% Gaussian quadrature nodes and weights for first interval
[s0, w0] = legpts(n0, [0, 2^(Lmin)]);
% initialize C_j(n)
% first interval
C0 = zeros(1,n0); C1 = zeros(1, n0);
% dyadic sub-intervals
% we have c dyadic sub-intervals
c = Lmax-Lmin+1;
C00 = zeros(c,n1); C11 = zeros(c,n1);

%% main time-stepping loop:

for n = 1:nsteps
     tnp = tn + k;   % = t_{n+1}
     
     % compute right hand side for linear system:
     rhs = A2*u;
     % fix-up right hand side by adding I_history
     rhs(1) = rhs(1) + Ihist0;
     rhs(m+2) = rhs(m+2) - Ihist1;  

     % solve linear system:
     u = A1\rhs;
     
     % plot results at desired times:
     if mod(n,nplot)==0 | n==nsteps
        ufine = utrue(xfine,tnp);
        plot(x,u,'b.-', xfine,ufine,'r')
        title(sprintf('t = %9.5e  after %4i time steps with %5i grid points',...
                       tnp,n,m+2))      
        error = max(abs(u-utrue(x,tnp)));
        disp(sprintf('at time t = %9.5e  max error =  %9.5e',tnp,error))
        if n<nsteps, input('Hit <return> to continue  '); end;
        end

     tn = tnp;   % for next time step
     
     %% I_history calculation for next time step
     
    % the value of u_x(0, (n-2)k), u_x(1, (n-2)k) is required
    ux0old = ux0; 
    ux1old = ux1;
     
    %u_x(0, (n-1)k), u_x(1, (n-1)k) : second-order, one-sided approximation
    ux0 = (-3*u(1)+4*u(2)-u(3))/(2*h);
    ux1 = (3*u(m+2)-4*u(m+1)+u(m))/(2*h);
    
    %recursive formula for C_j(n)
    %first interval
    for j = 1 : n0
        C0(j) = exp(-s0(j)^2*k)*C0(j) ...
            +(ux0old-(n-1)*(ux0-ux0old))*(exp(-s0(j)^2*k)...
            -exp(-s0(j)^2*2*k))/(s0(j)^2) ...
            +1/k*(ux0-ux0old)*((n*k*exp(-s0(j)^2*k)...
            -(n-1)*k*exp(-s0(j)^2*2*k))/(s0(j)^2)...
            -(exp(-s0(j)^2*k)-exp(-s0(j)^2*2*k))/(s0(j)^4));
        C1(j) = exp(-s0(j)^2*k)*C1(j) ...
            +(ux1old-(n-1)*(ux1-ux1old))*(exp(-s0(j)^2*k)...
            -exp(-s0(j)^2*2*k))/(s0(j)^2) ...
            +1/k*(ux1-ux1old)*((n*k*exp(-s0(j)^2*k)...
            -(n-1)*k*exp(-s0(j)^2*2*k))/(s0(j)^2)...
            -(exp(-s0(j)^2*k)-exp(-s0(j)^2*2*k))/(s0(j)^4));
    end
    %multiply 2/pi to weights to satisfy quadrature rule
    Ihist0 = dot(2/pi*w0,C0); 
    Ihist1 = dot(2/pi*w0,C1);
    
    %dyadic intervals
    for p = Lmin:Lmax
        [s, w] = legpts(n1, [2^p, 2^(p+1)]);
        for j = 1 : n1        
            C00(p-Lmin+1,j) = exp(-s(j)^2*k)*C00(p-Lmin+1,j) ...
            +(ux0old-(n-1)*(ux0-ux0old))*(exp(-s(j)^2*k)...
            -exp(-s(j)^2*2*k))/(s(j)^2) ...
            +1/k*(ux0-ux0old)*((n*k*exp(-s(j)^2*k)...
            -(n-1)*k*exp(-s(j)^2*2*k))/(s(j)^2)...
            -(exp(-s(j)^2*k)-exp(-s(j)^2*2*k))/(s(j)^4));
        
            C11(p-Lmin+1,j) = exp(-s(j)^2*k)*C11(p-Lmin+1,j) ...
            +(ux1old-(n-1)*(ux1-ux1old))*(exp(-s(j)^2*k)...
            -exp(-s(j)^2*2*k))/(s(j)^2) ...
            +1/k*(ux1-ux1old)*((n*k*exp(-s(j)^2*k)...
            -(n-1)*k*exp(-s(j)^2*2*k))/(s(j)^2)...
            -(exp(-s(j)^2*k)-exp(-s(j)^2*2*k))/(s(j)^4));
        end
        Ihist0 = Ihist0 + dot(2/pi*w, C00(p-Lmin+1,:));
        Ihist1 = Ihist1 + dot(2/pi*w, C11(p-Lmin+1,:));
    end
end