% plots = 1 for plots, 0 for no plots. 
% must also set d = 1 for plots
% clear
plots = 0;
%% Forming synthetic dataset

n = 1e+3;
d = 50;
X = zeros(n,d);
y = zeros(n,1);
beta = 0.1*ones(d,1);
beta(1:2) = 1;
sig = 1e-2; % std. error of noise term
sig2 = 1e-5; % std. error of correlation
X(:,1) = ones(n,1);
for i = 2:d
    rng(i)
    X(:,i) = 2*rand(1,n)-1;
%     X(i, 1) = X(i, 2) + sig2*randn(1);
end
y = X*beta + sig*randn(n,1);


%% LS via QR

[Q,R] = qr(X, 0);
beta = R\(Q'*y);


%% Modified Newton
maxit = 30; ftol = 1e-12;
eta = 1e-3; gamma = 1/2;
x1 = -2*ones(d,1);
x1(2) = 0.5;
x2 = x1;

%% Plotting contour
if plots == 1
    if d > 1
        error('Set d=1 for plots')
    end
    xp1 = linspace(-2.5,4,100);
    xp2 = linspace(-1,3,100);
    [xx1,xx2] = meshgrid(xp1,xp2);
    Z = zeros(100,100);
    for i = 1:100
        for j = 1:100
            xx = [xp1(i), xp2(j)];
            Z(i,j) = RSS(xx, X, y);
        end
    end
    contour(xx1, xx2, Z, [100, 400, 700, 1000, 1300, 1600, 1900])
    xlim([-2.5, 4])
    ylim([-1, 3])
    xN1 = x1(1); xN2 = x1(2);
    xR1 = x2(1); xR2 = x2(2);
end

%% setup
% f1 = RSS(x1, X, y);
% f2 = f1;
g1 = RSSGrad(x1, X, y);
g2 = g1;
H = RSSHess(X);
[U, D] = eig(H);
lam1 = max(H, [], 'all');
kappa = 100;
for i = 1:d
    if D(i,i) < lam1/kappa
        D(i,i) = lam1/kappa;
    end
end
% diag(D)
% rank(D)
for k = 1:maxit
    if norm(g2)<ftol
        break
    end
    %% pure newton
    p1 = -H\g1;
    alpha = 1;
    count = 0;
    while norm(RSSGrad(x1+alpha*p1, X, y))>(1-alpha*eta)*norm(RSSGrad(x1, X, y))
        alpha = alpha*gamma;
        count = count +1;
    end
    x1 = x1 + alpha*p1;
%     f1 = RSS(x1, X, y);
    g1 = RSSGrad(x1, X, y);
    if plots == 1
        xN1 = [xN1; x1(1)]; xN2 = [xN2 ; x1(2)];
    end
    %% Replace
    
    p2 = -U*(D\(U\g2));
    alpha = 1;
    while norm(RSSGrad(x2+alpha*p2, X, y))>(1-alpha*eta)*norm(RSSGrad(x2, X, y))
        alpha = alpha*gamma;
    end
    x2 = x2 + alpha*p2;
    f2 = RSS(x2, X, y);
    g2 = RSSGrad(x2, X, y);
    if plots == 1
        xR1 = [xR1; x2(1)]; xR2 = [xR2 ; x2(2)];
    end
end

if plots == 1
    hold on
    plot(xN1, xN2, 'b*-', 'MarkerSize', 8, 'LineWidth', 1)
    hold on
    plot(xR1, xR2, 'r.-', 'MarkerSize', 15, 'LineWidth', 1)
    legend('RSS', 'Pure', 'Replace')
    text(-2, 0.25, 'x_0')
end

for i = 1:d
    fprintf('%.6e & %.6e & %.6e %s\n', x2(i), x1(i), beta(i), '\\')
end

k
RSSGrad(x2, X, y)

%% RSS 
function f = RSS(beta, X, y)
[n,~] = size(X);
f = 0;
for i = 1:n
    f = f + 1/2*(y(i)-dot(beta, X(i,:)))^2;
end
end

%% RSSGrad
function g = RSSGrad(beta, X, y)
[n, d] = size(X);
g = zeros(d,1);
for i = 1:d
    for j = 1:n
        g(i) = g(i) - X(j, i)*(y(j)-dot(beta, X(j,:)));
    end
end
end

%% RSSHess
function H = RSSHess(X)
[n,d] = size(X);
H = zeros(d,d);
for i = 1:d
    for j = 1:d
        for k = 1:n
            H(i,j) = H(i,j) + X(k,j)*X(k,i);
        end
    end
end

end
