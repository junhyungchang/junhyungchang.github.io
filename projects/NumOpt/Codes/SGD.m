clear
nind1 = [2^10 2^12];
nind2 = 2^12;
tt = [];
numits = [];
OGap = [];
normg = [];
for n = nind2
    %% Forming synthetic dataset
    d = 100;
    X = zeros(n,d);
    y = zeros(n,1);
    beta = zeros(d,1);
    beta(1:2) = 1;
    sig = 1e-1; % std. dev. of noise term
    sig2 = 1e-5; % std. dev. of correlation
    X(:,1) = ones(n,1);
    rng('default')
    for i = 2:d
        X(:,i) = rand(1,n)-0.5;
    end
    y = X*beta + sig*randn(n,1);
    %% Setup 1
    tic
    maxit = 60; ftol = 1e-9;
    eta = 1e-3; gamma = 1/2;
    % x1 = 0.1*ones(d,1);
    x1 = zeros(d,1);
    % x1(1:2) = 1;
    g1 = RSSGrad(x1, X, y);
    p1 = ones(d,1);
    alpha = 1;
    m = 1000;
    toc


    %% Stochastic Gradient Descent
    tic
    for k = 1:maxit
        optgap = dot(g1, p1)^2/2;
        if n == 2^12
            OGap = [OGap, optgap];
            normg = [normg, norm(g1)];
        end
    %     if norm(g1)<ftol
        if optgap <= ftol
            break
        end
        index = randi(n,m,1);
        Xsamp = X(index, :);
        ysamp = y(index, :);
        g2 = RSSGrad(x1, Xsamp, ysamp);
        p1 = -g2;
        alpha = 2;
        count = 0;
        while norm(RSSGrad(x1+alpha*p1, X, y))>(1-alpha*eta)*norm(RSSGrad(x1, X, y))
            alpha = alpha*gamma;
            count = count + 1;
        end
        x1 = x1 + alpha*p1; 
    %     g1 = g2;
        g1 = RSSGrad(x1, X, y);
    end
    t = toc;
    tt = [tt , t];
    numits = [numits, k];
    if n == 2^12
        figure(1)
        hold on
        semilogy(1:k, OGap, 'c', 'LineWidth', 1.5, 'DisplayName', 'SGD')
        set(gca,'yscale','log')
        xlabel('Number of iterations')
        ylabel('Optimality Gap')
        figure(2)
        hold on
        semilogy(1:k, normg, 'c', 'LineWidth', 1.5, 'DisplayName', 'SGD')
        set(gca,'yscale','log')
        xlabel('Number of iterations')
        ylabel('norm of gradient')
    end
%     x1(1:10)
%     k
%     norm(g1)
%     optgap
end
% figure(3)
% hold on
% plot(nind1, numits, 'c', 'LineWidth', 1.5, 'DisplayName', 'SGD')
% xlabel('Size of n')
% ylabel('Iterations required')
% figure(4)
% hold on
% semilogy(nind1, tt, 'c', 'LineWidth', 1.5, 'DisplayName', 'SGD')
% set(gca,'yscale','log')
% xlabel('Size of n')
% ylabel('Run-time')
%% Subroutines
%% RSS Gradient evaluation
function g = RSSGrad(beta, X, y)
[n, d] = size(X);
g = zeros(d,1);
for i = 1:d
    for j = 1:n
        g(i) = g(i) - X(j, i)*(y(j)-dot(beta, X(j,:)));
    end
end
end