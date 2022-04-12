clear
nind1 = [2^10 2^12 2^14 2^16 2^18];
nind2 = 2^12;
tt = [];
numits = [];
OGap = [];
normg = [];
for n = nind1
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
    maxit = 500; ftol = 1e-9;
    eta = 1e-3; gamma = 1/2;
    x1 = zeros(d,1);
    g1 = RSSGrad(x1, X, y);
    p1 = ones(d,1);
    alpha = 1;
    % H1 = RSSHess(X);
    toc


    %% Newton with Matrix Sketch
    tic
    for k = 1:maxit
    %     if norm(g1)<ftol
        optgap = dot(g1, p1)^2/2;
        if n == 2^12
            OGap = [OGap, optgap];
            normg = [normg, norm(g1)];
        end
        if optgap <= ftol
            break
        end
        SS = SketchMatrix(X, 2);
        p1 = -SS\g1;
        alpha = 2;
        count = 0;
        while norm(RSSGrad(x1+alpha*p1, X, y))>(1-alpha*eta)*norm(RSSGrad(x1, X, y))
            alpha = alpha*gamma;
            count = count + 1;
        end
        x1 = x1 + alpha*p1;
        g1 = RSSGrad(x1, X, y);
    end
    t = toc;
    tt = [tt , t];
    numits = [numits, k];
    if n == 2^12
        figure(1)
        hold on
        semilogy(1:k, OGap, 'r', 'LineWidth', 1.5, 'DisplayName', 'Sketch')
        set(gca,'yscale','log')
        xlabel('Number of iterations')
        ylabel('Optimality Gap')
        legend
        figure(2)
        hold on
        semilogy(1:k, normg, 'r', 'LineWidth', 1.5, 'DisplayName', 'Sketch')
        set(gca,'yscale','log')
        xlabel('Number of iterations')
        ylabel('norm of gradient')
        legend
    end
%     x1(1:10)
%     k
%     norm(g1)
%     optgap
%     norm(H1-SS)
end
figure(3)
hold on
plot(nind1, numits, 'r', 'LineWidth', 1.5, 'DisplayName', 'Sketch')
xlabel('Size of n')
ylabel('Iterations required')
legend
figure(4)
hold on
semilogy(nind1, tt, 'r', 'LineWidth', 1.5, 'DisplayName', 'Sketch')
set(gca,'yscale','log')
xlabel('Size of n')
ylabel('Run-time')
legend
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

%% RSS Hessian evaluation
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

%% Efficiently apply sketch matrix to square root Hessian
function SS = SketchMatrix(X, method)
% method = 1: Gaussian sketch
% method = 2: FJLT sketch
[n, d] = size(X);
m = 1.2*d; % sample size
if method == 1
    S = randn(m,n);
    SX = S*X;
elseif method == 2
    SX = zeros(m, d);
    rad = (rand(n,1)>0.5)*2-1; % Rademacher dist. realizations
    DX = rad.*X;
    HDX = FWHT(DX); % fast Walsh-Hadamard transform: implemented below
%     ind = randperm(n, m); % without replacement
    ind = randi(n, m, 1); % with replacement
    for i = 1:m
        SX(i,:) = HDX(ind(i),:);
    end
else
    error('method must be 1 or 2')
end
SS = SX'*SX;
end

%% Fast Walsh-Hadamard Transform
function HDX = FWHT(X)
[n,~] = size(X);
HDX = X;
for i = 1:log2(n)    
    for j = 1:n/(2^i) 
        HDX(2^i*(j-1)+1:2^i*j,:)=[HDX(2^i*(j-1)+1:2^i*(j-1)+2^(i-1), :) + HDX(2^i*(j-1)+2^(i-1)+1:2^i*j, :);...
            HDX(2^i*(j-1)+1:2^i*(j-1)+2^(i-1), :) - HDX(2^i*(j-1)+2^(i-1)+1:2^i*j, :)];
    end
end
end

