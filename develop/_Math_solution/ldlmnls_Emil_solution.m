function x = ldlmnls(A,b)
% Minimum-norm least squares solver of normal equations by LDL
% decomposition, and QR refinement to get the minimum norm

%% normal equations
AtA=A'*A;

%% LDL
[L,D,p] = ldl(AtA,'vector');

tolerance = max(size(A))*abs(D(1,1))*eps(class(A));

r = abs(diag(D)) > tolerance;
L = L(:,r);
D = D(r,r);

%% least squares
x1 = A(:,p)'*b;
opts.LT = true;
x2 = linsolve(L,x1,opts);
x3 = mldivide(D,x2);
%% Least squares solution: (not minimum-norm)
% opts2.LT=true; opts2.TRANSA=true;
% x(p,:) = linsolve(L,x3,opts2);

%% QR to get the minimum norm solution to L'*x = x3 :
% Section 5.7.2, algorithm 5.7.2, of "Matrix Computations" Golub & Van Loan
[Q, R] = qr(L,0);
opts3.UT=true; opts3.TRANSA=true;
x(p,:) = Q*linsolve(R,x3,opts3);

