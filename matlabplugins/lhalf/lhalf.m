function [A,S,J,SAD,t]=lhalf(Aorg,A,S,X,q,delta,h,maxIter,calcMisc)

% delta gaur ~ 1 - 100
% h gaur ~ 1
% q default = 1/2

J=[];
t=[];
SAD=[];
S=S';
[M,r]=size(A);
[M,P]=size(X);

X=[X;delta*ones(1,P)];
A=[A;delta*ones(1,r)];
tic;
j=1;
for i=1:maxIter
    
    A(1:end-1,:)=lhalf_A_step(X(1:end-1,:)*S',S*S',A(1:end-1,:));
    S           =lhalf_S_step(S,A'*A,A'*X,h,q);

    if calcMisc==1 
        [SAD(i),~, ~,~]=calcSAD2(Aorg,A(1:end-1,:));
        if mod(i,10)==0
            hS=diag(h)*(S.^q);
            J(j)=0.5*sum(sum((X-A*S).^2))+sum(hS(:));j=j+1;
        end
    end
    t(i)=toc;
end
A=A(1:end-1,:);
S=S';
end

