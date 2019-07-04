function [sad, idxOrg, idxHat,sad_k,s0] =calcSAD(Aorg,Ahat,startCol)
% Calculates the mean spectral angle distance between the columns of
% Aorg and Ahat.
% idxOrg and idxHat are the indexes of the columns that are paired
% together.
%
[~,r1]=size(Aorg);
[~,r2]=size(Ahat);
if nargin==2
    startCol=0;
end
for i=1:r1
    ao=Aorg(:,i);
    for j=1:r2
        ah=Ahat(:,j);
        s(j,i)=min(cs(ao,ah),cs(ao,-ah));
    end
end
s0=s;
for p=1:r1
    if startCol>0&& p==1
        [sad(p),b]=min(s(:,startCol));
        idxHat(p)=b;
        idxOrg(p)=startCol;

    else
        [sad(p),b]=min(s(:));
        [idxHat(p),idxOrg(p)]=ind2sub(size(s),b);
    end

    s(:,idxOrg(p))=inf;
    s(idxHat(p),:)=inf;    
    if isinf(sad(p))
        idxHat(p)=Inf;
        idxOrg(p)=Inf;
    end
end
sad_k=sad;

[a,b]=sort(idxOrg);
idxOrg=a;
idxHat=idxHat(b);
sad_k=sad_k(b);

sad=mean(sad);
end
function s=cs(ai,aj)
    if sum(abs(aj-ai))==0
        s=0;
        return;
    end
    s=(aj'*ai)/(norm(ai,2)*norm(aj,2));
    s=acos(s);
    if abs(imag(s)<0.0000001)
        s=real(s);
    end
end





