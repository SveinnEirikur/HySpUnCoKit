load('/Volumes/Music/Datasets/Samson/Mat/samson_1.mat')
load('/Volumes/Music/Datasets/Samson/GroundTruth/end3.mat')
nEnd = 3;
X = Y;
Aref = M;
Sref = A;
Ae = VCA(X,'Endmembers',nEnd);
%[Ae,~] = EIA_ATGP(X,nEnd);
%Ae = Aman;
Ae = max(Ae,0);

[sad_m,idx_org_m,idx_hat_m,sad_k,s0] = calcSAD2(Aref,Ae)
S = (ones(nEnd,nRow*nCol)*0.33).';
q = 0.5;
delta = 15;
h = 0.6;
maxIter = 5000;
calcMisc = 1;
tic;
[A,S,J,SAD,t]=lhalf(Aref,Ae,S,X,q,delta,h,maxIter,calcMisc);
[sad_m,idx_org_m,idx_hat_m,sad_k,s0] = calcSAD2(A,Aref)
size(A)
toc
figure(2);
display_unmix;

figure(3);
clear Simg;
Simg(:,1) = S(:,1)./max(S(:,1));
Simg(:,2) = S(:,2)./max(S(:,2));
Simg(:,3) = S(:,3)./max(S(:,3));
Simg = reshape (Simg(:,[1 2 3]), [nRow nCol nEnd]);
imshow(Simg)
caxis('auto')

save('/Volumes/Music/Unmixing/Python/lhalf/samsonVCA.mat', '-v7')
