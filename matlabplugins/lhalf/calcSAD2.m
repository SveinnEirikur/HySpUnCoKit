function [sad_m, idxOrg_m, idxHat_m,sad_k_m,s0] =calcSAD2(Aorg,Ahat)
% Calculates the sad r times, each time staring with a new column. 
% The best result is returned. 
r=size(Aorg,2);
for i=1:r
    [sad(i), idxOrg(i,:), idxHat(i,:),sad_k(i,:),s0] =calcSAD(Aorg,Ahat,i);
end

[sad_m,idx]=min(sad);
idxOrg_m=idxOrg(idx,:);
idxHat_m=idxHat(idx,:);
sad_k_m=sad_k(idx,:);

end