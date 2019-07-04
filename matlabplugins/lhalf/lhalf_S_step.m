function S=lhalf_S_step(S,ATA,ATX,h,q)
	if q>0
     	S=(S.*(ATX))./((ATA)*S+diag(h)*q*S.^(q-1));
    else
        S=(S.*(ATX))./((ATA)*S);
	end
    S=max(10^-8,S);
end
