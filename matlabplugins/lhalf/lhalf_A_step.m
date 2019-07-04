function A=lhalf_A_step(XST,SST,A)
    A=(A.*(XST))./(A*(SST));
end