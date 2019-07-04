[~,nEnd] = size(A);

for i = 1 : nEnd
    subplot (2,nEnd, i);
    plot (A(:,i));
    title('Endmember ' + string(i));
    grid on;
    
    a2d = reshape (S(:,i), [nRow nCol]);
    subplot (2,nEnd, i+nEnd);
    mynd = imshow (a2d, []);
    caxis('auto')
    colormap( mynd.Parent, jet );
    colorbar( mynd.Parent );
    title('Endmember ' + string(i));
end