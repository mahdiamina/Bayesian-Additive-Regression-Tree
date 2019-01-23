function [X Y] = compare2Arrays(A,B)

% Transposing the arrays if they are not comlumn vectors
[m,n]=size(A);
if n>=m
A=A';
end
[m,n]=size(B);
if n>=m
B=B';
end

C=repmat(A,1,size(B,1));
D=C-repmat(B',size(A,1),1);
[X,Y]=find(D==0);

