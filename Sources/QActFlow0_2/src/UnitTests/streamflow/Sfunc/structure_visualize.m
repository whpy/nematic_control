clear;
clc;

load("x.csv")
load("y.csv")
load("r1.csv")
load("r2.csv")
load("Sa.csv")
load("S.csv")

figure(1);
contourf(x,y,r1,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("r1")

figure(2);
contourf(x,y,r2,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("r2")

figure(3);
contourf(x,y,Sa,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("Sa")

figure(4);
contourf(x,y,S,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("S")


% figure(4);
% ana = r;
% for i = 1:size(x,1)
%     for j = 1:size(y,2)
%         r2 = (x(i,j)-pi)^2 + (y(i,j)-pi)^2;
%         ana(i,j) = exp(-r2/0.2);
%     end
% end
% contourf(x,y,ana,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("analytic solution")

% figure(4);
% err = ana-r;
% contourf(x,y,err,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("err distribution")