clear;
clc;

load("x.csv")
load("y.csv")
load("wa.csv")
load("ua.csv")
load("va.csv")
load("w.csv")
load("u.csv")
load("v.csv")

figure(1);
contourf(x,y,wa,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("wa")

figure(2);
contourf(x,y,w,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("w")

figure(3);
contourf(x,y,ua,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("ua")

figure(4);
contourf(x,y,u,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("u")

figure(5);
contourf(x,y,va,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("va")

figure(6);
contourf(x,y,v,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("v")

figure(7);
contourf(x,y,wa-w,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("werr")

figure(8);
contourf(x,y,ua-u,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("uerr")

figure(9);
contourf(x,y,va-v,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("verr")

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