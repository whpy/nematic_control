clear;
clc;

load("x.csv")
load("y.csv")
load("u.csv")
load("v.csv")
load("ua.csv")
load("va.csv")
load("w.csv")
load("wa.csv")
load("r1.csv")
load("r2.csv")
load("Sa.csv")
load("S.csv")

figure(1);
contourf(x,y,w-wa,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("werr")

figure(2);
contourf(x,y,v-va,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("verr")

figure(3);
contourf(x,y,u-ua,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("uerr")

figure(4);
contourf(x,y,S-Sa,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("Serr")


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