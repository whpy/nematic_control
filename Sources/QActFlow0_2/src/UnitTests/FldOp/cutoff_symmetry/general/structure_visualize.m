clear;
clc;

load("x.csv")
load("y.csv")
load("benchmark.csv")
load("single.csv")
load("modifiedTrans.csv")

figure(1);
contourf(x,y,benchmark,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("bench")

figure(2);
contourf(x,y,single,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("single")

figure(3);
contourf(x,y,modifiedTrans,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("mod")


figure(4);
contourf(x,y,benchmark-single,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("s err")

figure(5);
contourf(x,y,benchmark-modifiedTrans,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("mod err")

disp("L2 of single")
disp( max(max( (benchmark-single).^2 )) );

disp("L2 of modified")
disp( max(max( (benchmark-modifiedTrans).^2 )) );




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