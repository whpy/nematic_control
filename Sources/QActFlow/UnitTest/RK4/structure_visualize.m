clear;
clc;

load("x.csv")
load("y.csv")
load("w.csv")
load("w150.csv")
load("w250.csv")
load("w350.csv")
load("w450.csv")
load("w550.csv")


% original function
viF(1,x,y,w,"w")

viF(2,x,y,w150,"w150")

viF(3,x,y,w250,"w250")

viF(4,x,y,w350,"w350")

viF(5,x,y,w450,"w450")

%%%%%%%%%%%%%%% function block %%%%%%%%%%%%%%%%%%
function viF(no, x, y, fld, name)
figure(no);
contourf(x,y,fld,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel(name)
end
% figure(2);
% contourf(x,y,F2,50,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("Fwd-Bwd test")
% 
% figure(3);
% contourf(x,y,r,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("results")
% 
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
% 
% figure(5);
% err = ana-r;
% contourf(x,y,err,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("err distribution")