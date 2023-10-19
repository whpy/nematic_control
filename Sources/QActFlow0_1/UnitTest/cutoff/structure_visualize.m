clear;
clc;

load("x.csv")
load("y.csv")
load("w.csv")
load("r1.csv")
load("r2.csv")
load("w_lap.csv")

viF(1,x,y,r1,"r1")

viF(2,x,y,w,"w")

viF(3,x,y,r2,"r2")

viF(4,x,y,w_lap,"w_l")

viF(5,x,y,(w_lap-r1).^2 ,"w_lap error, max(l_2):  "+num2str(max(max((w_lap-r1).^2 ))));


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