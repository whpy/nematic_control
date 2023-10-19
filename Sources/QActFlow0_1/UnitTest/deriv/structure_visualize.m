clear;
clc;

load("x.csv")
load("y.csv")
load("w.csv")
load("r1.csv")
load("alpha.csv")
load("alpha_set.csv")
load("w_x.csv")
load("S.csv")
load("r2.csv")


figure(1);
contourf(x,y,w,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("w")

figure(2);
contourf(x,y,r1,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("rq")

figure(3);
contourf(x,y,alpha,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("alpha")

viF(4,x,y,alpha_set,"alpha_set");

viF(5,x,y,w_x,"w_x");

viF(6,x,y,(w_x-r1).^2 ,"w_x error"+num2str(max(max((w_x-r1).^2 ))));

viF(7,x,y,S,"S");

viF(8,x,y,r2,"r2");


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