clear;
clc;

load("x.csv")
load("y.csv")

load("v2.csv")
load("u3.csv")
load("w2.csv")
visual(v2,1,x,y)
visual(u3,2,x,y)
visual(w2,3,x,y)
% figure(1);
% contourf(x,y,S-Sa,50,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("werr")
% 
% figure(2);
% contourf(x,y,v-va,50,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("verr")
% 
% figure(3);
% contourf(x,y,u-ua,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("uerr")
% 
% figure(4);
% contourf(x,y,S-Sa,20,"linestyle","none")
% colormap(jet)
% colorbar();
% daspect([1 1 1])
% xlabel("Serr")

function visual(f,n,x,y)
    figure(n)
    contourf(x,y,f,20,"linestyle","none")
    colormap(jet)
    colorbar();
    daspect([1 1 1])
end
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