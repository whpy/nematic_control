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
load("single1a.csv")
load("single1.csv")
load("single2a.csv")
load("single2.csv")
load("cross1.csv")
load("cross2.csv")
load("crossa.csv")


% visual(wa-w, 1,x,y)
% visual(ua-u,3,x,y)
% visual(va-v,4,x,y)
% visual(single1a-single1, 5, x,y)
% visual(single2a-single2,6,x,y)
% visual(single1a, 7,x,y)
% visual(single1, 8,x,y)
visual(cross1,9,x,y)
visual(crossa,10,x,y)
visual(cross2,11,x,y)
visual(cross1+cross2,12,x,y)

% visual(err./abs(crossa), 12, x, y);
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