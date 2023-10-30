clear;
clc;
close all;

namelist = dir("*.csv");
for i = 1:length(namelist)
    load(namelist(i).name)
end
% visual(X0u,1,x,y)
% visual(X2000u,1,x,y)
% visual(X4000u,1,x,y)
% visual(X6000u,1,x,y)
visual(p12a, 1, x,y)
title("p12a")
visual(p12, 2, x,y)
title("p12")
visual(p12a-p12, 3, x,y)
title("p12 err")

visual(p11a, 4, x,y)
title("p11a")
visual(p11, 5, x,y)
title("p11")
visual(p11a-p11, 6, x,y)
title("p11 err")
% % visual(Dxxp12, 3, x, y)
visual(nl0, 7, x, y)
visual(nl0a, 8, x, y)
visual(nl0a-nl0, 9, x, y)
% visual(nl1a, 2, x, y)
% title("nl1a")
% visual(nl1, 3, x, y)
% title("nl1")
% visual(nl2a, 4, x, y)
% title("nl2a")
% visual(nl2, 5, x, y)
% title("nl2")
% 
% visual(p11a, 6, x, y)
% title("p11a")
% visual(p11, 7, x, y)
% title("p11")
% 
% visual(p12a, 8, x, y)
% title("p12a")
% visual(p12, 9, x, y)
% title("p12")
% visual(p21a, 10, x, y)
% title("p21a")
% visual(p21, 11, x, y)
% title("p21")
% visual(nl0a, 12, x, y)
% title("nl0a")
% visual(nl0, 13, x, y)
% title("nl0")
% visual(p11-p11a,14,x,y)
% err = p11 - p11a;
% visual(err, 15,x,y)
% err = p21 - p21a;
% visual(err, 16,x,y)
% err = p12 - p12a;
% visual(err, 17,x,y)
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