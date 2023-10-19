clear;
clc;

load("x.csv")
load("y.csv")
load("F1.csv")
load("F2.csv")
load("r.csv")

figure(1);
contourf(x,y,F1,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("forcing term")

figure(2);
contourf(x,y,F2,50,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("Fwd-Bwd test")

figure(3);
contourf(x,y,r,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("results")

figure(4);
ana = r;
for i = 1:size(x,1)
    for j = 1:size(y,2)
        r2 = (x(i,j)-pi)^2 + (y(i,j)-pi)^2;
        ana(i,j) = exp(-r2/0.2);
    end
end
contourf(x,y,ana,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("analytic solution")

figure(5);
err = ana-r;
contourf(x,y,err,20,"linestyle","none")
colormap(jet)
colorbar();
daspect([1 1 1])
xlabel("err distribution")