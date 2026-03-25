import numpy as np
G=6.67430e-11
m1=5.972e24
m2=1000
R_E=6378e3
mu=398600


r1=np.array([0,0,0])
v1=np.array([0,0,0])
r2=R_E*np.array([2,0,0])
v2=np.array([0,0,np.sqrt(mu/np.linalg.norm(r2-r1))])
T=2*np.pi*np.sqrt(np.linalg.norm(r2-r1)**3/mu)
print("T=",T)

h=1
t_end=T/4000
t=0

#N=(t_end-t0)/h
#for i in range(0,N):
while t<t_end:
    #m1
    a1=(mu/m1)/np.linalg.norm(r2-r1)**3*(r2-r1)
    #m2
    a2=(mu/m2)/np.linalg.norm(r1-r2)**3*(r1-r2)

    r1=r1+v1*h
    v1=v1+a1*h

    r2=r2+v2*h
    v2=v2+a2*h
    t+=h
print("r1=",r1)
print("v1=",v1)
print("r2=",r2)
print("v2=",v2) 