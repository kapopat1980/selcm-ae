import numpy as np
PI=np.pi
def frac(v): return v-np.floor(v)
# Proposed 2D Sine-Logistic-Exponential Cross-Coupled Map (2D-SLECM)
def slecm(x,y,a,b):
    xn = frac(a*np.sin(PI*y) + b*np.exp(x)*x*(1-x))
    yn = frac(a*np.sin(PI*xn) + b*np.exp(y)*y*(1-y))
    return xn,yn
def slecm_jac(x,y,a,b):
    # derivatives (frac has derivative 1 a.e.)
    g = lambda u: np.exp(u)*(u*(1-u)) ; gp = lambda u: np.exp(u)*(u*(1-u)+1-2*u)
    xn = frac(a*np.sin(PI*y) + b*g(x))
    dxn_dx = b*gp(x); dxn_dy = a*PI*np.cos(PI*y)
    c = a*PI*np.cos(PI*xn)
    dyn_dx = c*dxn_dx; dyn_dy = c*dxn_dy + b*gp(y)
    return np.array([[dxn_dx,dxn_dy],[dyn_dx,dyn_dy]])
def lyap2d(step,jac,x0,y0,n=6000,burn=500):
    x,y=x0,y0
    for _ in range(burn): x,y=step(x,y)
    Q=np.eye(2); s=np.zeros(2)
    for _ in range(n):
        J=jac(x,y); x,y=step(x,y)
        Q,R=np.linalg.qr(J@Q); s+=np.log(np.abs(np.diag(R))+1e-300)
    return np.sort(s/n)[::-1]
# Classic baselines
def henon(x,y,a=1.4,b=0.3): return 1-a*x*x+y, b*x
def henon_jac(x,y,a=1.4,b=0.3): return np.array([[-2*a*x,1],[b,0]])
if __name__=='__main__':
    for a,b in [(5,5),(10,10),(20,20),(50,10),(100,50)]:
        print('SLECM',a,b, lyap2d(lambda x,y:slecm(x,y,a,b), lambda x,y:slecm_jac(x,y,a,b), 0.3,0.7))
    print('Henon', lyap2d(henon,henon_jac,0.1,0.1))
