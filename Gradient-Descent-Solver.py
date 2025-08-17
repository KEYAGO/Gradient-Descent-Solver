from sympy import symbols, diff, lambdify, sin, cos, pi, exp, sqrt, Abs, log
import math
s1, s2 = symbols('x1 x2')
f1 = (s1 + 2*s2 - 7)**2 + (2*s1 + s2 - 5)**2 # Booth Function
f2 = 0.26 * (s1**2 + s2**2) - 0.48 * s1 * s2 # Matyas Function
f3 = sin(s1 + s2) + (s1 - s2)**2 - 1.5*s1 + 2.5*s2 + 1 # McCormick Function
f4 = 2 * s1**2 - 1.05 * s1**4 + (s1**6) / 6 + s1 * s2 + s2**2 # Three-Hump Camel Function
f5 = (4 - 2.1 * s1**2 + (s1**4) / 3) * s1**2 + s1 * s2 + (-4 + 4 * s2**2) * s2**2 # Six-Hump Camel Function
f6 = s1**2 + 2 * s2**2 - 0.3 * cos(3 * pi * s1) - 0.4 * cos(4 * pi * s2) + 0.7 # Bohachevsky Function
f7 = -cos(s1) * cos(s2) * exp(-((s1 - pi)**2) - ((s2 - pi)**2)) # Easom Function
f8 = (1.5 - s1 + s1 * s2)**2 + (2.25 - s1 + s1 * s2**2)**2 + (2.625 - s1 + s1 * s2**3)**2 # Beale Function
f9 = 0.5 * ((s1**4 - 16*s1**2 + 5*s1) + (s2**4 - 16*s2**2 + 5*s2)) # Styblinski-Tang Function
f10 = sin(3 * pi * s1)**2 + (s1 - 1)**2 * (1 + sin(3 * pi * s2)**2) + (s2 - 1)**2 * (1 + sin(2 * pi * s2)**2) # Levy Function
f11 = 0.5 + (sin(s1**2 - s2**2)**2 - 0.5) / (1 + 0.001 * (s1**2 + s2**2))**2 # Schaffer Function No: 2

x1, x2=[], []
for i in range(11):
    x1.append(i)
    x2.append(i)
def e1(x1,x2, i):
    return ((x1[i]-x1[i-10])**2+(x2[i]-x2[i-10])**2)/2 # Adapted from MSE Loss
def e2(x1, x2, i):
    return (Abs(x1[i]-x1[i-10])+Abs(x2[i]-x2[i-10]))/2 # Adapted from MBE Loss

f=f1
e=e1
error=0.00001
x1=[0]
x2=[0]
min_x1, max_x1 = -100, 100
min_x2, max_x2 = -100, 100
alpha=0.02
print (x1, x2, alpha)
dfds1 = diff(f, s1)
dfds2 = diff(f, s2)
print("f(s1, s2) =", f)
print("df/ds1 =", dfds1)
print("df/ds2 =", dfds2)
fnum=lambdify((s1, s2), f)
grad1=lambdify((s1, s2), dfds1)
grad2=lambdify((s1, s2), dfds2)

for i in range(1, 5001):
    x1_new=x1[i-1]-alpha*grad1(x1[i-1], x2[i-1])
    x2_new=x2[i-1]-alpha*grad2(x1[i-1], x2[i-1])
    x1_new = max(min_x1, min(x1_new, max_x1))
    x2_new = max(min_x2, min(x2_new, max_x2))
    x1.append(x1_new)
    x2.append(x2_new)
    if i>10:
        loss=e(x1, x2, i)
        print(x1[i], x2[i], fnum(x1[i], x2[i]), i, loss)
        if loss<error:
            break
    else:
        print(x1[i], x2[i], fnum(x1[i], x2[i]), i, "loss")

print ("Optimisation done. Check Results. Especially for the non-convex functions.")