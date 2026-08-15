import numpy as np
import matplotlib.pyplot as plt

x=np.loadtxt("/content/q4x.dat",dtype=int)
y=np.loadtxt("/content/q4y.dat", dtype=str)

x_norm=(x-np.mean(x, axis=0))/np.std(x, axis=0)

mu_0=np.mean(x_norm[y=="Alaska"], axis=0)
mu_1=np.mean(x_norm[y=="Canada"], axis=0)
phi=np.mean(y=="Canada")
sigma=np.cov(x_norm.T)

print(f"mu_0: {mu_0}")
print(f"mu_1: {mu_1}")
print(f"sigma: {sigma}")
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Training data with decision boundary")

for i in range(len(x_norm)):
  if y[i]=="Alaska":
    ax.scatter(x_norm[i,0],x_norm[i,1],color='red',marker='o')
  else:
    ax.scatter(x_norm[i,0],x_norm[i,1],color='blue',marker='o')

# solve for linear boundary
x1_vals = np.linspace(x_norm[:,0].min(), x_norm[:,0].max(), 100)
sigma_inv=np.linalg.inv(sigma)
theta=sigma_inv@(mu_1-mu_0)
theta_0=0.5*(mu_0.T@sigma_inv@mu_0-mu_1.T@sigma_inv@mu_1)+np.log(phi/(1-phi))
plt.plot(x1_vals, -(theta[0]/theta[1])*x1_vals-theta_0/theta[1], color="green", linewidth=2, label="Decision boundary")
print(f"y={-theta[0]/theta[1]}x+{theta_0/theta[1]}")
# plt.show()
print(theta)
print(theta_0)

sigma_0=np.cov(x_norm[y=="Alaska"].T)
sigma_0_inv=np.linalg.inv(sigma_0)
sigma_1=np.cov(x_norm[y=="Canada"].T)
sigma_1_inv=np.linalg.inv(sigma_1)
print(f"sigma_0: {sigma_0}")
print(f"sigma_1: {sigma_1}")

# solve for quadratic boundary
a=0.5*(sigma_0_inv-sigma_1_inv)
b=sigma_1_inv@mu_1-sigma_0_inv@mu_0
c= -0.5*(mu_1 @ (sigma_1_inv @ mu_1) - mu_0 @ (sigma_0_inv @ mu_0)) \
    - 0.5*np.log(np.linalg.det(sigma_1)/np.linalg.det(sigma_0)) \
    + np.log(phi/(1-phi))

x1_min, x1_max = x_norm[:,0].min()-1, x_norm[:,0].max()+1
x2_min, x2_max = x_norm[:,1].min()-1, x_norm[:,1].max()+1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 400),
                       np.linspace(x2_min, x2_max, 400))

# Evaluate g(x) = x^T A x + b^T x + c on the grid
Xstack = np.stack([xx1, xx2], axis=-1)  # shape (H,W,2)
ax2 = np.einsum('...i,ij,...j->...', Xstack, a, Xstack)  # quadratic term
bx = np.einsum('i,...i->...', b, Xstack)                # linear term
g = ax2 + bx + c
print(a)
print(b)
print(c)

plt.contour(xx1, xx2, g, levels=[0], colors="yel",linewidths=2)  # decision boundary
for i in range(len(x_norm)):
  if y[i]=="Alaska":
    ax.scatter(x_norm[i,0],x_norm[i,1],color='red',marker='o')
  else:
    ax.scatter(x_norm[i,0],x_norm[i,1],color='blue',marker='o')

plt.show()
