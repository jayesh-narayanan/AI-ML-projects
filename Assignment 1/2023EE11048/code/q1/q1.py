import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1.1 (getting the learning rate and hypothesis function)
df_x=pd.read_csv("/content/linearX.csv")
df_y=pd.read_csv("/content/linearY.csv")
x = df_x.values.flatten()
y = df_y.values.flatten()
size = x.size
flag=False
learning_rate=0.1
epsilon=10**-5
max_iters=1000
prev_cost = float('inf')
# getting highest possible learning rate
while(not flag):
    print(f"learning_rate: {learning_rate}")
    theta_0=0
    theta_1=0
    for i in range(1000):
        cost=0
        for j in range(size):
            cost+=(y[j]-x[j]*theta_1-theta_0)**2
        if (abs(cost/(2*size)-prev_cost)<epsilon):
            flag=True
            break
        prev_cost=cost/(2*size)
        update_0=0
        update_1=0
        for j in range(size):
            update_0-=(y[j]-x[j]*theta_1-theta_0)
            update_1-=(y[j]-x[j]*theta_1-theta_0)*x[j]
        theta_0-=(update_0/size)*learning_rate
        theta_1-=(update_1/size)*learning_rate
        if i % 100 == 0:
          print(f"Iteration: {i}, Cost: {cost}, update_0: {update_0}, update_1: {update_1}")

    learning_rate/=10

final_learning_rate=learning_rate*10
final_theta_0=theta_0
final_theta_1=theta_1
print(f"final_learning_rate: {final_learning_rate}")
print(f"final_theta_0: {final_theta_0}")
print(f"final_theta_1: {final_theta_1}")
print(f"final_cost: {cost/(2*size)}")

# 1.2 (plot)
plt.scatter(x, y, label="Data points")
plt.plot(x, final_theta_0 + final_theta_1*x, color="red", label="Fitted line")
plt.legend()
plt.show()

# 1.3
def compute_cost(theta0, theta1, x, y):
    m = len(y)
    predictions = theta0 + theta1 * x
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)


# Define range 
theta_0_vals = np.linspace(0, 2*final_theta_0, 100)
theta_1_vals = np.linspace(0, 2*final_theta_1, 100)

# Meshgrid
T0, T1 = np.meshgrid(theta_0_vals, theta_1_vals)

# Compute cost 
J_vals = np.zeros_like(T0)
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        J_vals[i, j] = compute_cost(T0[i, j], T1[i, j], x, y)


# Plot 3D surface
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(T0, T1, J_vals, cmap="coolwarm", alpha=0.4)

ax.set_xlabel("Theta 0")
ax.set_ylabel("Theta 1")
ax.set_zlabel("Cost J")
ax.set_title("Cost Function Surface with Gradient Descent Path")
ax.view_init(elev=20, azim=120)
fig.colorbar(surf, shrink=0.5, aspect=3)

# Gradient descent visualization
X = np.array(x)
Y = np.array(y)
theta_0, theta_1 = 0, 0
prev_cost = float("inf")
for i in range(max_iters):
    predictions = theta_0 + theta_1 * X
    errors = predictions - Y
    update_0 = np.sum(errors) / size
    update_1 = np.sum(errors * X) / size
    theta_0 -= final_learning_rate* update_0
    theta_1 -= final_learning_rate* update_1
    cost = compute_cost(theta_0, theta_1, X, Y)
    ax.scatter(theta_0,theta_1,cost,color='g',s=5)
    plt.draw()
    plt.pause(0.2)
    # print(f"Iteration {i}: θ₀={theta_0:.2f}, θ₁={theta_1:.2f}, Cost={cost:.2f}")
    if abs(prev_cost - cost) < epsilon:
        break
    prev_cost = cost
plt.show()

# 1.4,1.5
learning_rates=[final_learning_rate,0.01,0.025,0.1]
for lr in learning_rates:

  # Contour plot
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111)
  cont= ax.contour(T0, T1, J_vals, levels=30,cmap="coolwarm", alpha=0.8)
  learning_rate=1
  ax.set_xlabel("Theta 0")
  ax.set_ylabel("Theta 1")
  ax.set_title("Cost Function Contour with Gradient Descent Path")
  fig.colorbar(surf, shrink=0.5, aspect=3)

  # Gradient descent visualization
  X = np.array(x)
  Y = np.array(y)
  theta_0, theta_1 = 0, 0
  prev_cost = float("inf")
  for i in range(max_iters):
      predictions = theta_0 + theta_1 * X
      errors = predictions - Y
      update_0 = np.sum(errors) / size
      update_1 = np.sum(errors * X) / size
      theta_0 -= lr* update_0
      theta_1 -= lr* update_1
      cost = compute_cost(theta_0, theta_1, X, Y)
      ax.scatter(theta_0,theta_1,color='r',s=5)
      plt.draw()
      plt.pause(0.2)
      # print(f"Iteration {i}: θ₀={theta_0:.2f}, θ₁={theta_1:.2f}, Cost={cost:.2f}")
      if abs(prev_cost - cost) < epsilon:
          break
      prev_cost = cost

  plt.show()

