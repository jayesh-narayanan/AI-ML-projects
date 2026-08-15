import numpy as np
import matplotlib.pyplot as plt
#2.1
actual_theta_0=3
actual_theta_1=1
actual_theta_2=2
np.random.seed(42)
x1_train=np.random.normal(3,2,800000)
x2_train=np.random.normal(-1,2,800000)
y_train=(actual_theta_0+actual_theta_1*x1_train+actual_theta_2*x2_train+np.random.normal(0,2**0.5,800000))

x1_test=np.random.normal(3,2,200000)
x2_test=np.random.normal(-1,2,200000)
y_test=actual_theta_0+actual_theta_1*x1_test+actual_theta_2*x2_test+np.random.normal(0,2**0.5,200000)
m=len(y_train)
batch_sizes=[1,80,8000]
theta_batch=np.zeros((len(batch_sizes),3))
learning_rate=0.001
epsilons=[10**-1,10**-1/(80**0.5),10**-1/(8000**0.5)]
for k in range(len(batch_sizes)):
  batch=batch_sizes[k]
  # variable convergence condition
  epsilon=epsilons[k]
  avg_grad=float('inf')
  curr_theta_0=0
  curr_theta_1=0
  curr_theta_2=0
  fig=plt.figure(figsize=(10,8))
  ax=fig.add_subplot(111,projection='3d')
  ax.set_xlabel("Theta 0")
  ax.set_ylabel("Theta 1")
  ax.set_zlabel("Theta 2")
  ax.set_title(f"Cost Function Surface with Gradient Descent Path with batch {batch_sizes[k]}")
  epoch=0
  path=[]
  while(avg_grad>epsilon):
    indices = np.arange(m)
    np.random.shuffle(indices)
    x1_train, x2_train, y_train = x1_train[indices], x2_train[indices], y_train[indices]
    total_grad_0=0
    total_grad_1=0
    total_grad_2=0
    for i in range(0,m,batch):
      x1_batch=x1_train[i:i+batch]
      x2_batch=x2_train[i:i+batch]
      y_batch=y_train[i:i+batch]
      errors = y_batch - (curr_theta_0 + curr_theta_1*x1_batch + curr_theta_2*x2_batch)
      curr_update_0 = -np.sum(errors)
      curr_update_1 = -np.sum(errors * x1_batch)
      curr_update_2 = -np.sum(errors * x2_batch)
      curr_theta_0-=(curr_update_0/batch)*learning_rate
      curr_theta_1-=(curr_update_1/batch)*learning_rate
      curr_theta_2-=(curr_update_2/batch)*learning_rate
      path.append((curr_theta_0, curr_theta_1, curr_theta_2))
      total_grad_0+=curr_update_0
      total_grad_1+=curr_update_1
      total_grad_2+=curr_update_2

    avg_grad = np.sqrt(total_grad_0**2 + total_grad_1**2 + total_grad_2**2) / m
    epoch+=1
    if epoch%10==0:
      print(f"[Batch {batch}] Epoch {epoch}, avg_grad={avg_grad:.5f}, "
            f"theta=({curr_theta_0:.3f}, {curr_theta_1:.3f}, {curr_theta_2:.3f})")

  path = np.array(path)
  ax.plot(path[:, 0], path[:, 1], path[:, 2], color='r', marker='o', markersize=1)
  ax.legend()
  plt.show()
  print(f"batch_size: {batch}")
  print(f"curr_theta_0: {curr_theta_0}")
  print(f"curr_theta_1: {curr_theta_1}")
  print(f"curr_theta_2: {curr_theta_2}")
  theta_batch[k][0]=curr_theta_0
  theta_batch[k][1]=curr_theta_1
  theta_batch[k][2]=curr_theta_2
  print(f"Batch: {batch}, Epochs_taken: {epoch}, testing_error={(1/(2*len(y_test)))*(np.sum((y_test-(theta_batch[k][0]+theta_batch[k][1]*x1_test+theta_batch[k][2]*x2_test))**2))}, "
         f"training_error={(1/(2*len(y_train)))*(np.sum((y_train-(theta_batch[k][0]+theta_batch[k][1]*x1_train+theta_batch[k][2]*x2_train))**2))}")

def closed_form_solution_step_by_step(X, y):
    # Step 1: Add a column of ones for the intercept term (theta_0)
    m = X.shape[0]  
    ones_column = np.ones(m)
    X_with_intercept = np.column_stack([ones_column, X])

    # Step 2: Compute X transpose (Xᵀ)
    X_transpose = X_with_intercept.T

    # Step 3: Compute XᵀX
    X_transpose_X = X_transpose @ X_with_intercept

    # Step 4: Compute (XᵀX)⁻¹ (inverse of XᵀX)
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)

    # Step 5: Compute XᵀY
    X_transpose_Y = X_transpose @ y

    # Step 6: Compute θ = (XᵀX)⁻¹XᵀY
    theta = X_transpose_X_inv @ X_transpose_Y

    return theta

# Example usage with your data
X_train = np.column_stack([x1_train, x2_train])

# Run step-by-step solution
print("=== STEP-BY-STEP CLOSED FORM SOLUTION ===")
theta_closed_form = closed_form_solution_step_by_step(X_train, y_train)

print(f"\n=== RESULTS ===")
print(f"Actual parameters:    θ₀ = {actual_theta_0}, θ₁ = {actual_theta_1}, θ₂ = {actual_theta_2}")
print(f"Estimated parameters: θ₀ = {theta_closed_form[0]:.4f}, θ₁ = {theta_closed_form[1]:.4f}, θ₂ = {theta_closed_form[2]:.4f}")
print(f"testing_error={(1/(2*len(y_test)))*(np.sum((y_test-(theta_closed_form[0]+theta_closed_form[1]*x1_test+theta_closed_form[2]*x2_test))**2))}, "
         f"training_error={(1/(2*len(y_train)))*(np.sum((y_train-(theta_closed_form[0]+theta_closed_form[1]*x1_train+theta_closed_form[2]*x2_train))**2))}")
