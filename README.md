# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries.
2. Load the Mall Customers dataset using Pandas.
3. Select Annual Income and Spending Score features.
4. Apply feature scaling using StandardScaler.
5. Initialize the K-Means algorithm with 5 clusters.
6. Fit the K-Means model to the dataset.
7. Predict the cluster labels for each customer.
8. Visualize the customer segments using a scatter plot.
9. Display the clusters and their centroids.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: 
RegisterNumber:  
*/
```
```PY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")
data.head()

# Select Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
X.head()

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

data['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=y_kmeans,
    cmap='rainbow',
    s=50
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title('K-Means Clustering for Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Display customer segmentation results
print("Customer Segmentation Results:")
print(data[['CustomerID',
            'Annual Income (k$)',
            'Spending Score (1-100)',
            'Cluster']].head(10))

# Count customers in each cluster
print("Number of Customers in Each Cluster:")
print(data['Cluster'].value_counts().sort_index())


## Output:
<img width="685" height="203" alt="image" src="https://github.com/user-attachments/assets/d3eac0fc-f905-48a6-93b4-1a00187e5d4c" />
<img width="447" height="208" alt="image" src="https://github.com/user-attachments/assets/9202cd44-9bd0-4b47-b223-c1fe42722b95" />
<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/9d01c76b-3ea1-4367-acd1-36bc3066b0ff" />
<img width="675" height="305" alt="image" src="https://github.com/user-attachments/assets/3ea8092b-2a78-48d8-ad4d-84e150c47a2e" />
<img width="400" height="210" alt="image" src="https://github.com/user-attachments/assets/b22a7386-06d7-47f7-80e8-81475ac18c81" />
<img width="400" height="210" alt="image" src="https://github.com/user-attachments/assets/5d029be6-eab2-4dc1-829b-05210e64abf2" />




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
