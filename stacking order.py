import matplotlib.pyplot as plt

# Define nodes
nodes = ['Input', 'Random Forest', 'CatBoost', 'SVM', 'Stacking', 'Output']

# Define edges
edges = [('Input', 'Random Forest'), ('Input', 'CatBoost'), ('Input', 'SVM'),
         ('Random Forest', 'Stacking'), ('CatBoost', 'Stacking'), ('SVM', 'Stacking'),
         ('Stacking', 'Output')]

# Draw diagram
plt.figure(figsize=(8, 6))
for edge in edges:
    plt.plot([nodes.index(edge[0]), nodes.index(edge[1])], [0, 1], 'b-')
plt.scatter([nodes.index(node) for node in nodes], [0]*len(nodes), color='blue', zorder=5)
plt.scatter([nodes.index('Stacking')], [1], color='red', zorder=5)
plt.yticks([])
plt.xticks(range(len(nodes)), nodes, rotation=45)
plt.title('Stacking Order Diagram')
plt.show()
