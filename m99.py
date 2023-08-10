# Testing dummy data for autoencoder
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d, e = dummy_data(n=13, scale=True, complex=False)
e_np = e.detach().numpy()

pca = PCA(n_components=3)

e_scores = pca.fit(e_np).transform(e_np)

print("Scaled e\n",np.round(e_np,3),"\n")
print("Components e\n",np.round(e_scores,3),"\n")

print("Raw d\n",d,"\n")
d_scaled = StandardScaler().fit_transform(d)
print("Scaled d\n",d_scaled,"\n")
d_scores = pca.fit(d_scaled).transform(d_scaled)
print("Components d\n",np.round(d_scores, 3),"\n")

print(np.sqrt(((e_scores - d_scores) ** 2).mean())) 


layer_widths = [10,8,6,4,2]
encz = zip(layer_widths[:-1], layer_widths[1:])
*hidden, second_last, last = encz
for n_in, n_out in hidden:
    print(n_in, " - ", n_out)
print(second_last[0], " - ", second_last[1], "  second_last")
print(last[0], " - ", last[1], "  last")