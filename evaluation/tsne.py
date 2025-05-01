import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

PATH = '/home/luan/Desktop/ufpr/2025-1/bioinformatica/t1/lbce_classifier/data/embeddings/CLS_fea.txt'
data = np.loadtxt(PATH, delimiter=',') 

labels = data[:, 0].astype(int)
embeddings = data[:, 1:]
print(f"Dimensão dos embeddings: {embeddings.shape[1]}")

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(
    embeddings_2d[labels == 0, 0],
    embeddings_2d[labels == 0, 1],
    label='Negativo (0)', color='red', alpha=0.6, s=40
)
plt.scatter(
    embeddings_2d[labels == 1, 0],
    embeddings_2d[labels == 1, 1],
    label='Positivo (1)', color='blue', alpha=0.6, s=40
)

plt.title('Projeção t-SNE dos Embeddings')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('tsne_plot.png', dpi=300)
plt.show()
