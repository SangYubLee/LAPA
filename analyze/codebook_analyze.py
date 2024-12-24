import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np

vae = torch.load('/home/work/open_x_dataset/LAPA/laq/results/codebook_size_32/vae.100000.pt')
codebook = vae['module.vq.codebooks'].detach().cpu().numpy()
# temp = torch.randn(8,32)

# 데이터 정규화
scaler = StandardScaler()
codebook = scaler.fit_transform(codebook)


def visualize_codebook_tsne(codebook, find_clusters=True):
    tsne = TSNE(n_components=2, 
                perplexity=5, 
                max_iter=1000, 
                verbose=0)
    tsne_result = tsne.fit_transform(codebook)

    # 클러스터 탐지
    if find_clusters:
        clusters = DBSCAN(eps=15, min_samples=3).fit(tsne_result)
        n_clusters = len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
        n_noise = list(clusters.labels_).count(-1)
        
    # t-SNE 결과 시각화
    plt.figure(figsize=(12, 10))
    if find_clusters:
        scatter = plt.scatter(tsne_result[:, 0], 
                            tsne_result[:, 1],
                            c=clusters.labels_,
                            cmap='Spectral_r',
                            alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f't-SNE Visualization (found {n_clusters} clusters, {n_noise} noise points)')
    else:
        plt.scatter(tsne_result[:, 0], 
                    tsne_result[:, 1],
                    alpha=0.6)
        plt.title('t-SNE Visualization of Codebook Vectors')
        
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, f'Codebook shape: {codebook.shape}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig('tsne.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_codebook_pca(codebook):
    # PCA 수행
    pca = PCA()
    pca.fit(codebook)

    # 분산 비율
    explained_variance_ratio = pca.explained_variance_ratio_

    # 그래프 시각화
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio") 
    plt.title("PCA of Codebook")
    plt.savefig('pca_codebook_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_embedding_independence(codebook):
    
    # Compute similarity matrix between embeddings
    similarity_matrix = np.dot(codebook, codebook.T)    # (K, K)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Plot explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance Ratio") 
    plt.title('Explained Variance Ratio Distribution')
    plt.grid(True)

    cumsum_ratio = np.cumsum(explained_variance_ratio)
    effective_dim = np.sum(cumsum_ratio < 0.95) # 95% of variance explained
    plt.text(0.02, 0.98, 
             f'Effective dimensions (95%): {effective_dim}\nCodebook shape: {codebook.shape}',
             transform=plt.gca().transAxes,
             verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig('explained_variance_ratio.png', dpi=300, bbox_inches='tight')
    plt.show()


# visualize_codebook_pca(temp)
visualize_codebook_tsne(codebook, find_clusters=True)
visualize_embedding_independence(codebook)
