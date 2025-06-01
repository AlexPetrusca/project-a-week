import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def generate_scaled_point_cloud(n_points=100):
    xs = np.random.randn(n_points)
    ys = np.random.randn(n_points)
    norm_xys = np.stack((xs, ys))

    transform = np.array([
        [2, 3],
        [3, 1]
    ])
    xys = transform @ norm_xys

    return xys.T


def compute_pca(points):
    """
    Compute PCA of the point cloud
    """
    # Center the data
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point

    # Compute PCA
    pca = PCA(n_components=2)
    pca.fit(centered_points)

    return pca, mean_point, centered_points


def plot_pca_visualization(points, pca, mean_point, centered_points):
    """
    Plot the point cloud with PCA components
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Original points with PCA vectors
    ax1.scatter(points[:, 0], points[:, 1], alpha=0.6, c='blue', s=20)
    ax1.scatter(mean_point[0], mean_point[1], color='red', s=100, marker='x', linewidth=3, label='Mean')

    # Draw principal components as arrows from the mean
    scale_factor = 3  # Scale factor for visibility
    for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
        # Scale arrow length by explained variance
        arrow_length = scale_factor * np.sqrt(variance)
        ax1.arrow(mean_point[0], mean_point[1],
                  component[0] * arrow_length, component[1] * arrow_length,
                  head_width=0.3, head_length=0.2, fc=f'C{i + 2}', ec=f'C{i + 2}',
                  linewidth=3, label=f'PC{i + 1} ({variance / np.sum(pca.explained_variance_):.1%})')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Original Point Cloud with Principal Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Points projected onto principal components
    transformed_points = pca.transform(centered_points)
    ax2.scatter(transformed_points[:, 0], transformed_points[:, 1], alpha=0.6, c='green', s=20)
    ax2.scatter(0, 0, color='red', s=100, marker='x', linewidth=3, label='Origin')

    # Draw axes in PCA space
    ax2.axhline(y=0, color='orange', linestyle='--', alpha=0.7, label='PC1 axis')
    ax2.axvline(x=0, color='purple', linestyle='--', alpha=0.7, label='PC2 axis')

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('Points in PCA Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    return fig


def print_pca_analysis(pca, points):
    """
    Print PCA analysis results
    """
    print("=== PCA Analysis Results ===")
    print(f"Original data shape: {points.shape}")
    print(f"Data range - X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Data range - Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print()

    print("Principal Components:")
    for i, (component, variance, ratio) in enumerate(zip(pca.components_,
                                                         pca.explained_variance_,
                                                         pca.explained_variance_ratio_)):
        print(f"PC{i + 1}: [{component[0]:6.3f}, {component[1]:6.3f}] | "
              f"Variance: {variance:6.2f} | Explained: {ratio:.1%}")

    print(f"\nTotal explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"Principal component angle: {np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0])):.1f}°")


def main():
    """
    Main function to run the complete PCA analysis and visualization
    """
    # Generate scaled point cloud
    print("Generating scaled point cloud...")
    points = generate_scaled_point_cloud(n_points=100)

    # Compute PCA
    print("Computing PCA...")
    pca, mean_point, centered_points = compute_pca(points)

    # Print analysis
    print_pca_analysis(pca, points)

    # Create visualization
    print("Creating visualization...")
    fig = plot_pca_visualization(points, pca, mean_point, centered_points)

    plt.show()

    return points, pca, mean_point


if __name__ == "__main__":
    points, pca, mean_point = main()