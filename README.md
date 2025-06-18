# rotation-forest

Rotation Forest Ensemble Method

Why Needed: No Rust implementation exists despite proven superior performance to Random Forest on continuous features.

Algorithm Innovation

Unlike Random Forest's feature subsampling, Rotation Forest applies PCA to feature subsets, creating "rotated" 
feature spaces that enable oblique decision boundaries rather than just axis-aligned splits. 

Key Implementation Components

Core Algorithm:

- For each tree: Bootstrap 75% of classes, then 75% of instances
- Randomly partition features into K non-overlapping subsets ResearchGatePubMed
- Apply PCA to each subset (retaining ALL principal components) ResearchGatePubMed
- Construct sparse rotation matrix from PCA results
- Transform entire dataset using rotation matrix
- Train decision tree on rotated features ResearchGate

Advanced Data Structures:

```rust
pub struct RotationMatrix {
    pub matrix: Array2<f64>,          // p×p rotation matrix
    pub feature_subsets: Vec<FeatureSubset>,
    pub subset_count: usize,
}

pub struct FeatureSubset {
    pub indices: Vec<usize>,
    pub pca_components: Array2<f64>,  // eigenvectors
    pub mean: Array1<f64>,            // for centering
}
```

Memory-Efficient Implementation:

- Sparse rotation matrix storage (most entries are zero)
- SVD-based PCA for numerical stability
- Block-wise matrix operations for cache efficiency

Linfa Integration:

- Implements Linfa's Fit and Predict traits
- Generic over base classifier types
- Parallel tree training with rayon
- Integration with existing decision tree implementations

Performance Advantages:

- Typically requires fewer trees than Random Forest (10-50 vs 100+)
- Better performance on correlated features
- Natural handling of non-orthogonal decision boundaries

