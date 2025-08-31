pub struct AppPaths {
    pub weights_path: String,
    pub dataset_root: String,
    pub artifact_dir: String,
}

impl AppPaths {
    pub fn from_env() -> Self {
        Self {
            weights_path: std::env::var("WEIGHTS_PATH")
                .unwrap_or_else(|_| "data/resnet50-weights.pth".into()),
            dataset_root: std::env::var("DATASET_ROOT").unwrap_or_else(|_| "data/processed".into()),
            artifact_dir: std::env::var("ARTIFACT_DIR")
                .unwrap_or_else(|_| "/tmp/resnet50_artifacts".into()),
        }
    }
}
