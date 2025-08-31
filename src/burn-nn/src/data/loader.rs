use color_eyre::{
    Result,
    eyre::{WrapErr, bail},
};

use burn::data::dataset::{
    Dataset,
    transform::{PartialDataset, ShuffledDataset},
    vision::{ImageDatasetItem, ImageFolderDataset},
};
use std::path::Path;

pub fn load_dataset(root: &str) -> Result<ImageFolderDataset> {
    if !Path::new(root).exists() {
        bail!("Dataset directory not found: {root}");
    }
    ImageFolderDataset::new_classification(root)
        .wrap_err_with(|| format!("Failed to load ImageFolderDataset at {root}"))
}

pub fn load_train_val_datasets(
    root: &str,
    seed: u64,
    train_ratio: f32,
) -> Result<(
    PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, ImageDatasetItem>,
    PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, ImageDatasetItem>,
)> {
    let dataset = load_dataset(root)?;
    let len = dataset.len();
    if len == 0 {
        bail!("Loaded dataset is empty");
    }
    let train_size = ((len as f32) * train_ratio).round() as usize;
    let train_size = train_size.clamp(1, len - 1);

    let shuffled_for_train: ShuffledDataset<_, _> = ShuffledDataset::with_seed(dataset, seed);
    let shuffled_for_val: ShuffledDataset<_, _> =
        ShuffledDataset::with_seed(load_dataset(root)?, seed);

    let train = PartialDataset::new(shuffled_for_train, 0, train_size);
    let val = PartialDataset::new(shuffled_for_val, train_size, len);

    Ok((train, val))
}
