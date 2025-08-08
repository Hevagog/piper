pub mod augmentation;
pub mod normalize;

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem, ImageFolderDataset, PixelDepth},
    },
    prelude::*,
    tensor::Tensor,
};

use crate::common::{CHANNELS, HEIGHT, WIDTH};

#[derive(Clone, Default)]
pub struct ImageBatcher {}

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, ImageDatasetItem, ImageBatch<B>> for ImageBatcher {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> ImageBatch<B> {
        let images: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|item| convert_dset_item_to_tensor(&item.image, device))
            .map(|tensor| normalize::normalize(tensor, &normalize::NormalizeConfig::default()))
            .collect();
        let labels: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| convert_dset_annotation_to_label(&item.annotation, device))
            .collect();

        let images = Tensor::cat(
            images, 0, // Concatenate along the batch dimension
        );
        let labels = Tensor::cat(
            labels, 0, // Concatenate along the batch dimension
        );
        ImageBatch {
            images: images,
            labels: labels,
        }
    }
}

fn convert_dset_item_to_tensor<B: Backend>(
    item: &Vec<PixelDepth>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let pixels: Vec<f32> = item
        .iter()
        .filter_map(|p| {
            if let PixelDepth::U8(val) = p {
                Some(*val as f32 / 255.0) // Normalize to [0, 1]
            } else {
                None
            }
        })
        .collect();
    Tensor::<B, 3>::from_data(
        TensorData::new(pixels, [CHANNELS, WIDTH, HEIGHT]).convert::<B::FloatElem>(),
        device,
    )
    .reshape([1, CHANNELS, WIDTH, HEIGHT])
    .to_device(device)
}

fn convert_dset_annotation_to_label<B: Backend>(
    annotation: &Annotation,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let id = match annotation {
        Annotation::Label(idx) => Some(*idx as i32),
        _ => None,
    };
    if let Some(label) = id {
        Tensor::from_data(
            TensorData::new(vec![label], [1]).convert::<B::IntElem>(),
            device,
        )
    } else {
        Tensor::from_data(
            TensorData::new(vec![-1], [1]).convert::<B::IntElem>(),
            device,
        )
    }
}

pub fn load_dataset() -> ImageFolderDataset {
    ImageFolderDataset::new_classification("data/processed").unwrap()
}

// pub fn load_train_test_datasets(
//     seed: u64,
// ) -> (
//     PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, ImageDatasetItem>,
//     PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, ImageDatasetItem>,
// ) {
//     let dataset = load_dataset();
//     let len = dataset.len();
//     let train_size = len * 8 / 10;
//     let shuffled: ShuffledDataset<ImageFolderDataset, ImageDatasetItem> =
//         ShuffledDataset::with_seed(dataset, seed);
//     let train: PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, _> =
//         PartialDataset::new(shuffled, 0, train_size);
//     let test: PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, _> =
//         PartialDataset::new(shuffled, train_size, len - train_size);

//     (train, test)
// }
