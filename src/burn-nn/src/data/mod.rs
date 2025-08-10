pub mod augmentation;
pub mod normalize;

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            Dataset,
            transform::{PartialDataset, ShuffledDataset},
            vision::{Annotation, ImageDatasetItem, ImageFolderDataset, PixelDepth},
        },
    },
    prelude::*,
    tensor::Tensor,
};

use crate::common::{CHANNELS, HEIGHT, WIDTH};
use crate::data::augmentation::{ImageAugmenter, image_dset_to_image};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatcherMode {
    Train,
    Eval,
}

#[derive(Clone)]
pub struct ImageBatcher {
    mode: BatcherMode,
    augmenter: Option<ImageAugmenter>,
}

impl Default for ImageBatcher {
    fn default() -> Self {
        Self {
            mode: BatcherMode::Eval,
            augmenter: None,
        }
    }
}

impl ImageBatcher {
    pub fn train_default() -> Self {
        Self {
            mode: BatcherMode::Train,
            augmenter: Some(ImageAugmenter::new(Default::default())),
        }
    }
    pub fn eval() -> Self {
        Self {
            mode: BatcherMode::Eval,
            augmenter: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, ImageDatasetItem, ImageBatch<B>> for ImageBatcher {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> ImageBatch<B> {
        let images: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|item| {
                // Convert dataset item to tensor, with optional augmentation in train mode.
                match (self.mode, &self.augmenter) {
                    (BatcherMode::Train, Some(aug)) => {
                        let img = image_dset_to_image(&item.image);
                        // Optional: keep size; rot adds black corners but is acceptable for now.
                        let img = aug.augment(&img);
                        convert_image_to_tensor::<B>(&img, device)
                    }
                    _ => convert_dset_item_to_tensor::<B>(&item.image, device),
                }
            })
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
        TensorData::new(pixels, [CHANNELS, HEIGHT, WIDTH]).convert::<B::FloatElem>(),
        device,
    )
    .reshape([1, CHANNELS, HEIGHT, WIDTH])
}

fn convert_image_to_tensor<B: Backend>(
    img: &imageproc::definitions::Image<image::Rgb<u8>>,
    device: &B::Device,
) -> Tensor<B, 4> {
    assert_eq!(
        img.width() as usize,
        WIDTH,
        "Unexpected width: {} != {}",
        img.width(),
        WIDTH
    );
    assert_eq!(
        img.height() as usize,
        HEIGHT,
        "Unexpected height: {} != {}",
        img.height(),
        HEIGHT
    );

    let hw = HEIGHT * WIDTH;
    let mut buf = vec![0f32; CHANNELS * hw];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let p = img.get_pixel(x as u32, y as u32).0;
            let idx = y * WIDTH + x;
            buf[0 * hw + idx] = p[0] as f32 / 255.0;
            buf[1 * hw + idx] = p[1] as f32 / 255.0;
            buf[2 * hw + idx] = p[2] as f32 / 255.0;
        }
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(buf, [CHANNELS, HEIGHT, WIDTH]).convert::<B::FloatElem>(),
        device,
    )
    .reshape([1, CHANNELS, HEIGHT, WIDTH])
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

pub fn load_train_val_datasets(
    seed: u64,
    train_ratio: f32,
) -> (
    PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, ImageDatasetItem>,
    PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, ImageDatasetItem>,
) {
    let dataset = load_dataset();
    let len = dataset.len();
    let train_size = ((len as f32) * train_ratio).round() as usize;
    let train_size = train_size.clamp(1, len - 1);

    let shuffled_for_train: ShuffledDataset<_, _> = ShuffledDataset::with_seed(dataset, seed);
    let shuffled_for_val: ShuffledDataset<_, _> = ShuffledDataset::with_seed(load_dataset(), seed);

    let train = PartialDataset::new(shuffled_for_train, 0, train_size);
    let val = PartialDataset::new(shuffled_for_val, train_size, len);

    (train, val)
}
