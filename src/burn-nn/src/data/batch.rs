use crate::data::{
    augmentation::{AugmentationConfig, ImageAugmenter, image_dset_to_image},
    conversion::{
        convert_dset_annotation_to_label, convert_dset_item_to_tensor, convert_image_to_tensor,
    },
    normalize::{NormalizeConfig, normalize},
};

use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::ImageDatasetItem},
    prelude::*,
    tensor::Tensor,
};

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
            augmenter: Some(ImageAugmenter::new(AugmentationConfig::default())),
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
                        match image_dset_to_image(&item.image).and_then(|img| aug.augment(&img)) {
                            Ok(img_aug) => convert_image_to_tensor::<B>(&img_aug, device),
                            Err(e) => {
                                eprintln!("[augment warning] {}", e);
                                convert_dset_item_to_tensor::<B>(&item.image, device)
                            }
                        }
                    }
                    _ => convert_dset_item_to_tensor::<B>(&item.image, device),
                }
            })
            .map(|tensor| normalize(tensor, &NormalizeConfig::default()))
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
