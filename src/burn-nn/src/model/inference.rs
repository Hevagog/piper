use crate::model::{self, resnet::ResNet50};
use burn::{
    prelude::*,
    tensor::{Tensor, cast::ToElement},
};

pub fn infer<B: Backend>(model: &ResNet50<B>, images: Tensor<B, 4>) -> usize {
    let raw_output = model.forward(images);
    let (score, idx) = raw_output.max_dim_with_indices(1);
    let idx = idx.into_scalar().to_usize();
    idx
}
