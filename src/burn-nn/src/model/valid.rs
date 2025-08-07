use crate::{
    data::ImageBatch,
    model::{metrics::accuracy, resnet::ResNet50},
};
use burn::{data::dataloader::DataLoader, nn::loss::CrossEntropyLoss, prelude::*};

pub fn validate_epoch<B: Backend>(
    model: &ResNet50<B>,
    dataloader: &std::sync::Arc<dyn DataLoader<B, ImageBatch<B>>>,
    device: &B::Device,
) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut total_accuracy = 0.0;
    let mut batch_count = 0;

    for batch in dataloader.iter() {
        let output = model.forward_head_only(batch.images);
        let loss =
            CrossEntropyLoss::new(None, device).forward(output.clone(), batch.labels.clone());
        let accuracy = accuracy(output, batch.labels);

        total_loss += loss.into_scalar().elem::<f32>();
        total_accuracy += accuracy;
        batch_count += 1;
    }

    (
        total_loss / batch_count as f32,
        total_accuracy / batch_count as f32,
    )
}
