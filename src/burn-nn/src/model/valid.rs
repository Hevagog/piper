use crate::{data::batch::ImageBatch, model::resnet::ResNet50, utils::metrics::accuracy};
use burn::{data::dataloader::DataLoader, nn::loss::CrossEntropyLossConfig, prelude::*};
use color_eyre::Result;

pub fn validate_epoch<B: Backend>(
    model: &ResNet50<B>,
    dataloader: &std::sync::Arc<dyn DataLoader<B, ImageBatch<B>>>,
) -> Result<(f32, f32)> {
    let mut total_loss = 0.0;
    let mut total_accuracy = 0.0;
    let mut batch_count = 0;

    for batch in dataloader.iter() {
        let output = model.forward_head_only(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), batch.labels.clone());
        let accuracy = accuracy(output, batch.labels);

        total_loss += loss.into_scalar().elem::<f32>();
        total_accuracy += accuracy;
        batch_count += 1;
    }

    Ok((
        total_loss / batch_count as f32,
        total_accuracy / batch_count as f32,
    ))
}
