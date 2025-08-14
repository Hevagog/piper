use crate::{
    data::{ImageBatch, ImageBatcher, load_train_val_datasets},
    model::{resnet::ResNet50, valid::validate_epoch},
    utils::metrics::accuracy,
};
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::AutodiffModule,
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep, metric::LossMetric,
    },
};
use color_eyre::{Result, eyre::WrapErr};

impl<B: Backend> ResNet50<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<ImageBatch<B>, ClassificationOutput<B>> for ResNet50<B> {
    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.labels);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ImageBatch<B>, ClassificationOutput<B>> for ResNet50<B> {
    fn step(&self, batch: ImageBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.labels)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamWConfig,

    #[config(default = 40)]
    pub num_epochs: usize,

    #[config(default = 18)]
    pub batch_size: usize,

    #[config(default = 14)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1.0e-3)]
    pub learning_rate: f64,

    #[config(default = 0.1)]
    pub label_smoothing: f32,

    #[config(default = 10)]
    pub lr_step_size: usize,

    #[config(default = 0.5)]
    pub lr_gamma: f64,

    #[config(default = 5)]
    pub early_stopping_patience: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    dataset_root: &str,
    model: ResNet50<B>,
    config: TrainingConfig,
    device: B::Device,
) -> Result<()> {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .wrap_err("Failed to save training config JSON")?;

    B::seed(config.seed);

    let batcher_train = ImageBatcher::train_default();
    let batcher_eval = ImageBatcher::eval();

    let (train_ds, val_ds) = load_train_val_datasets(dataset_root, config.seed, 0.8)
        .wrap_err("Failed to load / split datasets")?;

    let dataloader_train = DataLoaderBuilder::new(batcher_train.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_ds);

    let dataloader_test = DataLoaderBuilder::new(batcher_eval)
        .batch_size(config.batch_size)
        .shuffle(config.seed + 1)
        .num_workers(config.num_workers)
        .build(val_ds);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .wrap_err("Failed to save trained model checkpoint")?;
    Ok(())
}

pub fn train_head_only<B: AutodiffBackend>(
    artifact_dir: &str,
    dataset_root: &str,
    mut model: ResNet50<B>,
    device: B::Device,
) -> Result<()> {
    create_artifact_dir(artifact_dir);

    let optim_config = AdamWConfig::new().with_weight_decay(1.0e-2);
    let config = TrainingConfig::new(optim_config);
    B::seed(config.seed);
    config
        .save(format!("{artifact_dir}/config.json"))
        .wrap_err("Failed to save head-only config JSON")?;

    let batcher_train = ImageBatcher::train_default();
    let batcher_eval = ImageBatcher::eval();

    let (train_ds, val_ds) = load_train_val_datasets(dataset_root, config.seed, 0.8)
        .wrap_err("Failed to load / split datasets")?;

    let dataloader_train = DataLoaderBuilder::new(batcher_train.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_ds);

    let dataloader_val: std::sync::Arc<
        dyn DataLoader<B::InnerBackend, ImageBatch<B::InnerBackend>>,
    > = DataLoaderBuilder::new(batcher_eval.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed + 1)
        .num_workers(config.num_workers)
        .build(val_ds);

    let mut optim = config.optimizer.init();
    let mut best_val_accuracy = 0.0;
    let mut patience_counter = 0;
    let mut current_lr = config.learning_rate;

    println!(
        "Starting head-only training with {} epochs",
        config.num_epochs
    );
    println!("Initial learning rate: {:.6}", current_lr);

    for epoch in 1..config.num_epochs + 1 {
        model = model.to_device(&device);
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut batch_count = 0;

        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward_head_only(batch.images);
            let mut loss = CrossEntropyLoss::new(None, &device);
            loss.smoothing = Some(config.label_smoothing);
            let loss = loss.forward(output.clone(), batch.labels.clone());
            let accuracy = accuracy(output.clone(), batch.labels.clone());

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(current_lr, model.clone(), grads);

            if iteration % 75 == 0 {
                println!(
                    "[Train - Epoch {} - Iteration {}] Loss {:.4} | Accuracy {:.2}%",
                    epoch,
                    iteration,
                    loss.clone().into_scalar(),
                    accuracy.clone()
                );
            }

            total_loss += loss.clone().into_scalar().elem::<f32>();
            total_accuracy += accuracy;
            batch_count += 1;
        }

        let train_loss = total_loss / batch_count as f32;
        let train_acc = total_accuracy / batch_count as f32;

        let model_valid = model.valid();

        let (val_loss, val_acc) =
            validate_epoch(&model_valid, &dataloader_val).wrap_err("Validation epoch failed")?;

        println!(
            "Epoch {}: Train Loss {:.4}, Train Acc {:.2}% | Val Loss {:.4}, Val Acc {:.2}%",
            epoch, train_loss, train_acc, val_loss, val_acc
        );

        if epoch % config.lr_step_size == 0 {
            current_lr *= config.lr_gamma;
            println!("Learning rate decayed to: {:.6}", current_lr);
        }

        // Early stopping
        if val_acc > best_val_accuracy {
            best_val_accuracy = val_acc;
            patience_counter = 0;

            // Save best model
            let recorder = CompactRecorder::new();
            model
                .clone()
                .save_file(format!("{artifact_dir}/best_model"), &recorder)
                .wrap_err("Failed saving best_model checkpoint")?;
        } else {
            patience_counter += 1;
            if patience_counter >= config.early_stopping_patience {
                println!(
                    "Early stopping triggered at epoch {} (best val acc: {:.2}%)",
                    epoch, best_val_accuracy
                );
                break;
            }
        }

        // Checkpoint save
        if epoch % 5 == 0 {
            let recorder = CompactRecorder::new();
            model
                .clone()
                .save_file(format!("{artifact_dir}/model_epoch_{epoch}"), &recorder)
                .wrap_err_with(|| format!("Failed saving checkpoint for epoch {epoch}"))?;
        }
    }

    println!(
        "Training completed. Best validation accuracy: {:.2}%",
        best_val_accuracy
    );
    Ok(())
}
