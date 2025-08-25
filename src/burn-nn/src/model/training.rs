use crate::{
    data::{
        batch::{ImageBatch, ImageBatcher},
        loader::load_train_val_datasets,
    },
    model::{
        resnet::{ResNet50, ResNet50Record},
        valid::validate_epoch,
    },
    utils::{app_paths::AppPaths, metrics::accuracy},
};

use burn::{
    backend::{Autodiff, cuda::Cuda, cuda::CudaDevice},
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::AutodiffModule,
    nn::loss::CrossEntropyLoss,
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    prelude::*,
    record::{CompactRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use color_eyre::{
    Result,
    eyre::{WrapErr, bail},
};
use log::{info, warn};
use std::path::Path;

pub fn training_loop(paths: &AppPaths) -> Result<()> {
    let weights_path = Path::new(&paths.weights_path);
    if !weights_path.exists() {
        bail!(
            "Missing weights file: {:?}. Expected pretrained PyTorch ResNet50 weights.",
            weights_path
        );
    }

    type Backend = Cuda<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    let device = CudaDevice::default();

    let model = ResNet50::<AutodiffBackend>::resnet50(30, &device);

    let load_args = LoadArgs::new(paths.weights_path.clone().into())
        // Map conv1 parameters
        .with_key_remap(r"^conv1\.(.+)$", "conv1.$1")
        // Map top-level batchnorm 'bn1' to 'norm1'
        .with_key_remap(r"^bn1\.(.+)$", "norm1.$1")
        // Map layer blocks convolution parameters
        .with_key_remap(
            r"^layer([1-4])\.(\d+)\.conv([123])\.(.+)$",
            "layer$1.blocks.$2.conv$3.$4",
        )
        // Map layer blocks batchnorm parameters
        .with_key_remap(
            r"^layer([1-4])\.(\d+)\.bn([123])\.(.+)$",
            "layer$1.blocks.$2.norm$3.$4",
        )
        // Map downsample convolution in blocks
        .with_key_remap(
            r"^layer([1-4])\.(\d+)\.downsample\.0\.(.+)$",
            "layer$1.blocks.$2.downsample.conv.$3",
        )
        // Map downsample batchnorm in blocks
        .with_key_remap(
            r"^layer([1-4])\.(\d+)\.downsample\.1\.(.+)$",
            "layer$1.blocks.$2.downsample.norm.$3",
        )
        // Map fully connected layer
        .with_key_remap(r"^fc\.(.+)$", "fc.$1");

    let record: ResNet50Record<AutodiffBackend> =
        PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .wrap_err("Failed to load / map PyTorch ResNet50 state into Burn record")?;

    let model = model.load(record, &device, 30);

    info!("Model loaded successfully.");

    train_head_only(paths, model, device).wrap_err("Head-only training failed")?;

    // Example (commented) future usage:
    // let ds = data::load_dataset(&paths.dataset_root)?;
    Ok(())
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamWConfig,

    #[config(default = 1)]
    pub num_epochs: usize,

    #[config(default = 16)]
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

    #[config(default = 8)]
    pub gradient_accumulation_steps: usize,
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
    paths: &AppPaths,
    mut model: ResNet50<B>,
    device: B::Device,
) -> Result<()> {
    create_artifact_dir(&paths.artifact_dir);

    let optim_config = AdamWConfig::new().with_weight_decay(1.0e-2);
    let config = TrainingConfig::new(optim_config);
    B::seed(config.seed);

    config
        .save(format!("{}/config.json", paths.artifact_dir))
        .wrap_err("Failed to save head-only config JSON")?;

    let batcher_train = ImageBatcher::train_default();
    let batcher_eval = ImageBatcher::eval();

    let (train_ds, val_ds) = load_train_val_datasets(&paths.dataset_root, config.seed, 0.8)
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
    let acc_steps = config.gradient_accumulation_steps;

    info!(
        "Starting head-only training with {} epochs",
        config.num_epochs
    );
    info!("Initial learning rate: {:.6}", current_lr);

    for epoch in 1..config.num_epochs + 1 {
        model = model.to_device(&device);
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut batch_count = 0;

        let mut gradient_accumulator: GradientsAccumulator<ResNet50<B>> =
            GradientsAccumulator::default();

        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward_head_only(batch.images);

            let mut loss = CrossEntropyLoss::new(None, &device);
            loss.smoothing = Some(config.label_smoothing);

            // normalizing loss to account for gradient accumulation
            let loss = loss.forward(output.clone(), batch.labels.clone()) / (acc_steps as f32);

            let accuracy = accuracy(output.clone(), batch.labels.clone());

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            gradient_accumulator.accumulate(&model, grads);

            if (iteration + 1) % acc_steps == 0 {
                let grads_accumulated = gradient_accumulator.grads();
                model = optim.step(current_lr, model.clone(), grads_accumulated);
            }

            if iteration % 75 == 0 {
                info!(
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

        {
            let grads_remaining = gradient_accumulator.grads();
            model = optim.step(current_lr, model.clone(), grads_remaining);
        }

        let train_loss = total_loss / batch_count as f32;
        let train_acc = total_accuracy / batch_count as f32;

        let model_valid = model.valid();

        let (val_loss, val_acc) =
            validate_epoch(&model_valid, &dataloader_val).wrap_err("Validation epoch failed")?;

        info!(
            "Epoch {}: Train Loss {:.4}, Train Acc {:.2}% | Val Loss {:.4}, Val Acc {:.2}%",
            epoch, train_loss, train_acc, val_loss, val_acc
        );

        if epoch % config.lr_step_size == 0 {
            current_lr *= config.lr_gamma;
            info!("Learning rate decayed to: {:.6}", current_lr);
        }

        // Early stopping
        if val_acc > best_val_accuracy {
            best_val_accuracy = val_acc;
            patience_counter = 0;

            // Save best model
            let recorder = CompactRecorder::new();
            model
                .clone()
                .save_file(format!("{}/best_model", paths.artifact_dir), &recorder)
                .wrap_err("Failed saving best_model checkpoint")?;
        } else {
            patience_counter += 1;
            if patience_counter >= config.early_stopping_patience {
                info!(
                    "Early stopping triggered at epoch {} (best val acc: {:.2}%)",
                    epoch, best_val_accuracy
                );
                let model_final_dir = Path::new(&paths.weights_path)
                    .parent()
                    .expect("Failed to get parent directory")
                    .join("trained_model");

                let src = Path::new(&paths.artifact_dir).join("best_model.mpk");

                if src.exists() {
                    std::fs::create_dir_all(&model_final_dir)
                        .wrap_err("Failed to create model final directory")?;

                    let dest = match src.file_name() {
                        Some(name) => model_final_dir.join(name),
                        None => model_final_dir.join("best_model.mpk"),
                    };

                    match std::fs::rename(&src, &dest) {
                        Ok(_) => info!("Best model moved to {:?}", dest),
                        Err(e) => warn!("Failed to move best model: {:?}", e),
                    };
                } else {
                    info!("No best_model found at {:?}, skipping move", src);
                }

                break;
            }
        }

        if epoch % 1 == 0 {
            let recorder = CompactRecorder::new();
            model
                .clone()
                .save_file(
                    format!("{}/model_epoch_{}", paths.artifact_dir, epoch),
                    &recorder,
                )
                .wrap_err_with(|| format!("Failed saving checkpoint for epoch {epoch}"))?;
        }
    }

    let model_final_dir = Path::new(&paths.weights_path)
        .parent()
        .expect("Failed to get parent directory")
        .join("trained_model");

    if !model_final_dir.exists() {
        std::fs::create_dir_all(&model_final_dir)
            .wrap_err("Failed to create model final directory")?;
    }
    let src = Path::new(&paths.artifact_dir).join("best_model.mpk");

    let dest = match src.file_name() {
        Some(name) => model_final_dir.join(name),
        None => model_final_dir.join("best_model.mpk"),
    };

    match std::fs::rename(&src, &dest) {
        Ok(_) => info!("Best model moved to {:?}", dest),
        Err(e) => warn!("Failed to move best model: {:?}", e),
    };

    Ok(())
}
