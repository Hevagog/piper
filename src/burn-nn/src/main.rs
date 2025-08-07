mod common;
mod data;
mod model;
mod utils;
use std::path::Path;

use crate::model::resnet::{ResNet50, ResNet50Record};
use burn::{module::Module, tensor::Tensor};
use image::{DynamicImage, ImageBuffer};
use res_sat::data::normalize::normalize;

use {
    burn::{
        backend::{Autodiff, cuda::Cuda, cuda::CudaDevice},
        data::dataloader::{DataLoaderBuilder, Dataset},
        prelude::*,
        record::{FullPrecisionSettings, Recorder, RecorderError},
    },
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};
const DATASET_PATH: &str = "data/resnet50-weights.pth";

fn main() {
    let weights_path = Path::new(DATASET_PATH);
    if !weights_path.exists() {
        eprintln!("Error: resnet50-weights.pth file not found in the data folder.");
        std::process::exit(1);
    }

    type Backend = Cuda<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    let device = CudaDevice::default();

    let model = ResNet50::<AutodiffBackend>::resnet50(30, &device);

    let load_args = LoadArgs::new(DATASET_PATH.into())
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

    // Initialize the PyTorch file recorder and load weights
    let record: ResNet50Record<AutodiffBackend> =
        PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

    let model = model.load(record, &device, 30);

    print!("Model loaded successfully.\n");
    let artifact_dir = "/tmp/resnet50_artifacts";
    model::training::train_head_only(artifact_dir, model, device);

    let batcher = data::ImageBatcher::default();
    let dataloader_train: std::sync::Arc<
        dyn burn::data::dataloader::DataLoader<Backend, data::ImageBatch<Backend>>,
    > = DataLoaderBuilder::new(batcher.clone())
        .batch_size(256)
        .shuffle(1)
        .num_workers(2)
        .build(data::load_dataset());

    // model.forward(images);
}

// fn sample() {
//     let dataset = data::load_dataset();
//     let dataset_item = data::convert_u8_image(&dataset.get(2137).unwrap())
//         .unwrap()
//         .into_rgb8();
//     let im_tensor = Tensor::<Backend, 4>::from_data(
//         TensorData::new(
//             dataset_item.into_vec(),
//             [1, common::CHANNELS, common::WIDTH, common::HEIGHT],
//         )
//         .convert::<f32>(),
//         &device,
//     );
//     let im_tensor = normalize(
//         im_tensor,
//         &res_sat::data::normalize::NormalizeConfig::default(),
//     );
//     let out = model.forward(im_tensor.clone());
//     println!("Output shape: {:?}", out.shape());
//     let (score, idx) = out.max_dim_with_indices(1);
//     let idx = idx.into_scalar() as usize;

//     println!(
//         "Predicted: {}\nCategory Id: {}\nScore: {:.4}",
//         CLASSES[idx],
//         idx,
//         score.into_scalar()
//     );
// }
