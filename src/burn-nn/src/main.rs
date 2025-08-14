mod common;
mod data;
mod model;
mod utils;
use crate::model::resnet::{ResNet50, ResNet50Record};
use color_eyre::{
    eyre::{bail, WrapErr},
    Result,
};
use std::{error::Error, io, path::Path};
use {
    burn::{
        backend::{cuda::Cuda, cuda::CudaDevice, Autodiff},
        data::dataloader::DataLoaderBuilder,
        record::{FullPrecisionSettings, Recorder},
    },
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

struct AppPaths {
    weights_path: String,
    dataset_root: String,
    artifact_dir: String,
}

impl AppPaths {
    fn from_env() -> Self {
        Self {
            weights_path: std::env::var("WEIGHTS_PATH")
                .unwrap_or_else(|_| "/data/resnet50-weights.pth".into()),
            dataset_root: std::env::var("DATASET_ROOT")
                .unwrap_or_else(|_| "/data/processed".into()),
            artifact_dir: std::env::var("ARTIFACT_DIR")
                .unwrap_or_else(|_| "/tmp/resnet50_artifacts".into()),
        }
    }
}

fn training_loop(paths: &AppPaths) -> Result<()> {
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

    println!("Model loaded successfully.");
    model::training::train_head_only(&paths.artifact_dir, &paths.dataset_root, model, device)
        .wrap_err("Head-only training failed")?;

    // Example (commented) future usage:
    // let ds = data::load_dataset(&paths.dataset_root)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths = AppPaths::from_env();
    training_loop(&paths)?;
    Ok(())
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
