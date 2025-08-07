use burn::{prelude::*, tensor::Tensor};

pub struct NormalizeConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        NormalizeConfig {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

pub fn normalize<B: Backend>(tensor: Tensor<B, 4>, config: &NormalizeConfig) -> Tensor<B, 4> {
    let mean = Tensor::from_data(
        TensorData::new(config.mean.to_vec(), [1, 3, 1, 1]).convert::<B::FloatElem>(),
        &tensor.device(),
    );
    let std = Tensor::from_data(
        TensorData::new(config.std.to_vec(), [1, 3, 1, 1]).convert::<B::FloatElem>(),
        &tensor.device(),
    );

    (tensor - mean) / std
}
