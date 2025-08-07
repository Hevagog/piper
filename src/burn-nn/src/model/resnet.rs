use crate::model::blocks::LayerBlock;
use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct ResNet50<B: Backend> {
    conv1: Conv2d<B>,
    norm1: BatchNorm<B, 2>,
    relu: Relu,
    maxpool: MaxPool2d,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> ResNet50<B> {
    pub fn new(
        blocks: [usize; 4],
        num_classes: usize,
        expansion: usize,
        device: &Device<B>,
    ) -> Self {
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false)
            .init(device);
        let norm1 = BatchNormConfig::new(64).init(device);
        let relu = Relu::new();
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        // Initialize Residual blocks
        let layer1 = LayerBlock::new(blocks[0], 64, 64 * expansion, 1, device);
        let layer2 = LayerBlock::new(blocks[1], 64 * expansion, 128 * expansion, 2, device);
        let layer3 = LayerBlock::new(blocks[2], 128 * expansion, 256 * expansion, 2, device);
        let layer4 = LayerBlock::new(blocks[3], 256 * expansion, 512 * expansion, 2, device);

        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let fc = LinearConfig::new(512 * expansion, num_classes).init(device);

        ResNet50 {
            conv1,
            norm1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            relu,
            avgpool,
            fc,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = self.norm1.forward(x);
        let x = self.relu.forward(x);
        let x = self.maxpool.forward(x);

        let x = self.layer1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.layer3.forward(x);
        let x = self.layer4.forward(x);

        let x = self.avgpool.forward(x);
        let x = x.flatten(1, 3);
        self.fc.forward(x)
    }
    pub fn load(
        mut self,
        record: ResNet50Record<B>,
        device: &Device<B>,
        num_classes: usize,
    ) -> Self {
        self = self.load_record(record);
        self.fc = LinearConfig::new(512 * 4, num_classes).init(device);
        self
    }
}

impl<B: Backend> ResNet50<B> {
    pub fn resnet50(num_classes: usize, device: &Device<B>) -> Self {
        Self::new([3, 4, 6, 3], num_classes, 4, device)
    }
}
