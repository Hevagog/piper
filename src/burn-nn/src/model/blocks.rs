use burn::{
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: Conv2d<B>,
    norm1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    norm2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    norm3: BatchNorm<B, 2>,
    relu: Relu,
    downsample: Option<DownsampleBlock<B>>,
}

#[derive(Module, Debug)]
pub struct DownsampleBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
}

impl<B: Backend> DownsampleBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init(device);
        let norm = BatchNormConfig::new(out_channels).init(device);

        DownsampleBlock { conv, norm }
    }
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        self.norm.forward(x)
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        device: &Device<B>,
    ) -> Self {
        let inter_out_channels = out_channels / 4;

        let conv1 = Conv2dConfig::new([in_channels, inter_out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init(device);
        let norm1 = BatchNormConfig::new(inter_out_channels).init(device);

        let conv2 = Conv2dConfig::new([inter_out_channels, inter_out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .init(device);
        let norm2 = BatchNormConfig::new(inter_out_channels).init(device);

        let conv3 = Conv2dConfig::new([inter_out_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .init(device);
        let norm3 = BatchNormConfig::new(out_channels).init(device);

        let relu = Relu::new();

        let downsample = {
            if in_channels != out_channels {
                Some(DownsampleBlock::new(
                    in_channels,
                    out_channels,
                    stride,
                    device,
                ))
            } else {
                None
            }
        };

        ConvBlock {
            conv1,
            norm1,
            conv2,
            norm2,
            conv3,
            norm3,
            relu,
            downsample,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        let x = self.conv1.forward(input);
        let x = self.norm1.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv3.forward(x);
        let x = self.norm3.forward(x);

        // Skip connection
        let x = {
            match &self.downsample {
                Some(downsample) => x + downsample.forward(identity),
                None => x + identity,
            }
        };

        self.relu.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    blocks: Vec<ConvBlock<B>>,
}

impl<B: Backend> LayerBlock<B> {
    pub fn new(
        num_blocks: usize,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        device: &Device<B>,
    ) -> Self {
        let blocks = (0..num_blocks)
            .map(|b| {
                let stride = if b == 0 { stride } else { 1 };
                ConvBlock::init(in_channels, out_channels, stride, device)
            })
            .collect();

        LayerBlock { blocks }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = input;

        for block in &self.blocks {
            x = block.forward(x);
        }

        x
    }
}
