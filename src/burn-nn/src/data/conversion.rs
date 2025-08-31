use crate::common::{CHANNELS, HEIGHT, WIDTH};

use burn::{
    data::dataset::vision::{Annotation, PixelDepth},
    prelude::*,
    tensor::Tensor,
};

pub fn convert_dset_item_to_tensor<B: Backend>(
    item: &Vec<PixelDepth>,
    device: &B::Device,
) -> Tensor<B, 4> {
    // (Kept infallible for performance; non-u8 entries are skipped as before.)
    let pixels: Vec<f32> = item
        .iter()
        .filter_map(|p| match p {
            PixelDepth::U8(val) => Some(*val as f32 / 255.0),
            _ => None,
        })
        .collect();
    Tensor::<B, 3>::from_data(
        TensorData::new(pixels, [CHANNELS, HEIGHT, WIDTH]).convert::<B::FloatElem>(),
        device,
    )
    .reshape([1, CHANNELS, HEIGHT, WIDTH])
}

pub fn convert_image_to_tensor<B: Backend>(
    img: &imageproc::definitions::Image<image::Rgb<u8>>,
    device: &B::Device,
) -> Tensor<B, 4> {
    assert_eq!(
        img.width() as usize,
        WIDTH,
        "Unexpected width: {} != {}",
        img.width(),
        WIDTH
    );
    assert_eq!(
        img.height() as usize,
        HEIGHT,
        "Unexpected height: {} != {}",
        img.height(),
        HEIGHT
    );

    let hw = HEIGHT * WIDTH;
    let mut buf = vec![0f32; CHANNELS * hw];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let p = img.get_pixel(x as u32, y as u32).0;
            let idx = y * WIDTH + x;
            buf[idx] = p[0] as f32 / 255.0;
            buf[hw + idx] = p[1] as f32 / 255.0;
            buf[2 * hw + idx] = p[2] as f32 / 255.0;
        }
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(buf, [CHANNELS, HEIGHT, WIDTH]).convert::<B::FloatElem>(),
        device,
    )
    .reshape([1, CHANNELS, HEIGHT, WIDTH])
}

pub fn convert_dset_annotation_to_label<B: Backend>(
    annotation: &Annotation,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let id = match annotation {
        Annotation::Label(idx) => Some(*idx as i32),
        _ => None,
    };
    if let Some(label) = id {
        Tensor::from_data(
            TensorData::new(vec![label], [1]).convert::<B::IntElem>(),
            device,
        )
    } else {
        Tensor::from_data(
            TensorData::new(vec![-1], [1]).convert::<B::IntElem>(),
            device,
        )
    }
}
