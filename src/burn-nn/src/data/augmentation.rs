use crate::common::{CHANNELS, HEIGHT, WIDTH};
use burn::data::dataset::vision::PixelDepth;
use image::{ImageBuffer, Rgb};
use imageproc::noise::gaussian_noise;
use imageproc::{
    definitions::Image,
    geometric_transformations::{Interpolation, rotate_about_center},
};
use rand::prelude::*;
use std::f32::consts::PI;

/// Converts a vector of PixelDepth to an imageproc::Image<Rgb<u8>>.
/// Assumes input is in row-major order, 3 channels (RGB), and u8 pixel depth.
pub fn image_dset_to_image(item: &Vec<PixelDepth>) -> Image<Rgb<u8>> {
    assert_eq!(
        item.len(),
        (WIDTH * HEIGHT * CHANNELS) as usize,
        "Input length mismatch"
    );
    let mut buf = Vec::with_capacity(item.len());
    for p in item {
        match p {
            PixelDepth::U8(val) => buf.push(*val),
            _ => panic!("Non-u8 PixelDepth encountered"),
        }
    }
    let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_vec(WIDTH as u32, HEIGHT as u32, buf)
            .expect("Failed to create ImageBuffer");

    Image::from(img_buf)
}

#[derive(Clone)]
pub struct AugmentationConfig {
    /// Rotation range in degrees
    pub rotation_range: i16,
    /// Contrast adjustment range: (min, max multiplier)
    pub contrast_range: (f32, f32),
    /// Brightness adjustment range: (min, max delta)
    pub brightness_range: (f32, f32),
    /// Random erasing probability
    pub erasing_prob: f32,
    /// Random erasing area: (min, max ratio)
    pub erasing_area: (f32, f32),
    /// Hue shift range in degrees
    pub hue_shift_range: f32,
    /// Saturation adjustment range: (min, max multiplier)
    pub saturation_range: (f32, f32),
    /// Probability of applying Gaussian noise
    pub gaussian_prob: f32,
    /// Gaussian noise standard deviation
    pub gaussian_noise_std: f64,
    /// Gaussian noise mean
    pub gaussian_noise_mean: f64,
    /// Probability of flipping
    pub flip_prob: f32,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        AugmentationConfig {
            rotation_range: 360,
            contrast_range: (0.8, 1.2),
            brightness_range: (-0.1, 0.1),
            erasing_prob: 0.5,
            erasing_area: (0.02, 0.4),
            hue_shift_range: 20.0,
            saturation_range: (0.8, 1.2),
            gaussian_prob: 0.5,
            gaussian_noise_std: 0.01,
            gaussian_noise_mean: 0.0,
            flip_prob: 0.5,
        }
    }
}

#[derive(Clone)]
pub struct ImageAugmenter {
    config: AugmentationConfig,
}

impl ImageAugmenter {
    pub fn new(config: AugmentationConfig) -> Self {
        ImageAugmenter { config }
    }
    pub fn augment(&self, img: &Image<Rgb<u8>>) -> Image<Rgb<u8>> {
        let mut out_img = img.clone();
        out_img = self.rotate_image(&out_img);
        // out_img = self.flip_image(&out_img); // Uncomment if flip is needed
        out_img = self.noise_image(&out_img);
        out_img = self.random_erasing(&out_img);
        out_img = self.random_contrast_brightness(&out_img);
        out_img
    }

    fn rotate_image(&self, img: &Image<Rgb<u8>>) -> Image<Rgb<u8>> {
        let mut rng = rand::rng();
        let rotation_angle: i16 =
            rng.random_range(-self.config.rotation_range..=self.config.rotation_range);
        let theta: f32 = rotation_angle as f32 * PI / 180.0;
        rotate_about_center(img, theta, Interpolation::Bilinear, Rgb([0, 0, 0]))
    }
    // fn flip_image(&self, img: &Image<Rgb<u8>>) -> Image<Rgb<u8>> {
    //     let mut rng = rand::rng();
    //     if rng.random::<f32>() < self.config.flip_prob {
    //         img.flipped()
    //     } else {
    //         img.clone()
    //     }
    // }
    fn noise_image(&self, img: &Image<Rgb<u8>>) -> Image<Rgb<u8>> {
        let mut rng = rand::rng();
        let seed = rng.random::<u64>();
        if rng.random::<f32>() < self.config.gaussian_prob {
            gaussian_noise(
                img,
                self.config.gaussian_noise_mean,
                self.config.gaussian_noise_std,
                seed,
            )
        } else {
            img.clone()
        }
    }

    /// Random erasing (cutout): with probability erasing_prob, choose a rectangle of random
    /// area within erasing_area ratio of image area and set to either random color or mean color.
    fn random_erasing(&self, img: &Image<Rgb<u8>>) -> Image<Rgb<u8>> {
        let mut rng = rand::rng();
        if rng.random::<f32>() >= self.config.erasing_prob {
            return img.clone();
        }

        let (w, h) = (img.width() as f32, img.height() as f32);
        let img_area = w * h;

        let area_ratio = rng.random_range(self.config.erasing_area.0..=self.config.erasing_area.1);
        let target_area = area_ratio * img_area;

        let aspect_ratio = rng.random_range(0.3f32..=3.3f32);

        // Width and height of the cutout
        let cut_w = (target_area * aspect_ratio).sqrt().round().max(1.0) as u32;
        let cut_h = (target_area / aspect_ratio).sqrt().round().max(1.0) as u32;

        // Make sure we are in bounds of the image
        let cut_w = cut_w.min(img.width());
        let cut_h = cut_h.min(img.height());

        // Get the top-left corner of the cutout
        let x0 = rng.random_range(0..=(img.width() - cut_w));
        let y0 = rng.random_range(0..=(img.height() - cut_h));

        // 50% chance to use random color, otherwise use mean color
        let use_random_color = rng.random::<f32>() < 0.5;
        let fill_color = if use_random_color {
            Rgb([rng.random(), rng.random(), rng.random()])
        } else {
            let mut sum = [0u64; 3];
            for p in img.pixels() {
                sum[0] += p[0] as u64;
                sum[1] += p[1] as u64;
                sum[2] += p[2] as u64;
            }
            let total = (img.width() as u64) * (img.height() as u64);
            Rgb([
                (sum[0] / total) as u8,
                (sum[1] / total) as u8,
                (sum[2] / total) as u8,
            ])
        };

        let mut out_buf = img.clone();
        for yy in y0..(y0 + cut_h) {
            for xx in x0..(x0 + cut_w) {
                out_buf.put_pixel(xx, yy, fill_color);
            }
        }

        Image::from(out_buf)
    }

    /// Random contrast & brightness. Contrast is multiplicative around 128 (midpoint),
    /// brightness is additive delta in range [-1.0,1.0] => converted to -255..255
    fn random_contrast_brightness(&self, img: &Image<Rgb<u8>>) -> Image<Rgb<u8>> {
        let mut rng = rand::rng();
        let (width, height) = (img.width(), img.height());

        let contrast =
            rng.random_range(self.config.contrast_range.0..=self.config.contrast_range.1);
        let brightness_frac =
            rng.random_range(self.config.brightness_range.0..=self.config.brightness_range.1);
        let brightness_delta = brightness_frac * 255.0;

        let mut out_buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::new(width as u32, height as u32);

        for (x, y, pixel) in img.enumerate_pixels() {
            let channels = pixel.0;
            let mut new_pixel = [0u8; 3];
            for c in 0..3 {
                let v = channels[c] as f32;
                // apply contrast around midpoint 128
                let mut vv = (v - 128.0) * contrast + 128.0 + brightness_delta;
                // clamp to 0..255 safely
                if vv < 0.0 {
                    vv = 0.0;
                } else if vv > 255.0 {
                    vv = 255.0;
                }
                new_pixel[c] = vv as u8;
            }
            out_buf.put_pixel(x, y, Rgb(new_pixel));
        }

        Image::from(out_buf)
    }
}
