mod common;
mod data;
mod model;
mod utils;
use color_eyre::Result;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    colog::init();
    let paths = utils::app_paths::AppPaths::from_env();
    model::training::training_loop(&paths)?;

    // let terminal = tui::init();
    // let result = tui::app::Visualizer::new().run(&mut terminal.unwrap());

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
