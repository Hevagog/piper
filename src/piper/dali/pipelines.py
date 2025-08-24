from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def create_train_pipeline(
    file_list,
    width,
    height,
    augmentations,
    seed_,
):
    images, labels = fn.readers.file(
        file_list=file_list,
        lazy_init=True,
        pad_last_batch=False,
        seed=seed_,
        random_shuffle=True,
    )
    # TODO: Use tensor_init_bytes to hint for how much memory to allocate per image.
    images = fn.decoders.image(images, output_type=types.RGB)

    if augmentations:
        for aug in augmentations:
            if aug == "flip":
                direction = fn.random.coin_flip(shape=2)
                images = fn.flip(images, horizontal=direction[0], vertical=direction[1])
            elif aug == "rotate":
                rotation_angle = fn.random.uniform(range=(-30.0, 30.0))
                images = fn.rotate(
                    images,
                    angle=rotation_angle,
                    fill_value=0,
                    interp_type=types.INTERP_LINEAR,
                )
            elif aug == "brightness":
                brightness_factor = fn.random.uniform(range=(0.8, 1.2))
                images = fn.brightness(images, brightness=brightness_factor)
            elif aug == "contrast":
                contrast_factor = fn.random.uniform(range=(0.8, 1.2))
                images = fn.contrast(images, contrast=contrast_factor)
            elif aug == "sat":
                saturation_factor = fn.random.uniform(range=(0.8, 1.2))
                images = fn.saturation(images, saturation=saturation_factor)
            elif aug == "hue":
                hue_factor = fn.random.uniform(range=(-0.1, 0.1))
                images = fn.hue(images, hue=hue_factor)
            elif aug == "blur":
                sigma = fn.random.uniform(range=(0.0, 5.0))
                images = fn.gaussian_blur(images, sigma=sigma)
    images = fn.resize(
        images, resize_x=width, resize_y=height, interp_type=types.INTERP_LINEAR
    )
    return images, labels


@pipeline_def
def create_val_pipeline(
    file_list,
    width,
    height,
    seed_,
):
    images, labels = fn.readers.file(
        file_list=file_list,
        lazy_init=True,
        pad_last_batch=False,
        seed=seed_,
        random_shuffle=True,
    )
    # TODO: Use tensor_init_bytes to hint for how much memory to allocate per image.
    images = fn.decoders.image(images, output_type=types.RGB)
    images = fn.resize(
        images, resize_x=width, resize_y=height, interp_type=types.INTERP_LINEAR
    )
    return images, labels
