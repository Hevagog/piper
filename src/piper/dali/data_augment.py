import argparse
import os
import hashlib

from pipelines import create_train_pipeline, create_val_pipeline


class Pipeline:
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        input_dir,
        width,
        height,
        split_train_test,
        seed,
        batch_size,
        num_threads,
        augmentations,
    ):
        self.input_dir = input_dir
        self.train_list = os.path.join(input_dir, "train_list.txt")
        self.val_list = os.path.join(input_dir, "val_list.txt")
        self.width = width
        self.height = height
        self.split_train_test = split_train_test
        self.augmentations = augmentations
        self.seed = seed
        self.batch_size = batch_size
        self.num_threads = num_threads

        self.verify_file_list()

    def verify_file_list(self):
        """Check if train/test file lists exist. If not, create them with split defined by `self.split_train_test`.

        Uses a deterministic, memory-efficient hashing strategy to assign each image to train/val
        without loading all paths into memory or doing a full shuffle.
        The output file format is `path/to/image <label_index>`.
        """
        train_needed = not os.path.exists(self.train_list)
        val_needed = not os.path.exists(self.val_list)

        if not (train_needed or val_needed):
            return

        # Discover classes (subdirectories) and create a mapping to integer indices
        class_names = sorted([d.name for d in os.scandir(self.input_dir) if d.is_dir()])
        class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        total = 0
        train_count = 0
        val_count = 0

        train_f = open(self.train_list, "w") if train_needed else None
        val_f = open(self.val_list, "w") if val_needed else None

        def iter_images():
            for class_name in class_names:
                class_path = os.path.join(self.input_dir, class_name)
                with os.scandir(class_path) as it:
                    for ent in it:
                        if ent.is_file():
                            yield ent.path, class_name

        try:
            for img_path, class_name in iter_images():
                _, ext = os.path.splitext(img_path)
                if ext.lower() not in self.allowed_exts:
                    continue
                total += 1

                # Deterministic assignment using hash of path + seed
                h = hashlib.md5(
                    (
                        os.path.relpath(img_path, self.input_dir) + str(self.seed)
                    ).encode()
                )
                fraction = int.from_bytes(h.digest(), "little") / float(2**128 - 1)

                rel_path = os.path.relpath(img_path, self.input_dir)
                label = class_to_idx[class_name]
                line_to_write = f"{rel_path} {label}\n"

                if fraction < float(self.split_train_test):
                    if train_f:
                        train_f.write(line_to_write)
                    train_count += 1
                else:
                    if val_f:
                        val_f.write(line_to_write)
                    val_count += 1
        finally:
            if train_f:
                train_f.close()
            if val_f:
                val_f.close()

        print(
            f"Found {total} images in {len(class_names)} classes under {self.input_dir} -> train: {train_count}, val: {val_count}"
        )

    def build(self):
        self.train_pipeline = create_train_pipeline(
            file_list=self.train_list,
            width=self.width,
            height=self.height,
            augmentations=self.augmentations,
            seed_=self.seed,
            num_threads=self.num_threads,
            batch_size=self.batch_size,
        )
        self.val_pipeline = create_val_pipeline(
            file_list=self.val_list,
            width=self.width,
            height=self.height,
            seed_=self.seed,
            num_threads=self.num_threads,
            batch_size=self.batch_size,
        )
        self.train_pipeline.build()
        self.val_pipeline.build()

    def run_train(self):
        return self.train_pipeline.run()

    def run_val(self):
        return self.val_pipeline.run()

    def get_pytorch_train_data_loader(self):
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator

        return DALIClassificationIterator([self.train_pipeline], auto_reset=True)

    def get_pytorch_val_data_loader(self):
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator

        return DALIClassificationIterator([self.val_pipeline], auto_reset=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--split_train_test", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_threads", type=int, default=12)
    parser.add_argument("--augmentations")

    args = parser.parse_args()
    pipd = Pipeline(
        input_dir=args.input_dir,
        width=args.width,
        height=args.height,
        split_train_test=args.split_train_test,
        seed=args.seed,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        augmentations=args.augmentations.split(",") if args.augmentations else [],
    )
    pipd.build()
