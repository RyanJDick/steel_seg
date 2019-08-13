import argparse

from dataset import SeverstalSteelDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert raw severstal steel data to tfrecord files.')
    parser.add_argument('--config', required=True, help='Path to .yaml config file.')
    args = parser.parse_args()

    dataset = SeverstalSteelDataset.init_from_config(args.config)
    dataset.create_tfrecords()
