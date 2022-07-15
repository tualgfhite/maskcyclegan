"""
Arguments for MaskCycleGAN-VC testing.
Inherits BaseArgParser.
"""

from args.base_arg_parser import BaseArgParser

class CycleGANTestArgParser(BaseArgParser):
    """
    Class which implements an argument parser for args used only in training MaskCycleGAN-VC.
    It inherits TrainArgParser.
    """

    def __init__(self):
        super(CycleGANTestArgParser, self).__init__()
        self.parser.add_argument('--sample_rate', type=int, default=16384, help='Sampling rate of mel-spectrograms.')
        self.parser.add_argument(
            '--boat_A_id', type=str, default="2", help='Source boat id.')
        self.parser.add_argument(
            '--boat_B_id', type=str, default="1", help='Source boat id.')
        self.parser.add_argument(
            '--preprocessed_data_dir', type=str, default="data_preprocessing/data_preprocessed/raw_training/", help='Directory containing preprocessed dataset files.')
        self.parser.add_argument(
            '--ckpt_dir', type=str, default=None, help='Path to model ckpt.')
        self.parser.add_argument(
            '--model_name', type=str, choices=('generator_A2B', 'generator_B2A'), default='generator_A2B', help='Name of model to load.')
