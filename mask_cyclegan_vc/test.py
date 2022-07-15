import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from args.cycleGAN_test_arg_parser import CycleGANTestArgParser
from dataset.vc_dataset import VCDataset
from mask_cyclegan_vc.model import Generator
from saver.model_saver import ModelSaver


class MaskCycleGANVCTesting(object):
    """Tester for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store Args
        self.device = args.device
        self.converted_audio_dir = os.path.join(args.save_dir, args.name, 'converted_audio')
        os.makedirs(self.converted_audio_dir, exist_ok=True)
        self.model_name = args.model_name

        self.boat_A_id = args.boat_A_id
        self.boat_B_id = args.boat_B_id

        # Initialize boatA's dataset
        self.dataset_A = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.boat_A_id, f"{self.boat_A_id}_normalized.pickle"))
        dataset_A_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.boat_A_id, f"{self.boat_A_id}_norm_stat.npz"))
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']

        # Initialize boatB's dataset
        self.dataset_B = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.boat_B_id, f"{self.boat_B_id}_normalized.pickle"))
        dataset_B_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.boat_B_id, f"{self.boat_B_id}_norm_stat.npz"))
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        source_dataset = self.dataset_A if self.model_name == 'generator_A2B' else self.dataset_B
        self.dataset = VCDataset(datasetA=source_dataset,
                                 datasetB=None,
                                 valid=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           drop_last=False)

        # Generator
        self.generator = Generator().to(self.device)
        self.generator.eval()

        # Load Generator from ckpt
        self.saver = ModelSaver(args)
        self.saver.load_model(self.generator, self.model_name)

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def test(self):
        data_ori = np.ones([1, 128, 128])
        data_gen = np.ones([1, 128, 128])
        for i, sample in enumerate(tqdm(self.test_dataloader)):
            if self.model_name == 'generator_A2B':
                real_A = sample
                real_A = real_A.to(self.device, dtype=torch.float)
                fake_B = self.generator(real_A, torch.ones_like(real_A))
                ##这里本该用B的值，但是没有，有待进一步思考
                data_gen = np.vstack((data_gen, fake_B.cpu().detach().numpy()))
                data_ori = np.vstack((data_ori, real_A.cpu().detach().numpy()))
            else:
                real_B = sample
                real_B = real_B.to(self.device, dtype=torch.float)
                fake_A = self.generator(real_B, torch.ones_like(real_B))
                data_gen = np.vstack((data_gen, fake_A.cpu().detach().numpy()))
                data_ori = np.vstack((data_ori, real_B.cpu().detach().numpy()))
        if self.model_name == 'generator_A2B':
            save_path_gen = open(
                os.path.join(self.converted_audio_dir, f"converted_{self.boat_A_id}_to_{self.boat_B_id}.pickle"), 'wb')
            save_path_ori = open(os.path.join(self.converted_audio_dir,
                                              f"original_{self.boat_A_id}_to_{self.boat_B_id}.pickle"), 'wb')
        else:
            save_path_gen = open(
                os.path.join(self.converted_audio_dir, f"converted_{self.boat_B_id}_to_{self.boat_A_id}.pickle"), 'wb')
            save_path_ori = open(os.path.join(self.converted_audio_dir,
                                              f"original_{self.boat_B_id}_to_{self.boat_A_id}.pickle"), 'wb')
        pickle.dump(data_gen, save_path_gen)
        pickle.dump(data_ori, save_path_ori)


if __name__ == "__main__":
    parser = CycleGANTestArgParser()
    args = parser.parse_args()
    tester = MaskCycleGANVCTesting(args)
    tester.test()
