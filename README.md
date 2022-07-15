maskcyclegan
===================
### use data_preprocessing/datapreoprocess.py to process wav/mat file in data/ to pickle file
### use mask_cyclegan_vc/train.py to train cyclegan model
###
    python -W ignore::UserWarning -m mask_cyclegan_vc.train
    --name mask_cyclegan_lofar_1_3 --seed 0 --save_dir ../results/ --preprocessed_data_dir ../data_preprocessing/data_preprocessed/time_training/ --boat_A_id 1\
    --boat_B_id 3 --epochs_per_save 20 --epochs_per_plot 10 --num_epochs 420 --batch_size 64 --decay_after 1e4 --num_frames 128 --max_mask_len 25 --gpu_ids 0
### use mask_cyclegan_vc/test.py to generate lofar file with saved cyclegan model
###
    python -W ignore::UserWarning -m mask_cyclegan_vc.test --name mask_cyclegan_lofar_1_2 --save_dir ../results/ \
    --preprocessed_data_dir ../data_preprocessing/data_preprocessed/time_training --gpu_ids 0 --boat_A_id 1 --boat_B_id 2 \
    --ckpt_dir ../results/mask_cyclegan_lofar_1_2/ckpts --load_epoch 120 --model_name generator_A2B
### use classifier/train_torch_mini.py to train a classifier to test for fewshot
###
    python -W ignore::UserWarning -m classifier.train_torch_mini --lr 1e-4 --batch_size 256 --num_epochs 100 --weight_decay 1e-4 --model_choice resnet18 \
    --num_class 3 --datadir data/pickle_ori --expand
![image](https://github.com/tualgfhite/maskcyclegan/blob/master/results/acc_ori.png)

![image](https://github.com/tualgfhite/maskcyclegan/blob/master/results/acc_expand.png)
![image](https://github.com/tualgfhite/maskcyclegan/blob/master/results/mini_acc_ori.png)
