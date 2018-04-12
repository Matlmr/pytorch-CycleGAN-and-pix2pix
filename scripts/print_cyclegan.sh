python train.py --dataroot ./datasets/synt2real --name synt2real_crop --model cycle_gan --pool_size 50 --batchSize 3 --no_dropout --display_id 0 --gpu_ids 0,2,3  --crop_mask False
python test.py --dataroot ./datasets/synt2real --name synt2real_crop --model cycle_gan --phase test --no_dropout --my_html True
