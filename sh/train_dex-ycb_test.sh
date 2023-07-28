        nohup\
        python traineval.py --HO3D_root /data1/zhifeng/ho3d_v2 \
        --host_folder  host_folder/check_dexycb \
        --dex_ycb_root /data1/zhifeng/dex-ycb \
        --epochs 40 \
        --inp_res 256 \
        --lr 1e-4 \
        --train_batch 64 \
        --mano_lambda_regulshape 0 \
        --mano_lambda_regulpose  0 \
        --lr_decay_gamma 0.7 \
        --lr_decay_step 5 \
        --test_batch 64 \
        --evaluate \
        --resume /data1/zhifeng/checkpoint_dexycb.pth.tar\
        > train_check_test.log 2>&1 &

        #        CUDA_VISIBLE_DEVICES=0,1,3,4\
        
