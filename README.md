# Aim of the repo:
To see how simple an implementation I could make to generate some output out of SOLOv2 on some relatively simple data.

# Run Training and Inference
With default settings: \
Training: `python train.py` \
Inference: `python inference.py --epoch 10 --data_path /home/localssk23/SOLOv2/datasets/dummy_2d/test_imgs`

# To build dummy data
With default settings: \
run `python /datasets/dummy_2d/make_dummydata_and_json.py`

# To Dos
Current implementation does generate some output. I need to make one proper training run to see if it is actually possible to run something or I have revert to the original implementation.

- [ ] Compare with mmDet?


# Initial Changes made to get the [reference repo](https://github.com/OpenFirework/pytorch_solov2) working.
FOR Evaluation
1. Comment out line 9 and change the loss function from the minimal implementation in solov2_head.py
2. change the loss in line 310 after commenting out the init loss in solov2_head.py
3. Change np.bool to just bool in eval.py in line 188
4. Change the save path in eval.py and the save=True

FOR Training
1. in mask_feat_head.py, Make the following change:
            # feature_add_all_level += self.convs_all_levels[i](input_p)
            feature_add_all_level = self.convs_all_levels[i](input_p) + feature_add_all_level