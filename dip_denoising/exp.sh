# python main.py --dip_type dip    --net_type s2s  --exp_tag single_image  --desc sigma25  denoising   --sigma 25 --eval_data single_image
# python main.py --dip_type dip_sure    --net_type s2s  --exp_tag single_image --desc sigma25  denoising   --sigma 25 --eval_data single_image
# python main.py --dip_type eSURE_uniform --net_type s2s --exp_tag single_image --optim RAdam --force_steplr --desc sigma25   denoising --sigma 25 --eval_data single_image


# model: s2s

# python main.py --dip_type dip_sure --gray   --net_type s2s  --exp_tag test_image --desc s2s_sigma25  denoising   --sigma 25 --eval_data test_image
# python main.py --dip_type eSURE_uniform --gray   --net_type s2s  --exp_tag test_image --desc s2s_sigma25  denoising   --sigma 25 --eval_data test_image

# model Unet
# python main.py --dip_type dip_sure    --net_type UNet  --exp_tag single_image  --desc UNet_sigma25  denoising   --sigma 25 --eval_data single_image
# python main.py --dip_type eSURE_uniform    --net_type UNet  --exp_tag single_image  --desc UNet_sigma25  denoising   --sigma 25 --eval_data single_image

# model s2s1 (resnet) gray

python main.py --dip_type dip_sure    --net_type s2s  --exp_tag single_image  --desc s2s1_sigma25  denoising   --sigma 25 --eval_data single_image
# python main.py --dip_type dip_sure   --gray  --net_type s2s1  --exp_tag gray_image  --desc s2s1_sigma25  denoising   --sigma 25 --eval_data test_image
# python main.py --dip_type eSURE_uniform --gray   --net_type s2s1  --exp_tag gray_image  --desc s2s1_sigma25  denoising   --sigma 25 --eval_data test_image


# model s2s  
# color
# python main.py --dip_type dip_sure    --net_type s2s  --exp_tag single_image  --desc s2s_sigma25  denoising   --sigma 25 --eval_data single_image

# gray
# python main.py --dip_type dip_sure   --gray  --net_type s2s  --exp_tag gray_image  --desc s2s_sigma25  denoising   --sigma 25 --eval_data test_image
# python main.py --dip_type eSURE_uniform --gray   --net_type s2s  --exp_tag gray_image  --desc s2s_sigma25  denoising   --sigma 25 --eval_data test_image

# model s2s_dropout
# python main.py --dip_type dip_sure --net_type s2s_dropout --exp_tag single_image --desc s2sDropout_sigma25 denoising --sigma 25 --eval_data single_image