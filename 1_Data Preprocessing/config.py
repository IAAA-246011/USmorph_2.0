pix_x=100 #图像宽，需要改
pix_y=100  #图像长，需要改  
the_dim=pix_x*pix_y
learn_rate=3e-4 #最好不改
checkpoint_path="ckpt/"
logging_name="log_cae"
maxstep=5000000  #最好不改
batch_size=8  #可以根据实际情况修改
step_show=10 #最好不改
step_save=50 #最好不改
decay_steps = 2000 #最好不改
path = "fit_img" #最好不改
encode_structure=[{"cov":[5, 5, 1, 16],#first_layer
                  "cov_stride":[1,1,1,1],#最好不改
                  "cov_padding":"SAME",#最好不改
                  "pool":[1, 2, 2, 1],#最好不改
                  "pool_stride":[1, 2, 2, 1],#最好不改
                  "pool_padding":"SAME",
                  },
                  {"cov":[5, 5, 16, 16],#second_layer
                  "cov_stride":[1,1,1,1],
                  "cov_padding":"SAME",
                  "pool":[1, 2, 2, 1],#最好不改
                  "pool_stride":[1, 2, 2, 1],#最好不改
                  "pool_padding":"SAME",
                  },
                  ]    
structure=[{"encode":[]
}]

















