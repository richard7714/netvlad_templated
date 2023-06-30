# Hyperparmeter Setting
depth_range = dict(min_depth_eval=0.001,
                   max_depth_eval=15,
                   max_depth=15
                   )

image_size = dict(input_height=288,
                  input_width=512
                  )

train_parm = dict(num_threads=4,
                  batch_size=4,
                  num_epochs=20,
                  checkpoint_path=None,
                  retrain=True,
                  learning_rate=2.5e-4,  
                  )

test_parm = dict(test_checkpoint_path='save/joint.our.joint_id_6_rln_remove_blur1/checkpoints/eh_model-160500-best_uw_ssim_0.46377.pth',
                 is_save_gt_image=False,
                 is_save_input_image=False,
                 )


freq = dict(save_freq=1000,
            log_freq=100,
            eval_freq=1500
            )


etc = dict(is_checkpoint_save=True,
           do_mixed_precison=False,
           do_online_eval=True,
          #  do_use_logger=None,
           do_use_logger='Tensorboard',
          #  do_use_logger='Wandb'
           )

# Log & Save Setting
log_save_cfg = dict(save_root='save',
                    log_directory='log',
                    eval_log_directory='eval',
                    model_save_directory ='checkpoints',
                    wandb_save_path = 'dataset_root/wandb'
                    )


#Dataset
dataset = dict(train_data_path='dataset_root/Dataset/My_UW/fake_images/diml_fake_for_RAL/train',
               train_depth_gt_path='dataset_root/Dataset/DIML/indoor/train',
               train_enhanced_gt_path='dataset_root/Dataset/DIML/indoor/train',
               
               eval_data_path='dataset_root/Dataset/My_UW/fake_images/diml_fake_for_RAL/test',
               eval_depth_gt_path='dataset_root/Dataset/DIML/indoor/test',
               eval_enhanced_gt_path='dataset_root/Dataset/DIML/indoor/test',
               
               test_data_path='dataset_root/Dataset/Enhancement/Underwater/현대과제/230310/stag_2',
               test_depth_gt_path='dataset_root/Dataset/Enhancement/Underwater/SeaThru',
               test_enhanced_gt_path='dataset_root/Dataset/Enhancement/Underwater/SeaThru',

               sample_eval_data_path='result_joint.diml.jointformer_rln_final/enhanced_output',
               sample_eval_gt_path='dataset_root/Dataset/Enhancement/Underwater/EUVP/test_samples/GTr',
               
               train_txt_file='diml_fake_train_ral.txt',
               eval_txt_file='diml_fake_test_ral.txt',
               test_txt_file='hyundai_stag_2_input_dataset.txt'
               )




# Basic Setting
basic_cfg = dict(model_cfg = dict(type='Build_Structure',
                                  img_size=(image_size['input_height'], image_size['input_width']),
                                  structure_cfg = dict(type='Joint_ID_Structure',
                                                       # de_checkpoint='dataset_root/Save_checkpoint/220826_uw_jointlearning_depthestimation/virtualkitti/de.virtualkitti.jointformer_final/checkpoints/de_model-100500-best_de_silog_6.31936.pth',
                                                       de_checkpoint=None,
                                                       de_strict=True,
                                                       # eh_checkpoint='save/eh.virtualkitti.jointformer_rln_4/checkpoints/eh_model-45000-best_uw_ssim_0.54153.pth',
                                                       eh_checkpoint=None,
                                                       eh_strict=True,
                                                       is_de_no_grad=False,
                                                       is_eh_no_grad=False,
                                                       depth_model_cfg = dict(backbone_pretrained_path=None,
                                                                              strict=True,
                                                                           #    encoder_cfg = dict(type='de_mit_pp_b3'),
                                                                              encoder_cfg = dict(type='de_mit_pp_b3_2'),
                                                                              decoder_cfg = dict(type='single_d_decoder_b3'),
                                                                              task_cfg = dict(type='DepthEstimation_Task',
                                                                                              max_depth=depth_range['max_depth'])
                                                                              ),
                                                       enhanced_model_cfg = dict(backbone_pretrained_path=None,
                                                                                 strict=True,
                                                                                #  encoder_cfg = dict(type='eh_mit_pp_b3'),
                                                                                 encoder_cfg = dict(type='eh_mit_pp_b3_2'),
                                                                                 decoder_cfg = dict(type='single_i_decoder_b3'),
                                                                                 task_cfg = dict(type='Enhancement_Task')
                                                                                 )
                                                       )
                                  ),
          
                 loss_build_cfg = dict(type='Builder_Loss',
                                       loss_build_list = [dict(type='Build_Enhancement_Loss',
                                                               total_loss_lamda = 2.0,
                                                               loss_cfg_list=[dict(type='L2_loss', 
                                                                                   lambda_l2=1),
                                                                           #    dict(type='SSIMLoss', 
                                                                           #         lambda_ssim=2.5),
                                                                              dict(type='LaplacianLoss', 
                                                                                   lambda_laplacian=0.1)
                                                                              ]
                                                               ),
                                                          dict(type='Build_DepthEstimation_Loss', 
                                                               depth_min_eval=depth_range['min_depth_eval'],
                                                               total_loss_lamda = 1.5,
                                                               loss_cfg_list=[dict(type='Silog_loss', 
                                                                                   alpha_image_loss=0.85)]
                                                               ),
                                                          ]),
                 optimizer_cfg = dict(type='AdamW', 
                                      lr=train_parm['learning_rate'],  
                                      eps=1e-8,
                                      weight_decay=0.01
                                      ),
               
                 scheduler_cfg = dict(type='CosineAnnealingWarmupRestarts',
                                      first_cycle_steps=train_parm['num_epochs'],
                                      max_lr=train_parm['learning_rate'],
                                      min_lr=train_parm['learning_rate'] * 0.01,
                                      warmup_steps=1
                                      ),
                  
                 train_dataloader_cfg = dict(type='Build_DataLoader',
                                             batch_size=train_parm['batch_size'],
                                             num_threads=train_parm['num_threads'],
                                             mode='train',
                                             dataloader_dict = dict(type='Joint_De_Eh_Preprocess',
                                                                    auto_crop=True,
                                                                    img_size=(image_size['input_height'], image_size['input_width']),
                                                                    argumentation = dict(do_resize_crop=True,
                                                                                         do_center_crop=True,
                                                                                         do_random_crop=True,
                                                                                         do_random_rotate=True,
                                                                                         do_augment_color=True,
                                                                                         do_horison_flip=True,
                                                                                         do_vertical_flip=False,
                                                                                         random_crop_range=20,
                                                                                         degree=3.5),
                                                                    data_path=dataset['train_data_path'],
                                                                    depth_gt_path=dataset['train_depth_gt_path'],
                                                                    enhanced_gt_path=dataset['train_enhanced_gt_path'],
                                                                    dataset_txt_file=dataset['train_txt_file']
                                                                    )),
                 
                 eval_dataloader_cfg = dict(type='Build_DataLoader',
                                            batch_size=1,
                                            num_threads=train_parm['num_threads'],
                                            mode='eval',
                                            dataloader_dict = dict(type='Joint_De_Eh_Preprocess',
                                                                   auto_crop=True,
                                                                   img_size=(image_size['input_height'], image_size['input_width']),
                                                                   argumentation = dict(do_resize_crop=True,
                                                                                        do_center_crop=True,
                                                                                        do_random_crop=False,
                                                                                        do_random_rotate=False,
                                                                                        do_augment_color=False,
                                                                                        do_horison_flip=False,
                                                                                        do_vertical_flip=False,
                                                                                        random_crop_range=20,
                                                                                        degree=3.5),
                                                                   data_path=dataset['eval_data_path'],
                                                                   depth_gt_path=dataset['eval_depth_gt_path'],
                                                                   enhanced_gt_path=dataset['eval_enhanced_gt_path'],
                                                                   dataset_txt_file=dataset['eval_txt_file']
                                                                   )),
                 
                 test_dataloader_cfg = dict(type='Build_DataLoader',
                                            batch_size=1,
                                            num_threads=train_parm['num_threads'],
                                            mode='test',
                                            dataloader_dict = dict(type='Joint_De_Eh_Preprocess',
                                                                   auto_crop=False,
                                                                   img_size=(image_size['input_height'], image_size['input_width']),
                                                                   argumentation = dict(do_resize_crop=True,
                                                                                        do_center_crop=True,
                                                                                        do_random_crop=False,
                                                                                        do_random_rotate=False,
                                                                                        do_augment_color=False,
                                                                                        do_horison_flip=False,
                                                                                        do_vertical_flip=False,
                                                                                        random_crop_range=20,
                                                                                        degree=3.5),
                                                                   data_path=dataset['test_data_path'],
                                                                   depth_gt_path=dataset['test_depth_gt_path'],
                                                                   enhanced_gt_path=dataset['test_enhanced_gt_path'],
                                                                   dataset_txt_file=dataset['test_txt_file'],
                                                                   is_save_gt_image=test_parm['is_save_gt_image']
                                                                   )),    
                 
                 evaluator_cfg = dict(type='Build_Evaluator',
                                      device=None,
                                      dataloader_eval=None,
                                      ngpus=None,
                                      evaluator_cfg_list=[dict(type='Joint_DepthEstimation_Evaluator', 
                                                               min_depth_eval=depth_range['min_depth_eval'],
                                                               max_depth_eval=depth_range['max_depth_eval'],
                                                               is_checkpoint_save=etc['is_checkpoint_save']),
                                                          
                                                          dict(type='Joint_UW_Enhancement_Evaluator', 
                                                               is_checkpoint_save=etc['is_checkpoint_save']),
                                                          ]
                                      ),
                               
                 sample_eval_cfg = dict(type='Build_Sample_Evaluator',
                                      sample_eval_log_comment='result_jointlearning.diml.jointformer_rln_final',
                                      save_dir='./',
                                      evaluator_cfg_list=[dict(type='Sample_DepthEstimation_Evaluator',
                                                               img_size=(image_size['input_height'], image_size['input_width']),
                                                               is_do_crop=True,
                                                               eval_dir=dataset['sample_eval_data_path'],
                                                               gt_dir=dataset['sample_eval_gt_path'],
                                                               is_txt_save=True)]
                                      ),
                 )                 