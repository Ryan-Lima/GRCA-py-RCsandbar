# GRCA-py-RCsandbar
Python analysis of sandbar imagery from remote cameras in Grand Canyon 

DCNN_support_scripts.py 
	contains useful functions for running DCNN binary sandbar segmentation including many image processing tools
	functions:
		cosine_ratedecay(max_epochs, max_lr, min_lr=1e-6)
			ratedecay(epoch)
			visualize(image, mask, original_image=None, original_mask=None)
			split_test_train_val(image_dir,label_dir,split = .8)
			image_generator(image_files, label_files ,batch_size, sz)
			image_generator_augment(image_files, label_files ,batch_size, sz)
			mean_iou(y_true, y_pred)
			dice_coef(y_true, y_pred)
			dice_coef_loss(y_true, y_pred)
			load_models(model_directory, exts = ['.h5', '.H5'], optimizer = 'adam', loss = dice_coef_loss, batch_sz = 5)
			plot__history_metric(history_obj, metric, save = False, fig_run_name = 'No_name')
			plot_history_diceloss_and_loss(history_obj, save = False, fig_run_name = 'No_name')
			plot_history_all(history_obj, save = False, fig_run_name = 'No_name')
			evaluate_model_accuarcy(test_generator,model, threshold = 0.5)
			get_avg_f1(model,TEST_images, TEST_labels, threshold = 0.5)
			img_lab_to_list_path(img_dir, lab_dir)
			img_lab_to_list(img_dir, lab_dir)
			test_dir_to__test_generator(test_img_dir, test_lab_dir, sz)
			saveHist(path,history)
			loadHist(path)
			describe(array)
			
Unet_models.py 
	contains the different UNet models used in this analysis
	functions:
		unet_1(sz)
		unet_2(sz)
		
		batchnorm_act(x)
		conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1)
		bottleneck_block(x, filters, kernel_size=(3, 3), padding="same", strides=1)
		res_block(x, filters, kernel_size=(3, 3), padding="same", strides=1)
		upsamp_concat_block(x, xskip)
		res_unet(sz, batch_size)
		
		