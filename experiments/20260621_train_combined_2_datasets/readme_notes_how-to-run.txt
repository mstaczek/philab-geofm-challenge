# debug filenames of saved predictions
python experiments\20260621_train_combined_2_datasets\c_testing_saving.py

# pixel-wise, 1 epoch, check if saving zip works and if can run as script
python experiments\20260621_train_combined_2_datasets\a_pixelwise_test.py

# unet alphaearth combined with terraminds1 concatenated into after encoder before decoder - test if works and can save etc
python experiments\20260621_train_combined_2_datasets\b_unet_basic_alphaeart_terraminds1_test.py

# unet alphaearth combined with terraminds1 concatenated into after encoder before decoder - 50 epochs
python experiments\20260621_train_combined_2_datasets\b_unet_basic_alphaeart_terraminds1_try_1.py
