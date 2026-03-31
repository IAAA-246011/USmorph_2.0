The files in each directory are described as follows:
            checkpoint/: Contains the parameter data for the supervised model.
            no_labels_images/: Folder containing unlabeled data
            no_labels_images_175/: Folder containing unlabeled data processed via polar coordinate projection
            log/train_log1.txt: Training accuracy for each step
			log/train_log2.txt: Test set accuracy for each epoch
            log/train_log3.txt: Training set accuracy every 100 steps
			log/train_log4.txt: Test set accuracy every 100 steps

0  Configure input data dimensions, the number of epochs between saving weights, the number of steps between accuracy checks (default is 100, meaning accuracy is checked every 100 steps), dataset path, and test set path in conf/global_settings.py

1  Model training: Run the command: `python train.py -net model_name (e.g., googlenet) -gpu (omit if the device lacks a GPU; defaults to CPU) -b 32 (batch size; adjust based on server performance; reducing this value lowers memory usage; must be a multiple of 2)`
    Example usage:
	Input:
        python train.py -net googlenet -gpu  -b 32



2. Model testing: Run the command: python test.py -net model_name (googlenet/densenet121/attention56) -weight path_to_checkpoint_file -b 1000 (batch size; can be adjusted based on server performance; reducing the value reduces memory usage)
        Example usage:
		Input:
            python test.py -net googlenet -weight path_to_checkpoint_file.pth  -b 1000
        Output (model classification accuracy on the validation set):
            (2900, 1, 28, 175) Shapes of data_test after processing
			[tensor(954), tensor(949), tensor(856)] ================== 2759 Number of correctly classified files per batch (1000) and total number of correctly classified files
			{‘3_2’: 42, ‘1_2’: 29, ‘2_3’: 28, ‘3_4’: 9, ‘2_1’: 17, ‘4_3’: 12, ‘1_3’: 1, ‘0_1’: 1, ‘4_1’: 1, ‘1_0’: 1} Note: Overall misclassification statistics, formatted as predicted label + ‘_’ + true label + ‘:’ + the number of data points where the predicted label does not match the true label. For example, ‘3_2’: 42 indicates that there are 42 images in the validation set that belong to class 2 but were classified as class 3.
            Evaluating Network.....
            Test set: Accuracy: 0.9514 

3  After training is complete, use the model's saved weights to classify unlabeled images:
    Execute the command:
    python getlabel.py -net model_name (googlenet/densenet121/attention56) -weight path_to_checkpoint_data_file -b 1000 (number of labeled images processed per batch; can be adjusted based on server performance; reducing this value reduces memory usage)

4  “weight” refers to the pre-trained model weights
