Environment requirements: Please refer to the environment file:
scikit-learn==1.2.2
torch == 1.7.1
pytorch_pretrained_vit
If any libraries are missing, use `conda install + library_name`.
If that doesn’t work, use `pip install + library_name`.

Folder Descriptions:
Place this folder under the previously shared “Coding + Machine Learning” directory, so it is at the same level as these directories:
Data/: Data storage location
Data_process/: Data preprocessing code
cluster/: Clustering code

Execution Order:
First, generate data in the Data directory. Please follow this format:
Data/
    Datasets/
        train/
        test/
	raw/
Please store all data in the raw/ directory. Each image can be named using the following format: label + ‘_’ + filename (if the original data has no labels), e.g., 1_cosmos.jpg
Additionally, please randomly split the raw data into training and testing sets and store them in the train/ and test/ subdirectories under Datasets, respectively. The number of files in each folder is up to you.



Then proceed to the cluster directory:
1. Run `python encoding.py` to generate the raw encoding file.
2. Run `python main.py`. Follow the prompts to enter the number of clusters, the number of dimensions, and the clustering type (PCA clustering or raw encoding clustering). Select the output dimensions from the list provided in the code; you can set these arbitrarily for the first run.
When the program reaches the prompt, stop it and select the appropriate dimension based on the output s_ratio.jpg (the dimension where a sudden change occurs). You will find the multi-model clustering results in the result folder.
3. Run python get_final_result.py to generate the result_final/ folder, which contains the final results of the hybrid clustering.
4. The rest folder within result_final/ contains discarded data—classes that could not be distinguished using unsupervised methods.
5. Run `python check_result.py`. Follow the prompts to enter the number of clusters and the clustering type (PCA clustering or raw encoding clustering). An Excel file will be generated in the `log` folder, containing detailed information on correct classifications, detailed results for correctly classified pairs, detailed results for incorrectly classified pairs, and the overall classification results.
