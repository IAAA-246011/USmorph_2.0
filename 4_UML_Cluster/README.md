1 Configure config.py   Set the total number of clusters   
2 Run python main.py    Obtain the multi-model clustering results in the result folder   
3 Run python get_final_result.py  Generate the result_final/ directory, which contains the final results of the hybrid clustering (encoded data)   
4 Run `python readimg_fits.py` to convert the encoded data in `result_final/` into PNG images and output them to `result_img/` 
5 Visually classify the final clustering results (`result_img/`) and save the results to `result.txt`   
6 Run: python make_dataset.py to obtain the labeled dataset and output it to dataset/. In this dataset, the first character of each image filename represents the category (4: UNC, 3: IRR, 2: LTD, 1: ETD, 0: SPH).

