# dermaBCC
A proyect that aims to develop a software that's useful in BCC images clasification.

Files:
/r
|
|- /clasification training
|   -training_3_input_RA: a file for simple training
|
|
|- /architecture optimization loop
|   -training_loop: functions to generate a training loop to search and optimize the architecture 
|   -architecture_generation: a function to generate various architectures to becalled from a loop
|   -loop_results_: to recalculate with variable thresholds the loop results
|   -loop_results_representation: representation and filtering of best results within the loop training
|
|
|- /data base
|   -add_batch_to_db: adds and indexes a batch of images to the database
|   -images_sort:  a similar script to the previous, used to database management 
|
|
|- /processing
|   -confusion_matrix_custom: functions related to confusion matrix matters
|   -color_clustering: color clustering functions
|   -color_space_image_conversion: color space image conversion
|   -glcm_2: functions for glcm performing
|   -other_functions: basic and general purpouse functions
|   -data_augmentation: data augmentation functions
|   -ROC_curve_function: ROC and AUC functions
|
|
|- /results:
|   -final_results_bccNobcc: final classification BCC/NOBCC
|   -results_bu: a function to find best thresholds to improve a metric






