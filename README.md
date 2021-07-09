# dermaBCC
A proyect that aims to develop a software that's useful in BCC images clasification.


Files:
-entrenamientos_3_entradaTriple_RA: a file for simple training 
-genera_arquitectura: a function to generate various architectures to be called from a loop
-bucle_entrenamientos: functions to generate a training loop to search and optimize the architecture 
-curva_ROC_funcion: ROC and AUC functions
-recalcula_resultados_bucle: to recalculate with variable thresholds the loop results
-representacion_final_bucle: representation and filtering of best results within the loop training
-genera_resultados_finales: final classification BCC/NOBCC

-confusion_matrix_custom: functions related to confusion matrix matters
-genera_colores_y_centros: color clustering
-convierte_imagenes: color space image conversion
-glcm_2: functions for glcm performing
-funciones_procesado: basic and general purpouse functions
-aumento de datos: data augmentation functions

-agregaLoteImagenesalOrden: adds and indexes a batch of images to the database
-ordenaImagenes:  a similar script to the previous, used to database management
