# dermaBCC
A proyect that aims to develop a software that's useful in BCC images clasification.


Files:
-entrenamientos_3_entradaTriple_RA: a file for simple training 
-genera_arquitectura: a function to generate various architectures to be called from a loop
-bucle_entrenamientos: fichero con el funcionamiento general del bucle de entrenamientos

-curva_ROC_funcion: funciones de las curvas ROC y otras métricas
-recalcula_resultados_bucle: se recalculan los resultados del bucle con umbrales variables
-representacion_final_bucle: búsqueda de mejores resultados del bucle de la arquitectura
-genera_resultados_finales: clasificación final BCC/NOBCC

-confusion_matrix_custom: Funciones relacionadas con la matriz de confusión
-genera_colores_y_centros: Clustering de colores
-convierte_imagenes: Conversión de imágenes a espacios de color
-glcm_2: uso de las matrices de coocurrencia
-funciones_procesado: funciones básucas para el clustering y otros
-aumento de datos: para realizar data augmentation

-agregaLoteImagenesalOrden:
-ordenaImagenes:  este script y el anterior son para la gestión de la base de datos

