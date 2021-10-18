Descompacte os arquivos desta pasta, nesta pasta, para que os experimentos tenham os modelos de dados treinados.

Modelo de dados utilizados para treinamento do FacemarkAMM transportado de
https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip/


# Licença shape_predictor_68_face_landmarks.dat.bz2
 
Este é treinado no conjunto de dados ibug 300-W (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
Este modelo foi trazido do repositório (https://github.com/davisking/dlib-models)
  
    C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
    300 faces In-the-wild challenge: Database and results. 
    Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
   
The license for this dataset excludes commercial use and Stefanos Zafeiriou,
one of the creators of the dataset, asked me to include a note here saying
that the trained model therefore can't be used in a commerical product.  So
you should contact a lawyer or talk to Imperial College London to find out
if it's OK for you to use this model in a commercial product.  
 
Also note that this model file is designed for use with dlib's HOG face detector.  That is, it expects the bounding
boxes from the face detector to be aligned a certain way, the way dlib's HOG face detector does it.  It won't work
as well when used with a face detector that produces differently aligned boxes, such as the CNN based mmod_human_face_detector.dat face detector.


# Licença Demais Modelos Treinados

Todos os demais modelos de dados, não mencionados acima, treinados e utilizados neste projeto estão sob licença BSD e é vetado o uso para fins comerciais.
Ao baixar, copiar, instalar ou usar estes modelos, você concorda com esta licença.
Se você não concorda com esta licença, não baixe, instale, copie ou use o software.


                          License Agreement
                       For Open  Trained Models
                        (3-clause BSD License)