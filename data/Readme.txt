################   ACRIMA project    ################


The images of this database come from the ACRIMA project (TIN2013-46751-R) founded by the Ministerio de Economía y Competitividad of Spain, which aims to the development of automatic algorithms for retinal disease assessment. 

In this folder, the images of the ACRIMA database and the weights of fine-tuned models used in this work can be publicly accessed.

Folder "Algorithm&Weights" contains a python script and weights of the best fine-tuned models for each CNN.

Folder "Database" contains the images of the ACRIMA database and a brief description of the labels.

Folder "bootstrapTests" contains the scripts and weights needed to obtain the bootstrapping results of all the 3 experiments.



Bootstrap Tests:


Because of size reasons, we had to compress the CNN weights and scripts. 
bootstrapTests folder is organized like this tree:

-- bootstrapTests---|- bootstrap ------------------|- ModelsByFold --------------|- ModelsByFold_part1.zip
				    												             |- ModelsByFold_part2.zip

												   |- bootstrap_folds_trained.py
												   |- bootstrap_only_drishti.py
												   |- bootstrap_per_DB.py		

					|- imagesInFolds_OnlyPublic
					|- ModelsOnlyDrishti
					|- ModelsPerDb_Xception
					|- onlyDrishti
					|- originalImages
					|- Readme.txt


How to organize folder "bootstrapTests":

1. Create a new folder "bootstrapTests"
2. Unzipped the file "bootstrapTests_part1.zip" inside the folder "bootstrapTests"
3. Create a new folder "bootstrap" inside the folder "bootstrapTests"
4. Unzipped the file "bootstrapTests_part2_bootstrap.zip" inside the new folder "bootstrap"
5. Create a new folder "ModelsByFold" inside boostrap folder
6. Unzipped the files "bootstrapTests_part3_ModelsByFold_part1.zip" and "bootstrapTests_part4_ModelsByFold_part2.zip" inside the new folder "ModelsByFold"


Python version: Python 3.6.1
In "requirements.txt" are presented the python packages we used to train these models.
