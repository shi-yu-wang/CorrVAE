This is the sample code of CorrVAE handling dSprites dataset. The code is adapted from the code of PCVAE: https://github.com/xguo7/PCVAE.
===========================================================================================================
Running environment:
--------------------
Python 3.9; 

===========================================================================================================
Dependencies:
-------------
PyTorch 1.8.1
networkx 2.5
pandas 1.1.3
numpy 1.20.2
rdkit 2021.09.3

===========================================================================================================
Data: 
-----
The dSprites dataset can be downloaded from https://github.com/deepmind/dsprites-dataset. dSprites dataset is located in data folder. The code to reconstruct the dSprites dataset is in .utils/datasets.py

===========================================================================================================
Code description:
-----------------
To train the model, run:
python train.py
or:
directly run code in train.py

This will train the model with the dSprites dataset and returns the trained model as modelCorrVAE.pt.

For evaluation purpose, run code in test.py.
