# INRVideoRepresentation
Creates implicit neural representation of given video
Architecture used is similar to NeRF. We are using MLP with sin and cos embeddings for spatial and temporal coordinates
to allow the network to learn high frequency features.
To run network first run data/process.py to create dataset and then run_INR.py to train model.
