# Apply GNN on decompiled Source Code
Authors: Sajib Biswas and Fatema Liza.
Overview: The code here works with C source or decompiled code and JOERN to do the following:
  -run the JOERN script on C programs to create nodes.JSON and edges.JSON files for each C file. 
  -apply gcn and/or gat on the graphs created from each source program to predict node types.
  
Requirements:
  -Joern (you can download from here: https://github.com/joernio/joern. Follow its instructions)
  -Code tested on Python 3.8.10 but ideally, Python 3.x should work.
  -Each Python process can consume up to 512 mb of memory. RAM requirement is: core count * 512 mb.

Scripts have been tested and work on Ubuntu 20.04 with CUDA 11.8

Working with the DIRT dataset:
  1. We are working with the DIRT dataset provided by this github repository: https://github.com/CMUSTRUDEL/DIRTY
  2. To download the dataset, follow these steps.
      1. cd dirty/
      2. b. wget cmu-itl.s3.amazonaws.com/dirty/dirt.tar.gz -O dirt.tar.gz
      3. c. tar -xzf dirt.tar.gz
  3. Then run the xxx scripts to create the .C files for each of the functions in the dataset.
  4. After that, follow the steps described below.


  How-to use:
  1. Create a Directory in the project which will work as a master directory, it can have sub-directories.
  2. Run the command "python create_nodes_and_edges.py -m ${MASTER_DIR}", this will create the .JSON files.
  3. Run the command "python run.py -m ${MASTER_DIR}", this will create and train the model on datasets.
  4. The above steps are still incomplete.
