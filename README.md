# RotatE
This README file describes the implementation, training, and evaluation of the RotatE model, which is used for knowledge graph completion tasks. It includes training and evaluation using multiple thresholds and learning rates.

**Overview:**

The RotatE model embeds entities and relations from a knowledge graph into a complex vector space. Relations are modeled as rotations in the embedding space, and the distance metric determines whether a triple (h,r,t) is valid.


**This implementation:**

-Trains the model using different learning rates.

-Evaluates the model over a range of distance thresholds.

-Plots the results for both learning rates and thresholds.


**Prerequisites:** 
1. Libraries Required:

   -Python 3.x
   
   -PyTorch
   
   -Matplotlib

   -NetworkX (for graph representation)

2.Installation

   Install the required packages:
   
   pip install torch matplotlib networkx


**Code Structure:**
1. Model Definition

   The RotatE class defines the model, including entity and relation embeddings.

2. Training the Model
   
   The train_rotate function trains the RotatE model using a given learning rate and number of epochs.

3. Evaluation
   
   The model is evaluated based on Accuracy vs Threshold:

   The accuracy is calculated for multiple thresholds to find the best distance metric cutoff.

4. Visualization

   The results are plotted using Matplotlib: Accuracy vs Threshold


**Usage:**
1. Define Parameters

   Modify these parameters as needed in the code:

   learning_rates = [0.0001, 0.001, 0.01, 0.1]

   thresholds = [0.1, 0.2, 0.3, 0.5]

   embedding_dim = 100  # Set the dimension of embeddings

2. Train and Evaluate
   
   Run the train_and_evaluate function with the knowledge graph:
   
   train_and_evaluate(explicit_graph, learning_rates, thresholds)

4. View Results
   
   -Accuracy vs Threshold: Compares accuracy for different thresholds across learning rates.


**Sample Plots:**
1. Accuracy vs Threshold

This plot shows how accuracy varies for different thresholds and learning rates.

![image](https://github.com/user-attachments/assets/cbe55bf8-c7f2-4539-88ba-97c7f5ad5952)


**How to Run:**
1. Clone the repository.
2. Ensure the knowledge graph data is loaded.
3. Run the Python script.
4. View the accuracy plots for analysis.


**Future Work:**

-Add support for custom loss functions.

-Include additional evaluation metrics (e.g., precision, recall).

-Optimize the training process for large-scale knowledge graphs.


**Contributors:**

-Avantika Balaji

-Jashwanth Kandula

-Rakshitha Narasimhaiah

Feel free to reach out or submit an issue for questions or suggestions!
