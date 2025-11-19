# ✍️ MNIST Handwritten Digit Recognition - Artificial Neural Network

This project demonstrates the training process of a simple **Feedforward Artificial Neural Network (ANN)** model on the widely-used **MNIST** dataset and analyzes the effects of hyperparameter optimization. The implementation is built from scratch using the **NumPy** library in Python to provide a practical understanding of core neural network mechanics.

## 🌟 Project Features and Goals

* **Model Architecture:** A fully connected, 3-layer network (`[784, 30, 10]`).
    * **Input Layer (784):** One neuron for each pixel of the 28x28 MNIST image.
    * **Hidden Layer (30):** A single hidden layer with 30 neurons.
    * **Output Layer (10):** 10 neurons representing the digit classes 0 through 9.
* **Optimization:** Mini-Batch Stochastic Gradient Descent (SGD).
* **Activation Function:** Sigmoid.
* **Cost Function:** Cross-Entropy (provides stable results for classification).
* **Regularization:** L2 Regularization (Lambda=5.0) is applied.
* **Experiments:** The notebook conducts experiments to analyze how key hyperparameters—**learning rate (eta)**, **mini-batch size**, and **hidden layer neuron count**—affect model performance and final test accuracy.

## 📊 Summary of Experiment Results

The following table summarizes the final test accuracy after 5 epochs for the different hyperparameter settings tested:

| Experiment | Setting | Correct Predictions | Total Test Samples | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| Learning rate | eta = 0.1 | 9530 | 10000 | 95.30 |
| Learning rate | **eta = 0.5** | **9552** | 10000 | **95.52** |
| Learning rate | eta = 3.0 | 9089 | 10000 | 90.89 |
| Batch size | **batch = 10** | **9545** | 10000 | **95.45** |
| Batch size | batch = 50 | 9538 | 10000 | 95.38 |
| Batch size | batch = 100 | 9447 | 10000 | 94.47 |
| Hidden units | hidden = 16 | 9305 | 10000 | 93.05 |
| Hidden units | hidden = 30 | 9523 | 10000 | 95.23 |
| Hidden units | **hidden = 64** | **9701** | 10000 | **97.01** |

---

## 🛠️ Setup and Execution

1.  **Prerequisites:** Requires `numpy`, `matplotlib`, `pandas`, `tensorflow` (for MNIST load), and `scikit-learn`.
2.  **Run:** Open `MNIST.ipynb` in a Jupyter environment (or Google Colab) and run all cells sequentially.

## 👤 Author

Abdullah Çifçi - 220611037
