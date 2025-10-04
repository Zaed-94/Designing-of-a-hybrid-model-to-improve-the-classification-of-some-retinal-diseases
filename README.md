# Designing-of-a-hybrid-model-to-improve-the-classification-of-some-retinal-diseases
<img width="424" height="341" alt="11" src="https://github.com/user-attachments/assets/93ffcaef-9b57-461e-865e-4417ebb723d4" />

##  (Introduction)

This project aims to classify eye diseases using OCT (Optical Coherence Tomography) images, which are high-resolution medical images primarily used for the early detection of retinal disorders and vision-related diseases such as macular degeneration and diabetic retinopathy.
The research problem lies in the need to improve the accuracy of classification models while reducing irrelevant features, all while maintaining strong performance in medical diagnosis.
The project leverages artificial intelligence and deep learning techniques to extract features from OCT images, and then enhances the model’s performance using the Genetic Algorithm (GA) and the Dolphin Swarm Algorithm (DSA) to select the most significant features that improve classification accuracy.
The workflow is divided into separate Jupyter Notebook files within the Visual Studio Code Jupyter Extension, where each notebook handles a specific stage of the process.
This modular design makes the project more organized, reusable, and extensible for future researchers who may wish to build upon this work.
The ultimate goal is to provide a documented scientific framework that can be relied upon in further research and shared publicly via GitHub.

## (Requirements)

 - Python 3.9+
 -development environment: **Visual Studio Code** With the addition of **Jupyter Notebook Extension**
 - Libraries used:
    - pandas==1.3.3
    - numpy==1.19.5
    - matplotlib==3.4.3
    - seaborn==0.11.2
    - opencv-python==4.5.3.56
    - scikit-learn==0.24.2
    - tensorflow==2.5.0
    - keras==2.4.3
 - file `requirements.txt` Attached is a file containing all the necessary libraries..
---

## (Installation)

Install the required libraries:
pip install -r requirements.txt

## (Build)

There is no need for a complex build process. However, if you wish to convert the model into an executable file, you can use:
PyInstaller to create a .exe file.
TensorFlow SavedModel to save and transfer the final trained model.

## (Run)

To run the project, please follow these steps:
1. Open the development environment:
   Launch Visual Studio Code.
   Make sure the Jupyter Extension for VS Code is installed.
2. Select the Python environment:
   From the top of the VS Code window, choose the default Python environment or the virtual environment (venv/conda) that contains the libraries listed in requirements.txt.
3. Open the Notebooks:
   The project is divided into several Jupyter Notebooks, each handling a specific stage of the workflow:
   1_data_partitioning_and_initial_CNN.ipynb → Data partitioning and building the initial CNN model before optimization.
   2_genetic_algorithm.ipynb → Optimize the CNN model parameters using the Genetic Algorithm, then use the optimized model to extract features from OCT images.
   3_feature_selection_dolphin_swarm.ipynb → Feature selection using the Dolphin Swarm Algorithm.
   4_evaluation.ipynb → Evaluate results and display graphs (Accuracy, Confusion Matrix).
   5_tkinter_interface.ipynb → Create a simple graphical interface using Tkinter to facilitate model execution and interactive result display.
4. Execute the code:
   Inside each Notebook, click Run All Cells or execute cells step by step in order.
   It is recommended to run the notebooks in the sequence above to ensure correct results.
5. Project outputs:
   The final results (graphs, tables, and classification accuracy) will appear within the Notebooks after execution.
   Results can be saved or exported as CSV/PNG files directly from the cells if needed.
   
## (Publication)

   This project is part of the Master's thesis titled: "Designing a Hybrid Model to Improve the Classification of Certain Retinal Diseases"
   The related published research papers are:
   
       •	Image Classification Using Deep Learning Techniques: A Comparative Study and Evaluation.

       •	Improving Optical Coherence Tomography Image Classification of Eye Diseases Using the Genetic Algorithm and Dolphin Swarm Optimization.

       •	Enhancing Optical coherence tomography Image Classification via Swarm Optimization-Based Feature Selection and Machine Learning Models.

## (Contact)

Researcher: Zaid Al-Jubouri

Email: dac2014@mtu.edu.iq

GitHub: [https://github.com/Zaed-94]

   


