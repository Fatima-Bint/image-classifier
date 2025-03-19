# image-classifier

# Image Classifier Project

This project implements an image classifier using transfer learning in PyTorch to classify images of flowers into 102 categories. The project includes a Jupyter Notebook for interactive development as well as standalone Python scripts for training and prediction.

## Project Files

- **train.py**: Script to train the model on the flower dataset and save a checkpoint.
- **predict.py**: Script to load a saved checkpoint and predict the class of a given image.
- **Image_Classifier_Project.ipynb**: 
- **cat_to_name.json**: JSON file mapping numeric class labels to flower names.


## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/<YourUsername>/Image-Classifier-Project.git
    cd Image-Classifier-Project
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Prepare the dataset**:
    - Ensure you have a `flowers/` folder with `train`, `valid`, and `test` subdirectories.
    

## Usage

### Training the Model

Run the training script from the command line. For example, to train using DenseNet121 on the GPU for 3 epochs:

```bash
python train.py flowers --arch densenet121 --epochs 3 --gpu
