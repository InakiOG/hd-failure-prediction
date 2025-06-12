# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFdr, chi2
from sklearn import tree
from tqdm import tqdm
import joblib
import json
from datetime import datetime

def process_chunks(file_path):
    """
    Process large CSV files in chunks to handle memory constraints.
    
    Reads a CSV file containing hard drive data in chunks, applies data cleaning,
    and aggregates the results into a single DataFrame.
    
    Args:
        file_path (str): Path to the CSV file to be processed
        
    Returns:
        pd.DataFrame: Aggregated and cleaned DataFrame containing all processed chunks
    """
    chunksize = 10 ** 6
    dtype_dict = {'date': 'str', 'serial_number': 'str', 'model': 'str', 'capacity_bytes': 'int32', 'failure': 'bool', 'datacenter': 'str', 'cluster_id': 'int8', 'vault_id': 'int16', 'pod_id': 'int16', 'pod_slot_num': 'float32', 'is_legacy_format': 'bool', 'smart_1_normalized': 'float64', 'smart_1_raw': 'float64', 'smart_2_normalized': 'float64', 'smart_2_raw': 'float64', 'smart_3_normalized': 'float64', 'smart_3_raw': 'float64', 'smart_4_normalized': 'float64', 'smart_4_raw': 'float64', 'smart_5_normalized': 'float64', 'smart_5_raw': 'float64', 'smart_7_normalized': 'float64', 'smart_7_raw': 'float64', 'smart_8_normalized': 'float64', 'smart_8_raw': 'float64', 'smart_9_normalized': 'float64', 'smart_9_raw': 'float64', 'smart_10_normalized': 'float64', 'smart_10_raw': 'float64', 'smart_11_normalized': 'float64', 'smart_11_raw': 'float64', 'smart_12_normalized': 'float64', 'smart_12_raw': 'float64', 'smart_13_normalized': 'float64', 'smart_13_raw': 'float64', 'smart_15_normalized': 'float64', 'smart_15_raw': 'float64', 'smart_16_normalized': 'float64', 'smart_16_raw': 'float64', 'smart_17_normalized': 'float64', 'smart_17_raw': 'float64', 'smart_18_normalized': 'float64', 'smart_18_raw': 'float64', 'smart_22_normalized': 'float64', 'smart_22_raw': 'float64', 'smart_23_normalized': 'float64', 'smart_23_raw': 'float64', 'smart_24_normalized': 'float64', 'smart_24_raw': 'float64', 'smart_27_normalized': 'float64', 'smart_27_raw': 'float64', 'smart_71_normalized': 'float64', 'smart_71_raw': 'float64', 'smart_82_normalized': 'float64', 'smart_82_raw': 'float64', 'smart_90_normalized': 'float64', 'smart_90_raw': 'float64', 'smart_160_normalized': 'float64', 'smart_160_raw': 'float64', 'smart_161_normalized': 'float64', 'smart_161_raw': 'float64', 'smart_163_normalized': 'float64', 'smart_163_raw': 'float64', 'smart_164_normalized': 'float64', 'smart_164_raw': 'float64', 'smart_165_normalized': 'float64', 'smart_165_raw': 'float64', 'smart_166_normalized': 'float64', 'smart_166_raw': 'float64', 'smart_167_normalized': 'float64', 'smart_167_raw': 'float64', 'smart_168_normalized': 'float64', 'smart_168_raw': 'float64', 'smart_169_normalized': 'float64', 'smart_169_raw': 'float64', 'smart_170_normalized': 'float64', 'smart_170_raw': 'float64', 'smart_171_normalized': 'float64', 'smart_171_raw': 'float64', 'smart_172_normalized': 'float64', 'smart_172_raw': 'float64', 'smart_173_normalized': 'float64', 'smart_173_raw': 'float64', 'smart_174_normalized': 'float64', 'smart_174_raw': 'float64', 'smart_175_normalized': 'float64', 'smart_175_raw': 'float64', 'smart_176_normalized': 'float64', 'smart_176_raw': 'float64', 'smart_177_normalized': 'float64', 'smart_177_raw': 'float64', 'smart_178_normalized': 'float64', 'smart_178_raw': 'float64', 'smart_179_normalized': 'float64', 'smart_179_raw': 'float64', 'smart_180_normalized': 'float64', 'smart_180_raw': 'float64', 'smart_181_normalized': 'float64', 'smart_181_raw': 'float64', 'smart_182_normalized': 'float64', 'smart_182_raw': 'float64', 'smart_183_normalized': 'float64', 'smart_183_raw': 'float64', 'smart_184_normalized': 'float64', 'smart_184_raw': 'float64', 'smart_187_normalized': 'float64', 'smart_187_raw': 'float64', 'smart_188_normalized': 'float64', 'smart_188_raw': 'float64', 'smart_189_normalized': 'float64', 'smart_189_raw': 'float64', 'smart_190_normalized': 'float64', 'smart_190_raw': 'float64', 'smart_191_normalized': 'float64', 'smart_191_raw': 'float64', 'smart_192_normalized': 'float64', 'smart_192_raw': 'float64', 'smart_193_normalized': 'float64', 'smart_193_raw': 'float64', 'smart_194_normalized': 'float64', 'smart_194_raw': 'float64', 'smart_195_normalized': 'float64', 'smart_195_raw': 'float64', 'smart_196_normalized': 'float64', 'smart_196_raw': 'float64', 'smart_197_normalized': 'float64', 'smart_197_raw': 'float64', 'smart_198_normalized': 'float64', 'smart_198_raw': 'float64', 'smart_199_normalized': 'float64', 'smart_199_raw': 'float64', 'smart_200_normalized': 'float64', 'smart_200_raw': 'float64', 'smart_201_normalized': 'float64', 'smart_201_raw': 'float64', 'smart_202_normalized': 'float64', 'smart_202_raw': 'float64', 'smart_206_normalized': 'float64', 'smart_206_raw': 'float64', 'smart_210_normalized': 'float64', 'smart_210_raw': 'float64', 'smart_218_normalized': 'float64', 'smart_218_raw': 'float64', 'smart_220_normalized': 'float64', 'smart_220_raw': 'float64', 'smart_222_normalized': 'float64', 'smart_222_raw': 'float64', 'smart_223_normalized': 'float64', 'smart_223_raw': 'float64', 'smart_224_normalized': 'float64', 'smart_224_raw': 'float64', 'smart_225_normalized': 'float64', 'smart_225_raw': 'float64', 'smart_226_normalized': 'float64', 'smart_226_raw': 'float64', 'smart_230_normalized': 'float64', 'smart_230_raw': 'float64', 'smart_231_normalized': 'float64', 'smart_231_raw': 'float64', 'smart_232_normalized': 'float64', 'smart_232_raw': 'float64', 'smart_233_normalized': 'float64', 'smart_233_raw': 'float64', 'smart_234_normalized': 'float64', 'smart_234_raw': 'float64', 'smart_235_normalized': 'float64', 'smart_235_raw': 'float64', 'smart_240_normalized': 'float64', 'smart_240_raw': 'float64', 'smart_241_normalized': 'float64', 'smart_241_raw': 'float64', 'smart_242_normalized': 'float64', 'smart_242_raw': 'float64', 'smart_244_normalized': 'float64', 'smart_244_raw': 'float64', 'smart_245_normalized': 'float64', 'smart_245_raw': 'float64', 'smart_246_normalized': 'float64', 'smart_246_raw': 'float64', 'smart_247_normalized': 'float64', 'smart_247_raw': 'float64', 'smart_248_normalized': 'float64', 'smart_248_raw': 'float64', 'smart_250_normalized': 'float64', 'smart_250_raw': 'float64', 'smart_251_normalized': 'float64', 'smart_251_raw': 'float64', 'smart_252_normalized': 'float64', 'smart_252_raw': 'float64', 'smart_254_normalized': 'float64', 'smart_254_raw': 'float64', 'smart_255_normalized': 'float64', 'smart_255_raw': 'float64'}
    aggregated_result = pd.DataFrame()

    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtype_dict):
        chunk = cleandata_smart(chunk)
        
        aggregated_result = pd.concat([aggregated_result, chunk])

    return aggregated_result

def cleandata_smart(df):
    """
    Clean and preprocess hard drive data for SMART analysis.
    
    Removes unnecessary columns, keeps only specific SMART attributes valuable for 
    failure prediction analysis, handles missing values, and balances the dataset
    by sampling non-failed drives.
    
    Args:
        df (pd.DataFrame): Raw hard drive data DataFrame
        
    Returns:
        pd.DataFrame: Cleaned and balanced DataFrame with selected SMART attributes
    """
    df.head()
    columns_to_delete = ['date','serial_number','model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    smart_allowed = []
    rows_allowed = [1, 3, 5, 7, 9, 187, 189, 190, 195, 197]
    for i in rows_allowed: 
        smart_allowed.append(f'smart_{i}_normalized')
    for column in df.columns:
        if column != 'failure' and column not in smart_allowed and column != "smart_5_raw" and column != "smart_197_raw":
            columns_to_delete.append(column)

    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)

    failed_drives = df[df['failure'] == True]

    non_failed_drives = df[df['failure'] == False].sample(n=5*len(failed_drives), random_state=42)

    result_df = pd.concat([failed_drives, non_failed_drives])

    result_df = result_df.sample(frac=1, random_state=42)

    df = result_df
    return df

def getdata(folder_path):
    """
    Load and process all CSV files from a specified folder or from subfolders.

    If the folder contains CSV files, processes them directly.
    If the folder contains subfolders, processes all CSV files in each subfolder.

    Args:
        folder_path (str): Path to the folder containing CSV files or subfolders

    Returns:
        pd.DataFrame: Combined DataFrame containing all processed CSV data
    """
    df = pd.DataFrame()
    entries = [os.path.join(folder_path, entry) for entry in os.listdir(folder_path)]
    csv_files = [entry for entry in entries if os.path.isfile(entry) and entry.endswith(".csv")]
    subfolders = [entry for entry in entries if os.path.isdir(entry)]

    # If there are CSV files directly in the folder, process them
    if csv_files:
        for file_path in tqdm(csv_files, desc="Processing CSV files"):
            df = pd.concat([df, process_chunks(file_path)])
            # print(file_path, ' done')
    # Otherwise, process each subfolder for CSV files
    else:
        for subfolder in tqdm(subfolders, desc="Processing subfolders"):
            for file_name in tqdm(os.listdir(subfolder), desc=f"Processing files in {subfolder}"):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(subfolder, file_name)
                    df = pd.concat([df, process_chunks(file_path)])
                    # print(file_path, ' done')
    return df

# We will use cleandata_smart because it only uses the values valuable for the paper
def cleandata(df):
    """
    Clean hard drive data by removing unnecessary columns and keeping only normalized SMART values.
    
    Removes metadata columns and keeps only normalized SMART attributes for analysis.
    This is a simpler version of cleandata_smart that keeps all normalized values.
    
    Args:
        df (pd.DataFrame): Raw hard drive data DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with only normalized SMART attributes and failure column
    """
    df.head()
    columns_to_delete = ['date','serial_number','model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    for column in df.columns:
        if 'normalized' not in column and column != 'failure':
            columns_to_delete.append(column)
    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)

    return df

def cleandata_smart_balanced(df):
    """
    Clean and balance hard drive data using SMART attributes with equal sampling.
    
    Similar to cleandata_smart but balances the dataset by sampling equal numbers
    of failed and non-failed drives (using the minimum count between the two classes).
    
    Args:
        df (pd.DataFrame): Raw hard drive data DataFrame
        
    Returns:
        pd.DataFrame: Cleaned and balanced DataFrame with equal failed/non-failed samples
    """
    # Open the CSV file as a pandas table
    df.head()
    columns_to_delete = ['date','serial_number','model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    smart_allowed = []
    rows_allowed = [1, 3, 5, 7, 9, 187, 189, 190, 195, 197]
    for i in rows_allowed: 
        smart_allowed.append(f'smart_{i}_normalized')
    for column in df.columns:
        if 'normalized' not in column and column != 'failure' and column not in smart_allowed and column != "smart_5_raw" and column != "smart_197_raw":
            columns_to_delete.append(column)
    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)

    failure_count = df['failure'].value_counts()

    min_count = min(failure_count[0], failure_count[1])
    print(f"min_count: {min_count}")

    df = pd.concat([df[df['failure'] == 0].sample(min_count), df[df['failure'] == 1].sample(min_count)])

    return df

# Function to import the dataset
def importdata(path):
    """
    Import and prepare the dataset for analysis.
    
    Loads data from the 'data' folder, processes it, and displays basic information
    about the dataset including length, shape, and preview of the data.
    
    Returns:
        pd.DataFrame: Processed and balanced dataset ready for analysis
    """
    folder_path = path
    original_df = getdata(folder_path)
    balance_data = original_df
    # Displaying dataset information
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Dataset: ", balance_data.head())
    
    return balance_data

# Function to split the dataset into features and target variables, and apply SMOTE
def splitdataset(balance_data):
    """
    Split dataset into features and target variables, then apply SMOTE for balancing.
    
    Separates the features (X) from the target variable (Y), splits into train/test sets,
    and applies SMOTE (Synthetic Minority Oversampling Technique) to balance the training data.
    
    Args:
        balance_data (pd.DataFrame): Input dataset with features and target variable
        
    Returns:
        tuple: Contains X, Y, X_train_res, X_test, y_train_res, y_test
            - X: Feature matrix
            - Y: Target variable
            - X_train_res: SMOTE-resampled training features
            - X_test: Test features
            - y_train_res: SMOTE-resampled training targets
            - y_test: Test targets
    """
    X = balance_data.values[:, 1:]
    Y = balance_data.values[:, 0]
    Y = Y.astype('bool')
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=100)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("x_train (after SMOTE):", X_train_res)
    print("y_train (after SMOTE):", np.max(y_train_res), np.count_nonzero(y_train_res == 1), len(y_train_res))
    return X, Y, X_train_res, X_test, y_train_res, y_test

# Decision tree parameters
depth = 100 
leaf = 15

def train_using_gini(X_train, X_test, y_train):
    """
    Train a Decision Tree classifier using the Gini impurity criterion.
    
    Creates and trains a DecisionTreeClassifier with Gini criterion for measuring
    the quality of splits in the decision tree.
    
    Args:
        X_train (array-like): Training feature data
        X_test (array-like): Test feature data (not used in training but kept for consistency)
        y_train (array-like): Training target labels
        
    Returns:
        DecisionTreeClassifier: Trained decision tree classifier using Gini criterion
    """
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=depth, min_samples_leaf=leaf)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, X_test, y_train):
    """
    Train a Decision Tree classifier using the entropy criterion.
    
    Creates and trains a DecisionTreeClassifier with entropy criterion for measuring
    the quality of splits in the decision tree.
    
    Args:
        X_train (array-like): Training feature data
        X_test (array-like): Test feature data (not used in training but kept for consistency)
        y_train (array-like): Training target labels
        
    Returns:
        DecisionTreeClassifier: Trained decision tree classifier using entropy criterion
    """
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=depth, min_samples_leaf=leaf)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Function to make predictions
def prediction(X_test, clf_object):
    """
    Make predictions using a trained classifier.
    
    Uses the trained decision tree classifier to make predictions on test data.
    
    Args:
        X_test (array-like): Test feature data for making predictions
        clf_object (DecisionTreeClassifier): Trained decision tree classifier
        
    Returns:
        array-like: Predicted labels for the test data
    """
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy and other metrics
def cal_accuracy(y_test, y_pred, model=None, model_type=None, save_if_best=True):
    """
    Calculate and display comprehensive accuracy metrics for binary classification.
    
    Computes confusion matrix, accuracy, classification report, and various 
    performance metrics including sensitivity, specificity, precision, recall, etc.
    Optionally saves the model if it performs better than previous versions.
    
    Args:
        y_test (array-like): True labels for test data
        y_pred (array-like): Predicted labels from classifier
        model: Trained model object (optional, for saving)
        model_type (str): Type of model ('gini' or 'entropy', optional)
        save_if_best (bool): Whether to save model if it's the best performer
        
    Returns:
        float: Accuracy score
    """
    cnf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Confusion Matrix: ",
          cnf_matrix)
    print("Accuracy : ",
          accuracy*100)
    print("Report : ",
          classification_report(y_test, y_pred))
    
    # Handle edge cases where confusion matrix might not be 2x2
    if cnf_matrix.shape == (1, 1):
        # Only one class predicted and present in ground truth
        unique_class = np.unique(y_test)[0]
        if unique_class == 0:
            # Only class 0 (no failure)
            TN = cnf_matrix[0, 0]
            TP = FP = FN = 0
        else:
            # Only class 1 (failure)
            TP = cnf_matrix[0, 0]
            TN = FP = FN = 0
    elif cnf_matrix.shape == (2, 1):
        # Two classes in ground truth, but only one class predicted
        if np.unique(y_pred)[0] == 0:
            # Only predicted class 0
            TN = cnf_matrix[0, 0]
            FN = cnf_matrix[1, 0] if cnf_matrix.shape[0] > 1 else 0
            TP = FP = 0
        else:
            # Only predicted class 1
            TP = cnf_matrix[1, 0] if cnf_matrix.shape[0] > 1 else 0
            FP = cnf_matrix[0, 0]
            TN = FN = 0
    elif cnf_matrix.shape == (1, 2):
        # One class in ground truth, but model can predict two classes
        if np.unique(y_test)[0] == 0:
            # Only ground truth class 0
            TN = cnf_matrix[0, 0]
            FP = cnf_matrix[0, 1] if cnf_matrix.shape[1] > 1 else 0
            TP = FN = 0
        else:
            # Only ground truth class 1
            TP = cnf_matrix[0, 1] if cnf_matrix.shape[1] > 1 else 0
            FN = cnf_matrix[0, 0]
            TN = FP = 0
    else:
        # Standard 2x2 confusion matrix
        TN = cnf_matrix[0, 0]  # True Negatives
        FP = cnf_matrix[0, 1]  # False Positives
        FN = cnf_matrix[1, 0]  # False Negatives
        TP = cnf_matrix[1, 1]  # True Positives    # Convert to float for calculations
    FP = float(FP)
    FN = float(FN)
    TP = float(TP)
    TN = float(TN)

    # Calculate metrics with division by zero protection
    TPR = TP/(TP+FN) if (TP+FN) > 0 else 0  # Sensitivity, hit rate, recall, or true positive rate
    TNR = TN/(TN+FP) if (TN+FP) > 0 else 0  # Specificity or true negative rate
    PPV = TP/(TP+FP) if (TP+FP) > 0 else 0  # Precision or positive predictive value
    NPV = TN/(TN+FN) if (TN+FN) > 0 else 0  # Negative predictive value
    FPR = FP/(FP+TN) if (FP+TN) > 0 else 0  # Fall out or false positive rate
    FNR = FN/(TP+FN) if (TP+FN) > 0 else 0  # False negative rate
    FDR = FP/(TP+FP) if (TP+FP) > 0 else 0  # False discovery rate
    ACC = (TP+TN)/(TP+FP+FN+TN) if (TP+FP+FN+TN) > 0 else 0  # Overall accuracy

    print("FP: ", FP)
    print("FN: ", FN)
    print("TP: ", TP)
    print("TN: ", TN)
    print("TPR: ", TPR)
    print("TNR: ", TNR)
    print("PPV: ", PPV)
    print("NPV: ", NPV)
    print("FPR: ", FPR)
    print("FNR: ", FNR)
    print("FDR: ", FDR)
    print("ACC: ", ACC)
    
    # Save model if it's the best performer
    if save_if_best and model is not None and model_type is not None:
        metrics_dict = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        save_best_model(model, model_type, accuracy, metrics_dict)
    
    return accuracy

# Function to plot the decision tree
def plot_decision_tree(clf_object, feature_names, class_names):
    """
    Plot and visualize a trained decision tree.
    
    Creates a visual representation of the decision tree structure showing
    splits, feature importance, and decision paths.
    
    Args:
        clf_object (DecisionTreeClassifier): Trained decision tree classifier to visualize
        feature_names (list): List of feature names for labeling tree nodes
        class_names (list): List of class names for labeling leaf nodes
        
    Returns:
        None: Displays the decision tree plot
    """
    plt.figure(figsize=(25, 20))
    tree.plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()

# Function to save the model
def save_model(model, model_name):
    """
    Save the trained model to a file.
    
    Serializes and saves the trained model using joblib for persistence.
    
    Args:
        model (object): Trained model object to be saved
        model_name (str): Name for the model file (without extension)
        
    Returns:
        None: Saves the model to a file
    """
    # Create the 'models' directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model using joblib
    joblib.dump(model, f"models/DT/{model_name}.joblib")
    print(f"Model saved as: models/DT/{model_name}.joblib")

def save_best_model(model, model_type, accuracy, metrics_dict, model_dir="models/DT"):
    """
    Save a model only if it performs better than the previous best model.
    
    Args:
        model: The trained scikit-learn model
        model_type (str): Type of model ('gini' or 'entropy')
        accuracy (float): Model accuracy score
        metrics_dict (dict): Dictionary containing performance metrics
        model_dir (str): Directory to save models (default: 'models/DT')

    Returns:
        bool: True if model was saved (better performance), False otherwise
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    model_filename = f"{model_dir}/best_{model_type}_tree.joblib"
    metrics_filename = f"{model_dir}/best_{model_type}_metrics.json"
    
    # Check if previous model exists
    should_save = True
    if os.path.exists(metrics_filename):
        with open(metrics_filename, 'r') as f:
            previous_metrics = json.load(f)
        
        # Compare accuracy (you can add more sophisticated comparison logic)
        if accuracy <= previous_metrics.get('accuracy', 0):
            should_save = False
            print(f"❌ New {model_type} model accuracy ({accuracy:.4f}) not better than previous best ({previous_metrics.get('accuracy', 0):.4f})")
            return False
    
    if should_save:
        # Save the model
        joblib.dump(model, model_filename)
        
        # Save metrics with timestamp
        metrics_dict.update({
            'accuracy': accuracy,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'model_filename': model_filename
        })
        
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"✅ New best {model_type} model saved! Accuracy: {accuracy:.4f}")
        return True

# Function to load the model
def load_model(model_name):
    """
    Load a trained model from a file.
    
    Deserializes and loads the trained model using joblib.
    
    Args:
        model_name (str): Name of the model file to be loaded (without extension)
        
    Returns:
        object: Loaded model object
    """
    model = joblib.load(f"models/DT/{model_name}.joblib")
    print(f"Model loaded: models/DT/{model_name}.joblib")
    return model

def load_best_model(model_type, model_dir="models"):
    """
    Load the best saved model and its metrics.
    
    Args:
        model_type (str): Type of model to load ('gini' or 'entropy')
        model_dir (str): Directory where models are saved
    
    Returns:
        tuple: (model, metrics_dict) or (None, None) if no model found
    """
    model_filename = f"{model_dir}/best_{model_type}_tree.joblib"
    metrics_filename = f"{model_dir}/best_{model_type}_metrics.json"
    
    if os.path.exists(model_filename) and os.path.exists(metrics_filename):
        model = joblib.load(model_filename)
        with open(metrics_filename, 'r') as f:
            metrics = json.load(f)
        return model, metrics
    else:
        print(f"⚠️ No saved {model_type} model found")
        return None, None

# Function to log model performance
def log_model_performance(model_name, accuracy, params):
    """
    Log the model performance metrics to a JSON file.
    
    Records the model name, accuracy, and other parameters to a JSON file for tracking
    and comparison of different model runs.
    
    Args:
        model_name (str): Name of the model
        accuracy (float): Accuracy of the model
        params (dict): Dictionary of model parameters
        
    Returns:
        None: Appends the performance data to a JSON file
    """
    log_file = "model_performance.json"
    log_data = {
        "model_name": model_name,
        "accuracy": accuracy,
        "params": params,
        "timestamp": datetime.now().isoformat()
    }
    
    # Load existing log data
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            all_logs = json.load(f)
    else:
        all_logs = []
    
    # Append new log entry
    all_logs.append(log_data)
    
    # Save back to JSON file
    with open(log_file, "w") as f:
        json.dump(all_logs, f, indent=4)
    
    print(f"Logged model performance to {log_file}")

def load_lstm_predictions(lstm_features_path):
    """
    Load LSTM predictions from CSV file generated by smart.py
    
    Args:
        lstm_features_path (str): Path to the CSV file with LSTM-derived features
        
    Returns:
        pd.DataFrame: DataFrame with LSTM features for CT analysis
    """
    if not os.path.exists(lstm_features_path):
        print(f"❌ LSTM features file not found: {lstm_features_path}")
        return None
    
    lstm_features = pd.read_csv(lstm_features_path)
    print(f"✅ Loaded LSTM features from {lstm_features_path}")
    print(f"📊 Features shape: {lstm_features.shape}")
    print(f"🔧 Available features: {list(lstm_features.columns)}")
    
    return lstm_features

def merge_lstm_with_ground_truth(lstm_features_df, ground_truth_path=None, ground_truth_df=None):
    """
    Merge LSTM predictions with ground truth failure data
    
    Args:
        lstm_features_df (pd.DataFrame): DataFrame with LSTM-derived features
        ground_truth_path (str): Path to ground truth data (optional)
        ground_truth_df (pd.DataFrame): Ground truth DataFrame (optional)
        
    Returns:
        pd.DataFrame: Merged DataFrame with LSTM features and failure labels
    """
    if ground_truth_df is None and ground_truth_path is not None:
        if os.path.exists(ground_truth_path):
            ground_truth_df = importdata(ground_truth_path)
        else:
            print(f"⚠️ Ground truth file not found: {ground_truth_path}")
            print("Using synthetic failure labels for demonstration")
            # Create synthetic failure labels based on LSTM predictions
            lstm_features_df['failure'] = (lstm_features_df['lstm_avg_prediction'] > lstm_features_df['lstm_avg_prediction'].quantile(0.8)).astype(int)
            return lstm_features_df
    
    if ground_truth_df is not None:
        # Extract failure information by serial number
        failure_info = ground_truth_df.groupby('serial_number')['failure'].max().reset_index()
        
        # Merge with LSTM features
        merged_df = lstm_features_df.merge(failure_info, on='serial_number', how='left')
        merged_df['failure'] = merged_df['failure'].fillna(0).astype(int)
        
        print(f"✅ Merged LSTM features with ground truth")
        print(f"📊 Merged data shape: {merged_df.shape}")
        print(f"⚖️ Failure distribution: {merged_df['failure'].value_counts().to_dict()}")
        
        return merged_df
    else:
        print("⚠️ No ground truth data provided")
        return lstm_features_df

def prepare_lstm_features_for_ct(lstm_features_df, feature_selection_method='all'):
    """
    Prepare LSTM features for decision tree analysis
    
    Args:
        lstm_features_df (pd.DataFrame): DataFrame with LSTM features and failure labels
        feature_selection_method (str): Method for feature selection ('all', 'lstm_only', 'smart_only')
        
    Returns:
        tuple: (X, Y, feature_names) for decision tree training
    """
    # Remove non-feature columns
    exclude_cols = ['serial_number']
    if 'failure' in lstm_features_df.columns:
        Y = lstm_features_df['failure'].values
        exclude_cols.append('failure')
    else:
        print("⚠️ No failure column found, creating synthetic labels")
        Y = (lstm_features_df['lstm_avg_prediction'] > lstm_features_df['lstm_avg_prediction'].quantile(0.8)).astype(int).values
    
    # Select features based on method
    if feature_selection_method == 'lstm_only':
        feature_cols = [col for col in lstm_features_df.columns 
                       if col.startswith('lstm_') and col not in exclude_cols]
    elif feature_selection_method == 'smart_only':
        feature_cols = [col for col in lstm_features_df.columns 
                       if 'smart_' in col and 'lstm' not in col and col not in exclude_cols]
    else:  # 'all'
        feature_cols = [col for col in lstm_features_df.columns if col not in exclude_cols]
    
    X = lstm_features_df[feature_cols].values
    feature_names = feature_cols
    
    # Handle any missing values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"✅ Prepared features for CT analysis")
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🎯 Target distribution: {np.bincount(Y)}")
    print(f"🔧 Selected {len(feature_names)} features using '{feature_selection_method}' method")
    
    return X, Y, feature_names

def analyze_lstm_predictions_with_ct(lstm_features_path, ground_truth_path=None, 
                                   feature_selection_method='all', output_dir="ct_lstm_analysis"):
    """
    Complete pipeline to analyze LSTM predictions using decision trees
    
    Args:
        lstm_features_path (str): Path to LSTM features CSV
        ground_truth_path (str): Path to ground truth data (optional)
        feature_selection_method (str): Feature selection method
        output_dir (str): Directory to save analysis results
        
    Returns:
        dict: Analysis results including model performance and predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔬 Starting CT analysis of LSTM predictions...")
    print("="*60)
    
    # Load LSTM features
    lstm_features_df = load_lstm_predictions(lstm_features_path)
    if lstm_features_df is None:
        return None
    
    # Merge with ground truth if available
    merged_df = merge_lstm_with_ground_truth(lstm_features_df, ground_truth_path)
    
    # Prepare features for CT
    X, Y, feature_names = prepare_lstm_features_for_ct(merged_df, feature_selection_method)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    
    # Apply SMOTE if needed
    if len(np.unique(Y)) > 1:  # Only apply SMOTE if we have both classes
        smote = SMOTE(random_state=100)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"✅ Applied SMOTE: {X_train.shape} -> {X_train_res.shape}")
    else:
        X_train_res, y_train_res = X_train, y_train
        print("⚠️ Skipped SMOTE (only one class present)")
    
    # Train decision trees
    print("\n🌳 Training Decision Trees...")
    clf_gini = train_using_gini(X_train_res, X_test, y_train_res)
    clf_entropy = train_using_entropy(X_train_res, X_test, y_train_res)
    
    # Make predictions
    y_pred_gini = prediction(X_test, clf_gini)
    y_pred_entropy = prediction(X_test, clf_entropy)
    
    # Calculate accuracies
    print("\n📊 Gini Decision Tree Results:")
    print("="*40)
    gini_accuracy = cal_accuracy(y_test, y_pred_gini)
    
    print("\n📊 Entropy Decision Tree Results:")
    print("="*40)
    entropy_accuracy = cal_accuracy(y_test, y_pred_entropy)
    
    # Feature importance analysis
    gini_importances = clf_gini.feature_importances_
    entropy_importances = clf_entropy.feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gini_importance': gini_importances,
        'entropy_importance': entropy_importances
    }).sort_values('gini_importance', ascending=False)
    
    print(f"\n🔝 Top 10 Most Important Features (Gini):")
    print(importance_df.head(10))
    
    # Save results
    results = {
        'gini_accuracy': gini_accuracy,
        'entropy_accuracy': entropy_accuracy,
        'feature_importance': importance_df.to_dict('records'),
        'feature_selection_method': feature_selection_method,
        'test_samples': len(y_test),
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Save analysis results
    results_path = os.path.join(output_dir, "ct_lstm_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
      # Save feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['gini_importance'])
    plt.yticks(range(len(top_features)), top_features['feature'].tolist())
    plt.xlabel('Feature Importance (Gini)')
    plt.title('Top 15 Most Important Features for LSTM Prediction Classification')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Analysis results: {results_path}")
    print(f"📊 Feature importance plot: {plot_path}")
    
    return results

def run_integrated_lstm_ct_analysis(smart_model_path, dataset_path, output_dir="integrated_analysis"):
    """
    Run the complete integrated analysis: LSTM prediction + CT classification
    
    Args:
        smart_model_path (str): Path to trained LSTM model from smart.py
        dataset_path (str): Path to dataset for analysis
        output_dir (str): Directory to save all analysis results
        
    Returns:
        dict: Complete analysis results
    """
    print("🚀 Starting Integrated LSTM + CT Analysis Pipeline")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate LSTM predictions using smart.py functions
    print("Step 1: Generating LSTM predictions...")
    
    try:
        # Import smart.py functions
        import notebooks.LSTM.smart as smart
        predictions_df, ct_features_df = smart.export_for_ct_analysis(
            smart_model_path, dataset_path, 
            output_dir=os.path.join(output_dir, "lstm_exports")
        )
        
        if predictions_df is None or ct_features_df is None:
            print("❌ Failed to generate LSTM predictions")
            return None
            
    except ImportError:
        print("⚠️ Could not import smart.py. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        print(f"❌ Error generating LSTM predictions: {e}")
        return None
    
    # Step 2: Analyze LSTM predictions with CT
    print("\nStep 2: Analyzing LSTM predictions with Decision Trees...")
    
    lstm_features_path = os.path.join(output_dir, "lstm_exports", "ct_features.csv")
    
    ct_results = analyze_lstm_predictions_with_ct(
        lstm_features_path, 
        ground_truth_path=dataset_path,
        feature_selection_method='all',
        output_dir=os.path.join(output_dir, "ct_analysis")
    )
    
    if ct_results is None:
        print("❌ Failed to perform CT analysis")
        return None
    
    # Step 3: Generate comprehensive report
    print("\nStep 3: Generating comprehensive analysis report...")
    
    comprehensive_results = {
        'pipeline_type': 'LSTM + Decision Tree Integration',
        'lstm_model_path': smart_model_path,
        'dataset_path': dataset_path,
        'lstm_predictions_count': len(predictions_df),
        'ct_features_count': len(ct_features_df),
        'ct_analysis_results': ct_results,
        'pipeline_timestamp': datetime.now().isoformat(),
        'output_directory': output_dir
    }
    
    report_path = os.path.join(output_dir, "comprehensive_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=4)
    
    print(f"\n✅ Integrated analysis complete!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"📄 Comprehensive report: {report_path}")
    print(f"🎯 Best CT accuracy: {max(ct_results['gini_accuracy'], ct_results['entropy_accuracy']):.4f}")
    
    return comprehensive_results

def main():
    """Main execution function with LSTM integration options"""
    print("🌳 Decision Tree Analysis with Optional LSTM Integration")
    print("="*60)
    
    # Configuration options
    use_lstm_predictions = True  # Set to True to analyze LSTM predictions
    lstm_features_path = "ct_analysis/ct_features.csv"  # Path to LSTM-generated features
    smart_model_path = "models/LSTM/lstm_model.pth"     # Path to trained LSTM model
    data_path = "../../data/data_Q1_2025"              # Path to dataset
    
    if use_lstm_predictions and os.path.exists(lstm_features_path):
        print("🔬 Running CT analysis on LSTM predictions...")
        print("-" * 40)
        
        # Analyze LSTM predictions with decision trees
        results = analyze_lstm_predictions_with_ct(
            lstm_features_path=lstm_features_path,
            ground_truth_path=data_path,
            feature_selection_method='all',
            output_dir="ct_lstm_analysis"
        )
        
        if results:
            print(f"✅ LSTM-CT analysis completed successfully!")
            print(f"🎯 Best accuracy: {max(results['gini_accuracy'], results['entropy_accuracy']):.4f}")
        
    elif use_lstm_predictions and os.path.exists(smart_model_path):
        print("🚀 Running integrated LSTM + CT analysis pipeline...")
        print("-" * 40)
        
        # Run the complete integrated pipeline
        results = run_integrated_lstm_ct_analysis(
            smart_model_path=smart_model_path,
            dataset_path=data_path,
            output_dir="integrated_analysis"
        )
        
        if results:
            print(f"✅ Integrated analysis completed successfully!")
    
    else:
        print("📊 Running standard CT analysis on raw data...")
        print("-" * 40)
          # Standard CT analysis on raw data
        data = importdata(data_path)
        
        # Split dataset and apply SMOTE
        X, Y, X_train_res, X_test, y_train_res, y_test = splitdataset(data)
          # Train models
        print("\nTraining Decision Tree with Gini criterion...")
        clf_gini = train_using_gini(X_train_res, X_test, y_train_res)
        
        print("\nTraining Decision Tree with Entropy criterion...")
        clf_entropy = train_using_entropy(X_train_res, X_test, y_train_res)
        
        # Operational Phase - Test both models
        print("\n" + "="*50)
        print("Results Using Gini Index:")
        print("="*50)
        y_pred_gini = prediction(X_test, clf_gini)
        gini_accuracy = cal_accuracy(y_test, y_pred_gini, clf_gini, "gini")
        
        print("\n" + "="*50)
        print("Results Using Entropy:")
        print("="*50)
        y_pred_entropy = prediction(X_test, clf_entropy)
        entropy_accuracy = cal_accuracy(y_test, y_pred_entropy, clf_entropy, "entropy")
        
        # Save models
        save_model(clf_gini, "decision_tree_gini")
        save_model(clf_entropy, "decision_tree_entropy")
        
        # Visualizing the Decision Trees (optional - can be commented out for large trees)
        print("\nGenerating decision tree visualizations...")
        try:
            # Create feature names based on the selected SMART attributes
            smart_features = [f'smart_{i:03}' for i in range(1, 255)]
            
            plot_decision_tree(clf_gini, smart_features, ['No Failure', 'Failure'])
            plot_decision_tree(clf_entropy, smart_features, ['No Failure', 'Failure'])
        except Exception as e:
            print(f"Error plotting decision trees: {e}")
            print("Skipping visualization due to complexity or missing data.")
        
        print(f"\n✅ Standard CT analysis completed!")
        print(f"🎯 Gini accuracy: {gini_accuracy:.4f}")
        print(f"🎯 Entropy accuracy: {entropy_accuracy:.4f}")


if __name__ == "__main__":
    main()

    # Integrated LSTM + CT Analysis (uncomment to run)
    # smart_model_path = "path_to_your_trained_lstm_model"
    # dataset_path = "data"
    # integrated_results = run_integrated_lstm_ct_analysis(smart_model_path, dataset_path, output_dir="integrated_analysis")