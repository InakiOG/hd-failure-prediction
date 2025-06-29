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
from sklearn.model_selection import GridSearchCV

def process_chunks(file_path, normalized_rows, raw_rows, columns_to_delete):
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
    dtype_dict = {'date': 'str', 'serial_number': 'str', 'model': 'str', 'capacity_bytes': 'int64', 'failure': 'bool', 'datacenter': 'str', 'cluster_id': 'int8', 'vault_id': 'int16', 'pod_id': 'int16', 'pod_slot_num': 'float32', 'is_legacy_format': 'bool', 'smart_1_normalized': 'float64', 'smart_1_raw': 'float64', 'smart_2_normalized': 'float64', 'smart_2_raw': 'float64', 'smart_3_normalized': 'float64', 'smart_3_raw': 'float64', 'smart_4_normalized': 'float64', 'smart_4_raw': 'float64', 'smart_5_normalized': 'float64', 'smart_5_raw': 'float64', 'smart_7_normalized': 'float64', 'smart_7_raw': 'float64', 'smart_8_normalized': 'float64', 'smart_8_raw': 'float64', 'smart_9_normalized': 'float64', 'smart_9_raw': 'float64', 'smart_10_normalized': 'float64', 'smart_10_raw': 'float64', 'smart_11_normalized': 'float64', 'smart_11_raw': 'float64', 'smart_12_normalized': 'float64', 'smart_12_raw': 'float64', 'smart_13_normalized': 'float64', 'smart_13_raw': 'float64', 'smart_15_normalized': 'float64', 'smart_15_raw': 'float64', 'smart_16_normalized': 'float64', 'smart_16_raw': 'float64', 'smart_17_normalized': 'float64', 'smart_17_raw': 'float64', 'smart_18_normalized': 'float64', 'smart_18_raw': 'float64', 'smart_22_normalized': 'float64', 'smart_22_raw': 'float64', 'smart_23_normalized': 'float64', 'smart_23_raw': 'float64', 'smart_24_normalized': 'float64', 'smart_24_raw': 'float64', 'smart_27_normalized': 'float64', 'smart_27_raw': 'float64', 'smart_71_normalized': 'float64', 'smart_71_raw': 'float64', 'smart_82_normalized': 'float64', 'smart_82_raw': 'float64', 'smart_90_normalized': 'float64', 'smart_90_raw': 'float64', 'smart_160_normalized': 'float64', 'smart_160_raw': 'float64', 'smart_161_normalized': 'float64', 'smart_161_raw': 'float64', 'smart_163_normalized': 'float64', 'smart_163_raw': 'float64', 'smart_164_normalized': 'float64', 'smart_164_raw': 'float64', 'smart_165_normalized': 'float64', 'smart_165_raw': 'float64', 'smart_166_normalized': 'float64', 'smart_166_raw': 'float64', 'smart_167_normalized': 'float64', 'smart_167_raw': 'float64', 'smart_168_normalized': 'float64', 'smart_168_raw': 'float64', 'smart_169_normalized': 'float64', 'smart_169_raw': 'float64', 'smart_170_normalized': 'float64', 'smart_170_raw': 'float64', 'smart_171_normalized': 'float64', 'smart_171_raw': 'float64', 'smart_172_normalized': 'float64', 'smart_172_raw': 'float64', 'smart_173_normalized': 'float64', 'smart_173_raw': 'float64', 'smart_174_normalized': 'float64', 'smart_174_raw': 'float64', 'smart_175_normalized': 'float64', 'smart_175_raw': 'float64', 'smart_176_normalized': 'float64', 'smart_176_raw': 'float64', 'smart_177_normalized': 'float64', 'smart_177_raw': 'float64', 'smart_178_normalized': 'float64', 'smart_178_raw': 'float64', 'smart_179_normalized': 'float64', 'smart_179_raw': 'float64', 'smart_180_normalized': 'float64', 'smart_180_raw': 'float64', 'smart_181_normalized': 'float64', 'smart_181_raw': 'float64', 'smart_182_normalized': 'float64', 'smart_182_raw': 'float64', 'smart_183_normalized': 'float64', 'smart_183_raw': 'float64', 'smart_184_normalized': 'float64', 'smart_184_raw': 'float64', 'smart_187_normalized': 'float64', 'smart_187_raw': 'float64', 'smart_188_normalized': 'float64', 'smart_188_raw': 'float64', 'smart_189_normalized': 'float64', 'smart_189_raw': 'float64', 'smart_190_normalized': 'float64', 'smart_190_raw': 'float64', 'smart_191_normalized': 'float64', 'smart_191_raw': 'float64', 'smart_192_normalized': 'float64', 'smart_192_raw': 'float64', 'smart_193_normalized': 'float64', 'smart_193_raw': 'float64', 'smart_194_normalized': 'float64', 'smart_194_raw': 'float64', 'smart_195_normalized': 'float64', 'smart_195_raw': 'float64', 'smart_196_normalized': 'float64', 'smart_196_raw': 'float64', 'smart_197_normalized': 'float64', 'smart_197_raw': 'float64', 'smart_198_normalized': 'float64', 'smart_198_raw': 'float64', 'smart_199_normalized': 'float64', 'smart_199_raw': 'float64', 'smart_200_normalized': 'float64', 'smart_200_raw': 'float64', 'smart_201_normalized': 'float64', 'smart_201_raw': 'float64', 'smart_202_normalized': 'float64', 'smart_202_raw': 'float64', 'smart_206_normalized': 'float64', 'smart_206_raw': 'float64', 'smart_210_normalized': 'float64', 'smart_210_raw': 'float64', 'smart_218_normalized': 'float64', 'smart_218_raw': 'float64', 'smart_220_normalized': 'float64', 'smart_220_raw': 'float64', 'smart_222_normalized': 'float64', 'smart_222_raw': 'float64', 'smart_223_normalized': 'float64', 'smart_223_raw': 'float64', 'smart_224_normalized': 'float64', 'smart_224_raw': 'float64', 'smart_225_normalized': 'float64', 'smart_225_raw': 'float64', 'smart_226_normalized': 'float64', 'smart_226_raw': 'float64', 'smart_230_normalized': 'float64', 'smart_230_raw': 'float64', 'smart_231_normalized': 'float64', 'smart_231_raw': 'float64', 'smart_232_normalized': 'float64', 'smart_232_raw': 'float64', 'smart_233_normalized': 'float64', 'smart_233_raw': 'float64', 'smart_234_normalized': 'float64', 'smart_234_raw': 'float64', 'smart_235_normalized': 'float64', 'smart_235_raw': 'float64', 'smart_240_normalized': 'float64', 'smart_240_raw': 'float64', 'smart_241_normalized': 'float64', 'smart_241_raw': 'float64', 'smart_242_normalized': 'float64', 'smart_242_raw': 'float64', 'smart_244_normalized': 'float64', 'smart_244_raw': 'float64', 'smart_245_normalized': 'float64', 'smart_245_raw': 'float64', 'smart_246_normalized': 'float64', 'smart_246_raw': 'float64', 'smart_247_normalized': 'float64', 'smart_247_raw': 'float64', 'smart_248_normalized': 'float64', 'smart_248_raw': 'float64', 'smart_250_normalized': 'float64', 'smart_250_raw': 'float64', 'smart_251_normalized': 'float64', 'smart_251_raw': 'float64', 'smart_252_normalized': 'float64', 'smart_252_raw': 'float64', 'smart_254_normalized': 'float64', 'smart_254_raw': 'float64', 'smart_255_normalized': 'float64', 'smart_255_raw': 'float64'}
    aggregated_result = pd.DataFrame()

    for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtype_dict):
        chunk = cleandata_smart(chunk, normalized_rows, raw_rows, columns_to_delete)
        
        aggregated_result = pd.concat([aggregated_result, chunk])

    return aggregated_result

def cleandata_smart(df, normalized_rows, raw_rows, columns_to_delete=None):
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
    if columns_to_delete is None:
        columns_to_delete = ['date','serial_number','model','capacity_bytes','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format']
    
    smart_allowed = []
    for i in normalized_rows: 
        smart_allowed.append(f'smart_{i}_normalized')
    for i in raw_rows:
        smart_allowed.append(f'smart_{i}_raw')

    if len(normalized_rows) > 0:
        for column in df.columns:
            if column != 'failure' and column not in smart_allowed:
                columns_to_delete.append(column)

    df = df.drop(columns=columns_to_delete)
    df = df.fillna(0)

    failed_drives = df[df['failure'] == True]

    non_failed_drives = df[df['failure'] == False].sample(n=5*len(failed_drives), random_state=42)

    result_df = pd.concat([failed_drives, non_failed_drives])

    result_df = result_df.sample(frac=1, random_state=42)

    df = result_df
    return df

def getdata(folder_path, normalized_rows, raw_rows, columns_to_delete):
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
            df = pd.concat([df, process_chunks(file_path, normalized_rows, raw_rows, columns_to_delete)])
            # print(file_path, ' done')
    # Otherwise, process each subfolder for CSV files
    else:
        for subfolder in tqdm(subfolders, desc="Processing subfolders"):
            for file_name in tqdm(os.listdir(subfolder), desc=f"Processing files in {subfolder}"):
                if file_name.endswith(".csv"):
                    # print(f"Processing file: {file_name} in subfolder: {subfolder}")
                    file_path = os.path.join(subfolder, file_name)
                    df = pd.concat([df, process_chunks(file_path, normalized_rows, raw_rows, columns_to_delete)])
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
def importdata(path, normalized_rows = [], raw_rows = [], columns_to_delete = None):
    """
    Import and prepare the dataset for analysis.
    
    Loads data from the 'data' folder, processes it, and displays basic information
    about the dataset including length, shape, and preview of the data.
    
    Returns:
        pd.DataFrame: Processed and balanced dataset ready for analysis
    """
    folder_path = path
    original_df = getdata(folder_path, normalized_rows, raw_rows, columns_to_delete)
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
        X, Y, test_size=0.3, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("x_train (after SMOTE):", X_train_res)
    print("y_train (after SMOTE):", np.max(y_train_res), np.count_nonzero(y_train_res == 1), len(y_train_res))
    return X, Y, X_train_res, X_test, y_train_res, y_test

def train_using_gini(X_train, X_test, y_train, depth=100, leaf=15):
    """
    Train a Decision Tree classifier using the Gini impurity criterion.
    
    Creates and trains a DecisionTreeClassifier with Gini criterion for measuring
    the quality of splits in the decision tree.
    
    Args:
        X_train (array-like): Training feature data
        X_test (array-like): Test feature data (not used in training but kept for consistency)
        y_train (array-like): Training target labels
        depth (int): Maximum depth of the tree
        leaf (int): Minimum samples per leaf
    
    Returns:
        DecisionTreeClassifier: Trained decision tree classifier using Gini criterion
    """
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=42, max_depth=depth, min_samples_leaf=leaf)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, X_test, y_train, depth=100, leaf=15):
    """
    Train a Decision Tree classifier using the entropy criterion.
    
    Creates and trains a DecisionTreeClassifier with entropy criterion for measuring
    the quality of splits in the decision tree.
    
    Args:
        X_train (array-like): Training feature data
        X_test (array-like): Test feature data (not used in training but kept for consistency)
        y_train (array-like): Training target labels
        depth (int): Maximum depth of the tree
        leaf (int): Minimum samples per leaf
    
    Returns:
        DecisionTreeClassifier: Trained decision tree classifier using entropy criterion
    """
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=42,
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

def grid_search_decision_tree(X, y, param_grid=None):
    """
    Perform grid search to find the best max_depth and min_samples_leaf for DecisionTreeClassifier (Gini).
    Args:
        X: Features
        y: Labels
    Returns:
        dict: Best parameters
        float: Best score
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, None],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
        }
    
    clf = DecisionTreeClassifier(criterion="gini", random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best parameters from grid search: {grid_search.best_params_}")
    print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")
    return grid_search.best_params_, grid_search.best_score_

def main():
    data_path = "data/data_test"
    normalized_rows = [1, 3, 5, 7, 9, 187, 189, 190, 195, 197]
    raw_rows = [5, 197]
    data = importdata(data_path, normalized_rows, raw_rows)

    # Split dataset and apply SMOTE
    X, Y, X_train_res, X_test, y_train_res, y_test = splitdataset(data)

    # Grid search for best hyperparameters (Gini)
    print("\nPerforming grid search for Decision Tree (Gini)...")
    best_params, best_score = grid_search_decision_tree(X_train_res, y_train_res)
    depth = best_params['max_depth']
    leaf = best_params['min_samples_leaf']
    print(f"\nTraining Decision Tree with Gini criterion (best params: depth={depth}, leaf={leaf})...")
    clf_gini = train_using_gini(X_train_res, X_test, y_train_res, depth=depth, leaf=leaf)
    print("\nTraining Decision Tree with Entropy criterion (using same best params)...")
    clf_entropy = train_using_entropy(X_train_res, X_test, y_train_res, depth=depth, leaf=leaf)
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
    smart_features = [f'smart_{i:03}' for i in range(1, 255)]
    plot_decision_tree(clf_gini, smart_features, ['No Failure', 'Failure'])
    plot_decision_tree(clf_entropy, smart_features, ['No Failure', 'Failure'])

if __name__ == "__main__":
    main()