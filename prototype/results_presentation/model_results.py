#Visualise classification report

#Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from tabulate import tabulate
import io
from PIL import Image
import os
import joblib
import seaborn as sns

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'attack_simulation'))
DATA_DIR = os.path.join(BASE_DIR, "attack_simulation_results")
OUTPUT_DIR = os.path.join(BASE_DIR, "attack_simulation_viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load data from csv
def load_csv_data():
    """
    Parameters:
    filename (str): Name of the CSV files to read, defaults to "attack_results_fgsm_summary.csv" and "attack_results_pgd_summary"
    
    Returns:
    pandas.DataFrame: The data from the CSV files
    """

    """Load necessary resources for visualisation"""
    fgsm= os.path.join(DATA_DIR, "attack_results_fgsm_summary.csv")
    pgd = os.path.join(DATA_DIR, "attack_results_pgd_summary.csv")

    try:
        # Read the CSV file into a pandas DataFrame
        df_fgsm = pd.read_csv(fgsm)
        df_pgd = pd.read_csv(pgd)
        return df_fgsm, df_pgd
    except FileNotFoundError:
        print(f"Error: File '{fgsm}' not found in the current directory.")
        print(f"Error: File '{pgd}' not found in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None

#Version 2
def create_model_results_table():
    """
    Create a visualization table of model evaluation metrics and save to a single PNG file.
    
    Returns:
    --------
    str
        Path to the saved PNG file
    """
    # Load data
    df_fgsm, df_pgd = load_csv_data()
    
    # Set up the style
    sns.set_style("whitegrid")
    
    # Create a single figure to contain all visualizations
    fig = plt.figure(figsize=(16, 20))
    
    # Define a grid layout for all components
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 2, 1, 2])
    
    # Define metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    # 1. FGSM Summary Table (top-left)
    ax_fgsm_table = fig.add_subplot(gs[0, 0])
    ax_fgsm_table.axis('off')
    
    # Format the data for the FGSM table
    fgsm_table_data = df_fgsm.copy()
    for col in metrics:
        fgsm_table_data[col] = fgsm_table_data[col].round(3)
    
    # Create the FGSM table - show only first few rows if table is large
    sample_rows = 5  # Adjust as needed
    if len(fgsm_table_data) > sample_rows:
        table_display = fgsm_table_data.iloc[:sample_rows].values
        footnote = "(Showing first 5 rows)"
    else:
        table_display = fgsm_table_data.values
        footnote = ""
    
    fgsm_table = ax_fgsm_table.table(
        cellText=table_display,
        colLabels=fgsm_table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the FGSM table
    fgsm_table.auto_set_font_size(False)
    fgsm_table.set_fontsize(9)
    fgsm_table.scale(1, 1.5)
    
    ax_fgsm_table.set_title(f'FGSM Attack Results Summary {footnote}', fontsize=14, pad=20)
    
    # 2. PGD Summary Table (top-right)
    ax_pgd_table = fig.add_subplot(gs[0, 1])
    ax_pgd_table.axis('off')
    
    # Format the data for the PGD table
    pgd_table_data = df_pgd.copy()
    for col in metrics:
        pgd_table_data[col] = pgd_table_data[col].round(3)
    
    # Create the PGD table - show only first few rows if table is large
    if len(pgd_table_data) > sample_rows:
        table_display = pgd_table_data.iloc[:sample_rows].values
        footnote = "(Showing first 5 rows)"
    else:
        table_display = pgd_table_data.values
        footnote = ""
    
    pgd_table = ax_pgd_table.table(
        cellText=table_display,
        colLabels=pgd_table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the PGD table
    pgd_table.auto_set_font_size(False)
    pgd_table.set_fontsize(9)
    pgd_table.scale(1, 1.5)
    
    ax_pgd_table.set_title(f'PGD Attack Results Summary {footnote}', fontsize=14, pad=20)
    
    # 3. LogisticRegression FGSM Results (second row, left)
    # Filter data for LogisticRegression
    lr_fgsm_df = df_fgsm[df_fgsm['Classifier'] == 'LogisticRegression']
    
    # Line chart for LogisticRegression FGSM
    ax_lr_fgsm = fig.add_subplot(gs[1, 0])
    
    for metric in metrics:
        ax_lr_fgsm.plot(lr_fgsm_df['Epsilon'], lr_fgsm_df[metric], marker='o', linewidth=2, label=metric)
    
    ax_lr_fgsm.set_title('LogisticRegression vs Epsilon - FGSM', fontsize=14)
    ax_lr_fgsm.set_xlabel('Epsilon', fontsize=12)
    ax_lr_fgsm.set_ylabel('Score', fontsize=12)
    ax_lr_fgsm.grid(True, linestyle='--', alpha=0.7)
    ax_lr_fgsm.legend(fontsize=10)
    ax_lr_fgsm.set_xticks(lr_fgsm_df['Epsilon'])
    ax_lr_fgsm.set_ylim(-0.05, 1.05)
    
    # 4. LogisticRegression PGD Results (second row, right)
    # Filter data for LogisticRegression
    lr_pgd_df = df_pgd[df_pgd['Classifier'] == 'LogisticRegression']
    
    # Line chart for LogisticRegression PGD
    ax_lr_pgd = fig.add_subplot(gs[1, 1])
    
    for metric in metrics:
        ax_lr_pgd.plot(lr_pgd_df['Epsilon'], lr_pgd_df[metric], marker='o', linewidth=2, label=metric)
    
    ax_lr_pgd.set_title('LogisticRegression vs Epsilon - PGD', fontsize=14)
    ax_lr_pgd.set_xlabel('Epsilon', fontsize=12)
    ax_lr_pgd.set_ylabel('Score', fontsize=12)
    ax_lr_pgd.grid(True, linestyle='--', alpha=0.7)
    ax_lr_pgd.legend(fontsize=10)
    ax_lr_pgd.set_xticks(lr_pgd_df['Epsilon'])
    ax_lr_pgd.set_ylim(-0.05, 1.05)
    
    # 5. LogisticRegression FGSM Table (third row, left)
    ax_lr_fgsm_table = fig.add_subplot(gs[2, 0])
    ax_lr_fgsm_table.axis('off')
    
    # Format the data for the table
    lr_fgsm_table_data = lr_fgsm_df.copy()
    lr_fgsm_table_data = lr_fgsm_table_data.drop(columns=['Classifier'])  # Remove redundant column
    for col in metrics:
        lr_fgsm_table_data[col] = lr_fgsm_table_data[col].round(3)
    
    # Create the table
    lr_fgsm_detail_table = ax_lr_fgsm_table.table(
        cellText=lr_fgsm_table_data.values,
        colLabels=lr_fgsm_table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    lr_fgsm_detail_table.auto_set_font_size(False)
    lr_fgsm_detail_table.set_fontsize(10)
    lr_fgsm_detail_table.scale(1, 1.5)
    
    ax_lr_fgsm_table.set_title('LogisticRegression Results - FGSM', fontsize=14, pad=20)
    
    # 6. LogisticRegression PGD Table (third row, right)
    ax_lr_pgd_table = fig.add_subplot(gs[2, 1])
    ax_lr_pgd_table.axis('off')
    
    # Format the data for the table
    lr_pgd_table_data = lr_pgd_df.copy()
    lr_pgd_table_data = lr_pgd_table_data.drop(columns=['Classifier'])  # Remove redundant column
    for col in metrics:
        lr_pgd_table_data[col] = lr_pgd_table_data[col].round(3)
    
    # Create the table
    lr_pgd_detail_table = ax_lr_pgd_table.table(
        cellText=lr_pgd_table_data.values,
        colLabels=lr_pgd_table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    lr_pgd_detail_table.auto_set_font_size(False)
    lr_pgd_detail_table.set_fontsize(10)
    lr_pgd_detail_table.scale(1, 1.5)
    
    ax_lr_pgd_table.set_title('LogisticRegression Results - PGD', fontsize=14, pad=20)
    
    # 7. Comparison of all classifiers - FGSM (bottom left)
    ax_all_fgsm = fig.add_subplot(gs[3, 0])
    
    # For each classifier, plot accuracy against epsilon
    for classifier in df_fgsm['Classifier'].unique():
        clf_data = df_fgsm[df_fgsm['Classifier'] == classifier]
        ax_all_fgsm.plot(clf_data['Epsilon'], clf_data['Accuracy'], marker='o', linewidth=2, label=classifier)
    
    ax_all_fgsm.set_title('All Classifiers Accuracy vs Epsilon - FGSM', fontsize=14)
    ax_all_fgsm.set_xlabel('Epsilon', fontsize=12)
    ax_all_fgsm.set_ylabel('Accuracy', fontsize=12)
    ax_all_fgsm.grid(True, linestyle='--', alpha=0.7)
    ax_all_fgsm.legend(fontsize=10)
    ax_all_fgsm.set_xticks(clf_data['Epsilon'])
    ax_all_fgsm.set_ylim(-0.05, 1.05)
    
    # 8. Comparison of all classifiers - PGD (bottom right)
    ax_all_pgd = fig.add_subplot(gs[3, 1])
    
    # For each classifier, plot accuracy against epsilon
    for classifier in df_pgd['Classifier'].unique():
        clf_data = df_pgd[df_pgd['Classifier'] == classifier]
        ax_all_pgd.plot(clf_data['Epsilon'], clf_data['Accuracy'], marker='o', linewidth=2, label=classifier)
    
    ax_all_pgd.set_title('All Classifiers Accuracy vs Epsilon - PGD', fontsize=14)
    ax_all_pgd.set_xlabel('Epsilon', fontsize=12)
    ax_all_pgd.set_ylabel('Accuracy', fontsize=12)
    ax_all_pgd.grid(True, linestyle='--', alpha=0.7)
    ax_all_pgd.legend(fontsize=10)
    ax_all_pgd.set_xticks(clf_data['Epsilon'])
    ax_all_pgd.set_ylim(-0.05, 1.05)
    
    # Add an overall title
    plt.suptitle('Comprehensive Model Results for FGSM and PGD Attacks', fontsize=18, y=0.99)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save and show
    filename = 'model_results.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.show()
    print("Main visualization completed and saved to 'model_results.png'")
    
    return fig