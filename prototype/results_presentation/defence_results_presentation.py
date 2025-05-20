
#Visualise defence results

#Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os

# Define base paths
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'defence_prototype'))
except NameError:
    # __file__ is not defined (e.g., in Jupyter)
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'defence_prototype'))

DATA_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load data from csv
def load_csv_data():
    """
    Load data from a CSV file in the same GitHub folder.
    
    Parameters:
    filename (str): Name of the CSV file to read, defaults to 'defence_results.csv'
    
    Returns:
    pandas.DataFrame: The data from the CSV file
    """
    """Load necessary resources for visualisation"""
    filename = os.path.join(DATA_DIR, "defence_results.csv")

    try:
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None


#Version 2
def create_advanced_performance_comparison():
    """
    Create a sophisticated comparison of model performance across metrics
    
    Parameters:
    file_path (str): Path to the CSV file
    output_file (str): Path to save the table image
    """

    #Add output file for each data_type
    output_file1="advanced_performance_comparison.png"
    output_file2="advanced_performance_comparison_FGSM.png"
    output_file3="advanced_performance_comparison_PGD.png"

    #Load csv from GitHub file path
    file_path = 'https://raw.githubusercontent.com/UniSA-ICT-2025-SP1-P2/capstone_project/refs/heads/master/prototype/defence_prototype/results/defence_results.csv'
    
    # Read the CSV file
    df = pd.read_csv(file_path)

    def create_cleaned(df):

        #Filter dataframe to only show clean results
        data_to_remove = ['fgsm', 'fgsm_smoothed', 'pgd', 'pgd_smoothed']
        df = df[~df['data_type'].isin(data_to_remove)]
        
        # Create colorblind-friendly palette
        # Using a colorblind-friendly palette from ColorBrewer
        colors = ['#1b9e77', '#d95f02', '#7570b3']
        model_colors = dict(zip(df['model'].unique(), colors))
        
        # Set up figure with GridSpec for complex layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        
        # Plot 1: Bar plot with error bars (top left)
        ax1 = plt.subplot(gs[0, 0])
        
        # Calculate improvement percentages for accuracy
        pivot_acc = df.pivot_table(index='model', columns='defence', values='accuracy')
        pivot_acc['improvement'] = (pivot_acc['feature_smoothing'] - pivot_acc['none']) / pivot_acc['none'] * 100
    
        # Create bar plot for accuracy with improved styling
        ax1 = sns.barplot(
            x='model',
            y='accuracy',
            hue='defence',
            data=df,
            palette=['#2c7fb8', '#7fcdbb'],  # Blue for none, teal for feature_smoothing
            ax=ax1,
            alpha=0.85
        )
        
        # Enhance with actual values
        for i, p in enumerate(ax1.patches):
            height = p.get_height()
            ax1.text(
                p.get_x() + p.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha="center", 
                fontsize=9,
                color='black',
                fontweight='bold'
            )
        
        # Add improvement percentages for each model
        for i, model in enumerate(pivot_acc.index):
            improvement = pivot_acc.loc[model, 'improvement']
            if abs(improvement) >= 0.1:  # Only show if change is at least 0.1%
                color = 'green' if improvement > 0 else 'red'
                ax1.text(
                    i, 
                    0.05,
                    f"{improvement:+.2f}%",
                    ha="center", 
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )
        
        ax1.set_title('Model Accuracy by Defence Method', fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Bar plot for F1 score (top right)
        ax2 = plt.subplot(gs[0, 1])
        
        # Calculate improvement percentages for F1 score
        pivot_f1 = df.pivot_table(index='model', columns='defence', values='f1_score')
        pivot_f1['improvement'] = (pivot_f1['feature_smoothing'] - pivot_f1['none']) / pivot_f1['none'] * 100
        
        # Create bar plot for f1_score with improved styling
        ax2 = sns.barplot(
            x='model',
            y='f1_score',
            hue='defence',
            data=df,
            palette=['#2c7fb8', '#7fcdbb'],  # Blue for none, teal for feature_smoothing
            ax=ax2,
            alpha=0.85
        )
        
        # Enhance with actual values
        for i, p in enumerate(ax2.patches):
            height = p.get_height()
            ax2.text(
                p.get_x() + p.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha="center", 
                fontsize=9,
                color='black',
                fontweight='bold'
            )
        
        # Add improvement percentages for each model
        for i, model in enumerate(pivot_f1.index):
            improvement = pivot_f1.loc[model, 'improvement']
            if abs(improvement) >= 0.1:  # Only show if change is at least 0.1%
                color = 'green' if improvement > 0 else 'red'
                ax2.text(
                    i, 
                    0.05,
                    f"{improvement:+.2f}%",
                    ha="center", 
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )
        
        ax2.set_title('Model F1 Score by Defence Method', fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1 Score')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 3: Performance metrics comparison (bottom left)
        ax3 = plt.subplot(gs[1, 0])
        
        # Create a dot plot instead of radar chart - better for direct comparisons
        # Before using precision and recall, check if they exist in the dataframe
        metrics = ['accuracy', 'f1_score']
        if 'precision' in df.columns:
            metrics.append('precision')
        if 'recall' in df.columns:
            metrics.append('recall')
            
        # Melt the dataframe to get all metrics in a single column
        df_metrics = df.melt(
            id_vars=['model', 'defence'],
            value_vars=metrics,
            var_name='metric', 
            value_name='value'
        )
        
        # Create the dot plot - split by defense method
        # First, plot 'none' defense
        sns.pointplot(
            data=df_metrics[df_metrics['defence'] == 'none'],
            x='metric',
            y='value',
            hue='model',
            palette=colors,
            markers='o',  # Circle marker
            linestyles='-',  # Solid line
            ax=ax3,
            errwidth=2,
            capsize=0.2
        )
        
        # Then plot 'feature_smoothing' defense with different markers and linestyle
        sns.pointplot(
            data=df_metrics[df_metrics['defence'] == 'feature_smoothing'],
            x='metric',
            y='value',
            hue='model',
            palette=colors,
            markers='s',  # Square marker
            linestyles='--',  # Dashed line
            ax=ax3,
            errwidth=2,
            capsize=0.2
        )
        
        # Create a custom legend that includes defense methods
        from matplotlib.lines import Line2D
        handles, labels = ax3.get_legend_handles_labels()
        defense_handles = [
            Line2D([0], [0], marker='o', color='gray', linestyle='-', markersize=8, label='No Defense'),
            Line2D([0], [0], marker='s', color='gray', linestyle='--', markersize=8, label='Feature Smoothing')
        ]
        # Remove the existing legend
        ax3.get_legend().remove()

        # Add a new legend with both model and defense information
        ax3.legend(handles=handles + defense_handles, 
                labels=labels + ['No Defense', 'Feature Smoothing'],
                title='Model & Defense',
                loc='best')
        
        ax3.set_title('Performance Metrics Comparison', fontweight='bold')
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Score')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.set_ylim(0, 1.0)  # Set y-axis from 0 to 1 for better comparison
        
        # Add a table summarizing key findings
        table_data = []
        models = df['model'].unique()
        
        for model in models:
            model_data = df[df['model'] == model]
            
            # Safely get values, handling potential missing data
            try:
                acc_none = model_data[model_data['defence'] == 'none']['accuracy'].values[0]
                acc_fs = model_data[model_data['defence'] == 'feature_smoothing']['accuracy'].values[0]
                f1_none = model_data[model_data['defence'] == 'none']['f1_score'].values[0]
                f1_fs = model_data[model_data['defence'] == 'feature_smoothing']['f1_score'].values[0]
                
                acc_change = ((acc_fs - acc_none) / acc_none) * 100
                f1_change = ((f1_fs - f1_none) / f1_none) * 100
                
                table_data.append([model, f"{acc_change:+.2f}%", f"{f1_change:+.2f}%"])
            except (IndexError, ZeroDivisionError):
                # Handle cases where data might be missing
                table_data.append([model, "N/A", "N/A"])
        
        # Only create table if we have data
        if table_data:
            table = ax3.table(
                cellText=table_data,
                colLabels=['Model', 'Acc. Change', 'F1 Change'],
                loc='bottom',
                cellLoc='center',
                bbox=[0.0, -0.35, 1.0, 0.2]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            plt.subplots_adjust(bottom=0.25)
        
        # Plot 4: Heatmap comparison (bottom right)
        ax4 = plt.subplot(gs[1, 1])
        
        # Reshape data for heatmap
        pivot_acc = df.pivot_table(index='model', columns='defence', values='accuracy')
        pivot_f1 = df.pivot_table(index='model', columns='defence', values='f1_score')
        
        # Create labels for annotations
        def create_annotation_text(acc, f1):
            return f'Acc: {acc:.4f}\nF1: {f1:.4f}'
        
        # Create annotation labels
        annotations = np.empty_like(pivot_acc, dtype=object)
        for i in range(pivot_acc.shape[0]):
            for j in range(pivot_acc.shape[1]):
                annotations[i,j] = create_annotation_text(
                    pivot_acc.iloc[i,j], 
                    pivot_f1.iloc[i,j]
                )
        
        # Create a composite metric for coloring (e.g., average of acc and F1)
        composite_score = (pivot_acc + pivot_f1) / 2
        
        # Create a better colormap for the heatmap (yellow-green-blue)
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', 
            ['#4575b4', '#91bfdb', '#e0f3f8', '#ffffbf', '#fee090', '#fc8d59', '#d73027'][::-1]
        )
        
        # Create heatmap with improved styling
        sns.heatmap(
            composite_score, 
            annot=annotations, 
            fmt='', 
            cmap=cmap, 
            linewidths=.5, 
            cbar_kws={'label': 'Avg(Accuracy, F1)'},
            ax=ax4,
            annot_kws={"fontsize": 9, "fontweight": "bold"}
        )
        
        ax4.set_title('Performance Metrics by Model and Defence', fontweight='bold')
        ax4.set_xlabel('Defence Method')
        ax4.set_ylabel('Model')
        
        # Add overall figure title
        plt.suptitle('Advanced Analysis of Model Performance with Defence Methods', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Add explanatory text box
        description_text = """
        Feature Smoothing Defence: A technique that reduces adversarial vulnerability by
        smoothing input features, potentially at a small cost to performance on clean data.
        
        Key observations:
        • Random Forest (rf) models significantly outperform other architectures
        • Neural networks show minimal change with feature smoothing
        • Best overall performance: Random Forest with no defence
        """
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        fig.text(0.5, 0.02, description_text, ha='left', va='bottom', fontsize=11, 
                bbox=props, transform=fig.transFigure)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.07, 1, 0.96])
        
        # Save figure
        plt.savefig(output_file1, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Advanced performance comparison saved to {output_file1}")
        return fig
    
    def create_FGSM(df):

        #Filter dataframe to only show clean results
        data_to_remove = ['clean', 'clean_smoothed', 'pgd', 'pgd_smoothed']
        df = df[~df['data_type'].isin(data_to_remove)]
        
        # Create colorblind-friendly palette
        # Using a colorblind-friendly palette from ColorBrewer
        colors = ['#1b9e77', '#d95f02', '#7570b3']
        model_colors = dict(zip(df['model'].unique(), colors))
        
        # Set up figure with GridSpec for complex layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        
        # Plot 1: Bar plot with error bars (top left)
        ax1 = plt.subplot(gs[0, 0])
        
        # Calculate improvement percentages for accuracy
        pivot_acc = df.pivot_table(index='model', columns='defence', values='accuracy')
        pivot_acc['improvement'] = (pivot_acc['feature_smoothing'] - pivot_acc['none']) / pivot_acc['none'] * 100
    
        # Create bar plot for accuracy with improved styling
        ax1 = sns.barplot(
            x='model',
            y='accuracy',
            hue='defence',
            data=df,
            palette=['#2c7fb8', '#7fcdbb'],  # Blue for none, teal for feature_smoothing
            ax=ax1,
            alpha=0.85
        )
        
        # Enhance with actual values
        for i, p in enumerate(ax1.patches):
            height = p.get_height()
            ax1.text(
                p.get_x() + p.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha="center", 
                fontsize=9,
                color='black',
                fontweight='bold'
            )
        
        # Add improvement percentages for each model
        for i, model in enumerate(pivot_acc.index):
            improvement = pivot_acc.loc[model, 'improvement']
            if abs(improvement) >= 0.1:  # Only show if change is at least 0.1%
                color = 'green' if improvement > 0 else 'red'
                ax1.text(
                    i, 
                    0.05,
                    f"{improvement:+.2f}%",
                    ha="center", 
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )
        
        ax1.set_title('Model Accuracy by Defence Method after FGSM', fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Bar plot for F1 score (top right)
        ax2 = plt.subplot(gs[0, 1])
        
        # Calculate improvement percentages for F1 score
        pivot_f1 = df.pivot_table(index='model', columns='defence', values='f1_score')
        pivot_f1['improvement'] = (pivot_f1['feature_smoothing'] - pivot_f1['none']) / pivot_f1['none'] * 100
        
        # Create bar plot for f1_score with improved styling
        ax2 = sns.barplot(
            x='model',
            y='f1_score',
            hue='defence',
            data=df,
            palette=['#2c7fb8', '#7fcdbb'],  # Blue for none, teal for feature_smoothing
            ax=ax2,
            alpha=0.85
        )
        
        # Enhance with actual values
        for i, p in enumerate(ax2.patches):
            height = p.get_height()
            ax2.text(
                p.get_x() + p.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha="center", 
                fontsize=9,
                color='black',
                fontweight='bold'
            )
        
        # Add improvement percentages for each model
        for i, model in enumerate(pivot_f1.index):
            improvement = pivot_f1.loc[model, 'improvement']
            if abs(improvement) >= 0.1:  # Only show if change is at least 0.1%
                color = 'green' if improvement > 0 else 'red'
                ax2.text(
                    i, 
                    0.05,
                    f"{improvement:+.2f}%",
                    ha="center", 
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )
        
        ax2.set_title('Model F1 Score by Defence Method after FGSM', fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1 Score')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 3: Performance metrics comparison (bottom left)
        ax3 = plt.subplot(gs[1, 0])
        
        # Create a dot plot instead of radar chart - better for direct comparisons
        # Before using precision and recall, check if they exist in the dataframe
        metrics = ['accuracy', 'f1_score']
        if 'precision' in df.columns:
            metrics.append('precision')
        if 'recall' in df.columns:
            metrics.append('recall')
            
        # Melt the dataframe to get all metrics in a single column
        df_metrics = df.melt(
            id_vars=['model', 'defence'],
            value_vars=metrics,
            var_name='metric', 
            value_name='value'
        )
        
        # Create the dot plot - split by defense method
        # First, plot 'none' defense
        sns.pointplot(
            data=df_metrics[df_metrics['defence'] == 'none'],
            x='metric',
            y='value',
            hue='model',
            palette=colors,
            markers='o',  # Circle marker
            linestyles='-',  # Solid line
            ax=ax3,
            errwidth=2,
            capsize=0.2
        )
        
        # Then plot 'feature_smoothing' defense with different markers and linestyle
        sns.pointplot(
            data=df_metrics[df_metrics['defence'] == 'feature_smoothing'],
            x='metric',
            y='value',
            hue='model',
            palette=colors,
            markers='s',  # Square marker
            linestyles='--',  # Dashed line
            ax=ax3,
            errwidth=2,
            capsize=0.2
        )
        
        # Create a custom legend that includes defense methods
        from matplotlib.lines import Line2D
        handles, labels = ax3.get_legend_handles_labels()
        defense_handles = [
            Line2D([0], [0], marker='o', color='gray', linestyle='-', markersize=8, label='No Defense'),
            Line2D([0], [0], marker='s', color='gray', linestyle='--', markersize=8, label='Feature Smoothing')
        ]
        # Remove the existing legend
        ax3.get_legend().remove()

        # Add a new legend with both model and defense information
        ax3.legend(handles=handles + defense_handles, 
                labels=labels + ['No Defense', 'Feature Smoothing'],
                title='Model & Defense',
                loc='best')
        
        ax3.set_title('Performance Metrics Comparison after FGSM', fontweight='bold')
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Score')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.set_ylim(0, 1.0)  # Set y-axis from 0 to 1 for better comparison
        
        # Add a table summarizing key findings
        table_data = []
        models = df['model'].unique()
        
        for model in models:
            model_data = df[df['model'] == model]
            
            # Safely get values, handling potential missing data
            try:
                acc_none = model_data[model_data['defence'] == 'none']['accuracy'].values[0]
                acc_fs = model_data[model_data['defence'] == 'feature_smoothing']['accuracy'].values[0]
                f1_none = model_data[model_data['defence'] == 'none']['f1_score'].values[0]
                f1_fs = model_data[model_data['defence'] == 'feature_smoothing']['f1_score'].values[0]
                
                acc_change = ((acc_fs - acc_none) / acc_none) * 100
                f1_change = ((f1_fs - f1_none) / f1_none) * 100
                
                table_data.append([model, f"{acc_change:+.2f}%", f"{f1_change:+.2f}%"])
            except (IndexError, ZeroDivisionError):
                # Handle cases where data might be missing
                table_data.append([model, "N/A", "N/A"])
        
        # Only create table if we have data
        if table_data:
            table = ax3.table(
                cellText=table_data,
                colLabels=['Model', 'Acc. Change', 'F1 Change'],
                loc='bottom',
                cellLoc='center',
                bbox=[0.0, -0.35, 1.0, 0.2]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            plt.subplots_adjust(bottom=0.25)
        
        # Plot 4: Heatmap comparison (bottom right)
        ax4 = plt.subplot(gs[1, 1])
        
        # Reshape data for heatmap
        pivot_acc = df.pivot_table(index='model', columns='defence', values='accuracy')
        pivot_f1 = df.pivot_table(index='model', columns='defence', values='f1_score')
        
        # Create labels for annotations
        def create_annotation_text(acc, f1):
            return f'Acc: {acc:.4f}\nF1: {f1:.4f}'
        
        # Create annotation labels
        annotations = np.empty_like(pivot_acc, dtype=object)
        for i in range(pivot_acc.shape[0]):
            for j in range(pivot_acc.shape[1]):
                annotations[i,j] = create_annotation_text(
                    pivot_acc.iloc[i,j], 
                    pivot_f1.iloc[i,j]
                )
        
        # Create a composite metric for coloring (e.g., average of acc and F1)
        composite_score = (pivot_acc + pivot_f1) / 2
        
        # Create a better colormap for the heatmap (yellow-green-blue)
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', 
            ['#4575b4', '#91bfdb', '#e0f3f8', '#ffffbf', '#fee090', '#fc8d59', '#d73027'][::-1]
        )
        
        # Create heatmap with improved styling
        sns.heatmap(
            composite_score, 
            annot=annotations, 
            fmt='', 
            cmap=cmap, 
            linewidths=.5, 
            cbar_kws={'label': 'Avg(Accuracy, F1)'},
            ax=ax4,
            annot_kws={"fontsize": 9, "fontweight": "bold"}
        )
        
        ax4.set_title('Performance Metrics by Model and Defence after FGSM', fontweight='bold')
        ax4.set_xlabel('Defence Method')
        ax4.set_ylabel('Model')
        
        # Add overall figure title
        plt.suptitle('Advanced Analysis of Model Performance with Defence Methods after FGSM', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Add explanatory text box
        description_text = """
        Feature Smoothing: Significantly degrades Random Forest perforance while
        having minimal impact on neural network models.
        
        Key observations:
        • Random Forest (rf) models significantly outperform other architectures
        • Neural networks remain largely unaffected by feature smoothing
        • Best overall performance: Random Forest with no defence
        """
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        fig.text(0.5, 0.02, description_text, ha='left', va='bottom', fontsize=11, 
                bbox=props, transform=fig.transFigure)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.07, 1, 0.96])
        
        # Save figure
        plt.savefig(output_file2, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Advanced performance comparison saved to {output_file2}")
        return fig
    
    def create_pgd(df):

        #Filter dataframe to only show clean results
        data_to_remove = ['fgsm', 'fgsm_smoothed', 'clean', 'clean_smoothed']
        df = df[~df['data_type'].isin(data_to_remove)]
        
        # Create colorblind-friendly palette
        # Using a colorblind-friendly palette from ColorBrewer
        colors = ['#1b9e77', '#d95f02', '#7570b3']
        model_colors = dict(zip(df['model'].unique(), colors))
        
        # Set up figure with GridSpec for complex layout
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        
        # Plot 1: Bar plot with error bars (top left)
        ax1 = plt.subplot(gs[0, 0])
        
        # Calculate improvement percentages for accuracy
        pivot_acc = df.pivot_table(index='model', columns='defence', values='accuracy')
        pivot_acc['improvement'] = (pivot_acc['feature_smoothing'] - pivot_acc['none']) / pivot_acc['none'] * 100
    
        # Create bar plot for accuracy with improved styling
        ax1 = sns.barplot(
            x='model',
            y='accuracy',
            hue='defence',
            data=df,
            palette=['#2c7fb8', '#7fcdbb'],  # Blue for none, teal for feature_smoothing
            ax=ax1,
            alpha=0.85
        )
        
        # Enhance with actual values
        for i, p in enumerate(ax1.patches):
            height = p.get_height()
            ax1.text(
                p.get_x() + p.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha="center", 
                fontsize=9,
                color='black',
                fontweight='bold'
            )
        
        # Add improvement percentages for each model
        for i, model in enumerate(pivot_acc.index):
            improvement = pivot_acc.loc[model, 'improvement']
            if abs(improvement) >= 0.1:  # Only show if change is at least 0.1%
                color = 'green' if improvement > 0 else 'red'
                ax1.text(
                    i, 
                    0.05,
                    f"{improvement:+.2f}%",
                    ha="center", 
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )
        
        ax1.set_title('Model Accuracy by Defence Method after PGD', fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Bar plot for F1 score (top right)
        ax2 = plt.subplot(gs[0, 1])
        
        # Calculate improvement percentages for F1 score
        pivot_f1 = df.pivot_table(index='model', columns='defence', values='f1_score')
        pivot_f1['improvement'] = (pivot_f1['feature_smoothing'] - pivot_f1['none']) / pivot_f1['none'] * 100
        
        # Create bar plot for f1_score with improved styling
        ax2 = sns.barplot(
            x='model',
            y='f1_score',
            hue='defence',
            data=df,
            palette=['#2c7fb8', '#7fcdbb'],  # Blue for none, teal for feature_smoothing
            ax=ax2,
            alpha=0.85
        )
        
        # Enhance with actual values
        for i, p in enumerate(ax2.patches):
            height = p.get_height()
            ax2.text(
                p.get_x() + p.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha="center", 
                fontsize=9,
                color='black',
                fontweight='bold'
            )
        
        # Add improvement percentages for each model
        for i, model in enumerate(pivot_f1.index):
            improvement = pivot_f1.loc[model, 'improvement']
            if abs(improvement) >= 0.1:  # Only show if change is at least 0.1%
                color = 'green' if improvement > 0 else 'red'
                ax2.text(
                    i, 
                    0.05,
                    f"{improvement:+.2f}%",
                    ha="center", 
                    fontsize=10,
                    color=color,
                    fontweight='bold'
                )
        
        ax2.set_title('Model F1 Score by Defence Method after PGD', fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1 Score')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 3: Performance metrics comparison (bottom left)
        ax3 = plt.subplot(gs[1, 0])
        
        # Create a dot plot instead of radar chart - better for direct comparisons
        # Before using precision and recall, check if they exist in the dataframe
        metrics = ['accuracy', 'f1_score']
        if 'precision' in df.columns:
            metrics.append('precision')
        if 'recall' in df.columns:
            metrics.append('recall')
            
        # Melt the dataframe to get all metrics in a single column
        df_metrics = df.melt(
            id_vars=['model', 'defence'],
            value_vars=metrics,
            var_name='metric', 
            value_name='value'
        )
        
        # Create the dot plot - split by defense method
        # First, plot 'none' defense
        sns.pointplot(
            data=df_metrics[df_metrics['defence'] == 'none'],
            x='metric',
            y='value',
            hue='model',
            palette=colors,
            markers='o',  # Circle marker
            linestyles='-',  # Solid line
            ax=ax3,
            errwidth=2,
            capsize=0.2
        )
        
        # Then plot 'feature_smoothing' defense with different markers and linestyle
        sns.pointplot(
            data=df_metrics[df_metrics['defence'] == 'feature_smoothing'],
            x='metric',
            y='value',
            hue='model',
            palette=colors,
            markers='s',  # Square marker
            linestyles='--',  # Dashed line
            ax=ax3,
            errwidth=2,
            capsize=0.2
        )
        
        # Create a custom legend that includes defense methods
        from matplotlib.lines import Line2D
        handles, labels = ax3.get_legend_handles_labels()
        defense_handles = [
            Line2D([0], [0], marker='o', color='gray', linestyle='-', markersize=8, label='No Defense'),
            Line2D([0], [0], marker='s', color='gray', linestyle='--', markersize=8, label='Feature Smoothing')
        ]
        # Remove the existing legend
        ax3.get_legend().remove()

        # Add a new legend with both model and defense information
        ax3.legend(handles=handles + defense_handles, 
                labels=labels + ['No Defense', 'Feature Smoothing'],
                title='Model & Defense',
                loc='best')
        
        ax3.set_title('Performance Metrics Comparison after PGD', fontweight='bold')
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Score')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.set_ylim(0, 1.0)  # Set y-axis from 0 to 1 for better comparison
        
        # Add a table summarizing key findings
        table_data = []
        models = df['model'].unique()
        
        for model in models:
            model_data = df[df['model'] == model]
            
            # Safely get values, handling potential missing data
            try:
                acc_none = model_data[model_data['defence'] == 'none']['accuracy'].values[0]
                acc_fs = model_data[model_data['defence'] == 'feature_smoothing']['accuracy'].values[0]
                f1_none = model_data[model_data['defence'] == 'none']['f1_score'].values[0]
                f1_fs = model_data[model_data['defence'] == 'feature_smoothing']['f1_score'].values[0]
                
                acc_change = ((acc_fs - acc_none) / acc_none) * 100
                f1_change = ((f1_fs - f1_none) / f1_none) * 100
                
                table_data.append([model, f"{acc_change:+.2f}%", f"{f1_change:+.2f}%"])
            except (IndexError, ZeroDivisionError):
                # Handle cases where data might be missing
                table_data.append([model, "N/A", "N/A"])
        
        # Only create table if we have data
        if table_data:
            table = ax3.table(
                cellText=table_data,
                colLabels=['Model', 'Acc. Change', 'F1 Change'],
                loc='bottom',
                cellLoc='center',
                bbox=[0.0, -0.35, 1.0, 0.2]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            plt.subplots_adjust(bottom=0.25)
        
        # Plot 4: Heatmap comparison (bottom right)
        ax4 = plt.subplot(gs[1, 1])
        
        # Reshape data for heatmap
        pivot_acc = df.pivot_table(index='model', columns='defence', values='accuracy')
        pivot_f1 = df.pivot_table(index='model', columns='defence', values='f1_score')
        
        # Create labels for annotations
        def create_annotation_text(acc, f1):
            return f'Acc: {acc:.4f}\nF1: {f1:.4f}'
        
        # Create annotation labels
        annotations = np.empty_like(pivot_acc, dtype=object)
        for i in range(pivot_acc.shape[0]):
            for j in range(pivot_acc.shape[1]):
                annotations[i,j] = create_annotation_text(
                    pivot_acc.iloc[i,j], 
                    pivot_f1.iloc[i,j]
                )
        
        # Create a composite metric for coloring (e.g., average of acc and F1)
        composite_score = (pivot_acc + pivot_f1) / 2
        
        # Create a better colormap for the heatmap (yellow-green-blue)
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', 
            ['#4575b4', '#91bfdb', '#e0f3f8', '#ffffbf', '#fee090', '#fc8d59', '#d73027'][::-1]
        )
        
        # Create heatmap with improved styling
        sns.heatmap(
            composite_score, 
            annot=annotations, 
            fmt='', 
            cmap=cmap, 
            linewidths=.5, 
            cbar_kws={'label': 'Avg(Accuracy, F1)'},
            ax=ax4,
            annot_kws={"fontsize": 9, "fontweight": "bold"}
        )
        
        ax4.set_title('Performance Metrics by Model and Defence after PGD', fontweight='bold')
        ax4.set_xlabel('Defence Method')
        ax4.set_ylabel('Model')
        
        # Add overall figure title
        plt.suptitle('Advanced Analysis of Model Performance with Defence Methods', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Add explanatory text box
        description_text = """
        Feature Smoothing: negatively impacts Random Forest performance while enhancing
        ensemble models results, neural networks are relatively unaffected.
        
        Key observations:
        • Random Forest (rf) models significantly outperform other architectures
        • The ensemble model shows positive performance changes with feature smoothing
        • Best overall performance: Random Forest with no defence
        """
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        fig.text(0.5, 0.02, description_text, ha='left', va='bottom', fontsize=11, 
                bbox=props, transform=fig.transFigure)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.07, 1, 0.96])
        
        # Save figure
        plt.savefig(output_file3, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Advanced performance comparison saved to {output_file3}")
        return fig
    
    create_cleaned(df)
    create_FGSM(df)
    create_pgd(df)


