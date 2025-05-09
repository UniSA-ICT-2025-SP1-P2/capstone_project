
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

#Load data from csv
def load_csv_data(filename="defence_results.csv"):
    """
    Load data from a CSV file in the same GitHub folder.
    
    Parameters:
    filename (str): Name of the CSV file to read, defaults to 'defence_results.csv'
    
    Returns:
    pandas.DataFrame: The data from the CSV file
    """
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

#Table visualisation of defence results
def visualise_defence_results(csv_file="defence_results.csv", output_file="results_table_simple.png"):
    """
    Alternative method to export table as PNG without requiring dataframe_image.
    Uses matplotlib to create and export the table.
    
    Parameters:
    csv_file (str): Path to the CSV file
    output_file (str): Path to save the table image
    """
    try:
        df = pd.read_csv(csv_file)

        #Filter dataframe to only show fgsm results
        data_to_remove = ['fgsm', 'fgsm_smoothed', 'pgd', 'pgd_smoothed']
        df = df[~df['data_type'].isin(data_to_remove)]
        
        # Format numeric columns to 4 decimal places
        for col in ['accuracy', 'f1_score']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
        
        # Get index of max and min values before conversion to string
        accuracy_max_idx = pd.to_numeric(df['accuracy']).idxmax()
        accuracy_min_idx = pd.to_numeric(df['accuracy']).idxmin()
        f1_max_idx = pd.to_numeric(df['f1_score']).idxmax()
        f1_min_idx = pd.to_numeric(df['f1_score']).idxmin()
        
        # Create figure and axis - size based on data dimensions
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Hide axes
        ax.axis('off')
        
        # Create custom table without using matplotlib's table function
        col_labels = df.columns
        n_rows, n_cols = df.shape
        
        # Create cell text data
        cell_text = df.values.tolist()
        
        # Add a title
        plt.title('Model Defence Results', fontsize=16, pad=20)
        
        # Create a table without scaling
        table = plt.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colColours=['#e6e6e6'] * n_cols
        )
        
        # Set font size directly
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        
        # Add cell colors for max/min values
        for i in range(n_rows):
            # Colors for accuracy column
            if i == accuracy_max_idx:
                table[(i+1, col_labels.get_loc('accuracy'))].set_facecolor('lightgreen')
            if i == accuracy_min_idx:
                table[(i+1, col_labels.get_loc('accuracy'))].set_facecolor('lightsalmon')
                
            # Colors for f1_score column
            if i == f1_max_idx:
                table[(i+1, col_labels.get_loc('f1_score'))].set_facecolor('lightgreen')
            if i == f1_min_idx:
                table[(i+1, col_labels.get_loc('f1_score'))].set_facecolor('lightsalmon')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Table successfully exported to {output_file}")
        
    except Exception as e:
        print(f"Error in export_simple_table_to_png: {e}")
        import traceback
        traceback.print_exc()

def visualise_results_graph_simple(csv_file="defence_results.csv", metric="both"):
    """
    Visualise the results from the CSV file as a graph.
    
    Parameters:
    csv_file (str): Path to the CSV file
    metric (str): Which metric to plot - 'accuracy', 'f1_score', or 'both'
    
    Returns:
    matplotlib.figure.Figure: The generated figure
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        #Filter dataframe to only show fgsm results
        data_to_remove = ['fgsm', 'fgsm_smoothed', 'pgd', 'pgd_smoothed']
        df = df[~df['data_type'].isin(data_to_remove)]
        
        # Create a figure and axes
        if metric == "both":
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            metrics = ['accuracy', 'f1_score']
            titles = ['Accuracy Comparison', 'F1 Score Comparison']
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
            metrics = [metric]
            titles = [f'{metric.capitalize()} Comparison']
        
        # Plot the data
        for i, (m, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            sns.barplot(
                data=df,
                x='model',
                y=m,
                hue='defence',
                palette='viridis',
                ax=ax
            )
            ax.set_title(title)
            ax.set_ylim(0, max(df[m]) * 1.1)  # Add some space above the highest bar
            ax.set_xlabel('Model')
            ax.set_ylabel(m.replace('_', ' ').title())
            
            # Add value labels on top of bars
            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.4f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    rotation=45
                )
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error visualising results graph: {e}")
        return None

def create_advanced_performance_comparison(csv_file="defence_results.csv", output_file="advanced_performance_comparison.png"):
    """
    Create a sophisticated comparison of model performance across metrics
    
    Parameters:
    csv_file (str): Path to the CSV file containing performance data
    output_file (str): Path to save the visualization
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)

    #Filter dataframe to only show fgsm results
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Advanced performance comparison saved to {output_file}")
    return fig

def create_defence_impact_analysis(csv_file="defence_results.csv", output_file="defence_impact_analysis.png"):
    """
    Create visualization showing the impact of defence mechanisms on performance
    
    Parameters:
    df (pandas.DataFrame): Input data
    output_file (str): Path to save the visualization
    """

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Calculate the impact of defence as percentage change
    impact_data = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Get baseline metrics (no defence)
        baseline = model_data[model_data['defence'] == 'none'].iloc[0]
        baseline_acc = baseline['accuracy']
        baseline_f1 = baseline['f1_score']
        
        # Get defence metrics
        defence_rows = model_data[model_data['defence'] != 'none']
        
        for _, defence_row in defence_rows.iterrows():
            defence_name = defence_row['defence']
            acc_change = ((defence_row['accuracy'] - baseline_acc) / baseline_acc) * 100
            f1_change = ((defence_row['f1_score'] - baseline_f1) / baseline_f1) * 100
            
            impact_data.append({
                'model': model,
                'defence': defence_name,
                'accuracy_change': acc_change,
                'f1_score_change': f1_change
            })
    
    # Create DataFrame from impact data
    impact_df = pd.DataFrame(impact_data)
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Custom diverging colormap for positive/negative changes
    colors = sns.color_palette("RdBu_r", n_colors=11)
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    # Plot accuracy changes
    ax0 = axes[0]
    sns.barplot(
        data=impact_df,
        x='model',
        y='accuracy_change',
        hue='defence',
        palette='viridis',
        ax=ax0
    )
    
    ax0.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax0.grid(axis='y', linestyle='--', alpha=0.7)
    ax0.set_title('Impact of Defence on Accuracy (%)', fontweight='bold')
    ax0.set_xlabel('Model')
    ax0.set_ylabel('Percentage Change in Accuracy')
    
    # Add value labels
    for p in ax0.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax0.text(
                p.get_x() + p.get_width()/2.,
                height + (1 if height >= 0 else -3),
                f'{height:.2f}%',
                ha="center", 
                fontsize=9
            )
    
    # Plot F1 score changes
    ax1 = axes[1]
    sns.barplot(
        data=impact_df,
        x='model',
        y='f1_score_change',
        hue='defence',
        palette='viridis',
        ax=ax1
    )
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_title('Impact of Defence on F1 Score (%)', fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Percentage Change in F1 Score')
    
    # Add value labels
    for p in ax1.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax1.text(
                p.get_x() + p.get_width()/2.,
                height + (1 if height >= 0 else -3),
                f'{height:.2f}%',
                ha="center", 
                fontsize=9
            )
    
    # Add overall figure title
    plt.suptitle('Impact Analysis of Defence Mechanisms on Model Performance', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add explanation
    fig.text(0.5, 0.01, 
            "This visualization shows how implementing defence mechanisms affects model performance.\n"
            "Negative values indicate performance degradation compared to no defence.", 
            ha='center', fontsize=12, style='italic')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Defence impact analysis saved to {output_file}")
    return fig

def create_model_comparison_report(csv_file="defence_results.csv", output_file="model_comparison_report.png"):
    """
    Create a comprehensive visual report comparing models
    
    Parameters:
    df (pandas.DataFrame): Input data
    output_file (str): Path to save the visualization
    """

    # Read the CSV file
    df = pd.read_csv(csv_file)

    #Filter dataframe to only show fgsm results
    data_to_remove = ['fgsm', 'fgsm_smoothed', 'pgd', 'pgd_smoothed']
    df = df[~df['data_type'].isin(data_to_remove)]

    # Aggregate metrics by model
    model_metrics = df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-index columns
    model_metrics.columns = ['model', 'accuracy_mean', 'accuracy_std', 'f1_mean', 'f1_std']
    
    # Calculate combined performance score
    model_metrics['performance_score'] = (model_metrics['accuracy_mean'] + model_metrics['f1_mean']) / 2
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Plot 1: Bar chart with error bars for accuracy
    ax1 = plt.subplot(gs[0, 0])
    
    # Plot accuracy with error bars
    ax1.bar(
        model_metrics['model'],
        model_metrics['accuracy_mean'],
        yerr=model_metrics['accuracy_std'],
        capsize=10,
        color=sns.color_palette("viridis", len(model_metrics)),
        alpha=0.7
    )
    
    # Add actual values on top of bars
    for i, (_, row) in enumerate(model_metrics.iterrows()):
        ax1.text(
            i,
            row['accuracy_mean'] + row['accuracy_std'] + 0.01,
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}",
            ha='center',
            fontsize=9,
            rotation=0
        )
    
    ax1.set_title('Average Accuracy by Model', fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Bar chart with error bars for F1 score
    ax2 = plt.subplot(gs[0, 1])
    
    # Plot F1 score with error bars
    ax2.bar(
        model_metrics['model'],
        model_metrics['f1_mean'],
        yerr=model_metrics['f1_std'],
        capsize=10,
        color=sns.color_palette("viridis", len(model_metrics)),
        alpha=0.7
    )
    
    # Add actual values on top of bars
    for i, (_, row) in enumerate(model_metrics.iterrows()):
        ax2.text(
            i,
            row['f1_mean'] + row['f1_std'] + 0.01,
            f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f}",
            ha='center',
            fontsize=9,
            rotation=0
        )
    
    ax2.set_title('Average F1 Score by Model', fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('F1 Score')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Overall performance score
    ax3 = plt.subplot(gs[1, 0])
    
    # Create horizontal sorted bar chart
    sorted_metrics = model_metrics.sort_values('performance_score', ascending=True)
    bars = ax3.barh(
        sorted_metrics['model'],
        sorted_metrics['performance_score'],
        color=sns.color_palette("viridis", len(model_metrics)),
        alpha=0.8
    )
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax3.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{width:.4f}",
            ha='left',
            va='center',
            fontsize=10
        )
    
    ax3.set_title('Combined Performance Score by Model', fontweight='bold')
    ax3.set_xlabel('Performance Score (Higher is Better)')
    ax3.set_ylabel('Model')
    ax3.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Plot 4: Defense effectiveness across models
    ax4 = plt.subplot(gs[1, 1])
    
    # Calculate defense effectiveness
    defense_comparison = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Get data for each defense type
        no_defense = model_data[model_data['defence'] == 'none']
        with_defense = model_data[model_data['defence'] != 'none']
        
        if not no_defense.empty and not with_defense.empty:
            no_def_acc = no_defense['accuracy'].values[0]
            no_def_f1 = no_defense['f1_score'].values[0]
            
            # Get mean metrics with defense
            def_acc = with_defense['accuracy'].mean()
            def_f1 = with_defense['f1_score'].mean()
            
            # Calculate percentage impact
            acc_impact = ((def_acc - no_def_acc) / no_def_acc) * 100
            f1_impact = ((def_f1 - no_def_f1) / no_def_f1) * 100
            
            defense_comparison.append({
                'model': model,
                'acc_impact': acc_impact,
                'f1_impact': f1_impact
            })
    
    # Convert to DataFrame
    defense_df = pd.DataFrame(defense_comparison)
    
    # Melt for seaborn
    defense_melt = pd.melt(
        defense_df,
        id_vars=['model'],
        value_vars=['acc_impact', 'f1_impact'],
        var_name='metric',
        value_name='impact'
    )
    
    # Replace column names for better labels
    defense_melt['metric'] = defense_melt['metric'].replace({
        'acc_impact': 'Accuracy Impact',
        'f1_impact': 'F1 Score Impact'
    })
    
    # Create grouped bar chart for defense impact
    sns.barplot(
        data=defense_melt,
        x='model',
        y='impact',
        hue='metric',
        palette=['#3498db', '#e74c3c'],
        ax=ax4
    )
    
    # Add reference line at 0
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for p in ax4.patches:
        height = p.get_height()
        if not np.isnan(height):
            sign = "+" if height > 0 else ""
            ax4.text(
                p.get_x() + p.get_width()/2.,
                height + (0.5 if height >= 0 else -2),
                f"{sign}{height:.2f}%",
                ha="center", 
                fontsize=9
            )
    
    ax4.set_title('Defense Impact on Model Performance', fontweight='bold')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Percentage Impact (%)')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall figure title
    plt.suptitle('Comprehensive Model Performance Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Add methodology note
    fig.text(0.5, 0.02, 
            "Performance Score = Average of Accuracy and F1 Score metrics\n"
            "Defense Impact = % change in performance with defense vs. without defense", 
            ha='center', fontsize=11, style='italic')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison report saved to {output_file}")
    return fig
