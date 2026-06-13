import matplotlib.pyplot as plt
import numpy as np

# Set global styling
plt.style.use('bmh')

def create_confusion_matrix():
    # 1. Confusion Matrix
    cm = np.array([[24096, 6615], 
                   [848, 9031]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues', alpha=0.8)
    
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            if i==0 and j==0: label = "True Negatives\n(Baseline Suppressions)"
            elif i==0 and j==1: label = "False Positives\n(Transient Alerts)"
            elif i==1 and j==0: label = "False Negatives\n(Missed Migrations)"
            elif i==1 and j==1: label = "True Positives\n(Saved Sessions)"
            
            # Determine text color based on background darkness
            color = 'black'
            ax.text(j, i, f"{val:,}\n\n{label}", ha='center', va='center', 
                    color=color, fontsize=11, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Safe', 'Predicted SLA Crash'], fontsize=11)
    ax.set_yticklabels(['Actual Safe', 'Actual SLA Crash'], fontsize=11, rotation=90, va='center')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title('Test Fold Extrapolations (40,590 Events)', pad=15, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('patent_fig1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_chart():
    # 2. Key Metrics Bar Chart
    metrics = ['AUC-ROC', 'PR-AUC', 'F2-Score (Optimized)']
    scores = [0.8968, 0.7605, 0.8710]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, scores, color=['#3498db', '#9b59b6', '#e74c3c'], width=0.5, edgecolor='black', linewidth=1.5)
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Performance Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Scientific Validation Metrics (Unseen Data)', fontsize=14, fontweight='bold')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig('patent_fig2_validation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hysteresis_chart():
    # 3. Hysteresis Debouncing Conceptual Plot
    time_steps = np.arange(1, 16) # 15-step sequence
    
    # Synthetic risk index showing noise spike vs real failure cascade
    risk_index = [0.05, 0.08, 0.12, 0.25, 0.15, 0.10, 0.08, 0.18, 0.24, 0.28, 0.35, 0.45, 0.60, 0.80, 0.90]
    threshold = 0.20

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the Risk Index
    ax.plot(time_steps, risk_index, marker='o', linestyle='-', color='#2c3e50', linewidth=2.5, markersize=8, label='Ensemble Risk Index')
    
    # Plot the Threshold Line
    ax.axhline(y=threshold, color='#e74c3c', linestyle='--', linewidth=2, label='F2 Threshold (0.20)')
    
    # Highlight transient noise (Tick 4)
    ax.scatter(4, 0.25, color='orange', s=200, zorder=5, label='Transient Noise (Migration Suppressed)')
    ax.annotate('1 Tick\nSuppressed', xy=(4, 0.26), xytext=(4, 0.4), 
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), 
                ha='center', fontweight='bold')
    
    # Highlight true failure trigger (Tick 10 - the 2nd tick above threshold)
    ax.axvspan(10, 15, color='#e74c3c', alpha=0.15, label='Migration Danger Zone (Action Executed)')
    ax.scatter(10, 0.28, color='red', s=200, zorder=5, label='Hysteresis Trigger (2 Consecutive Ticks)')
    ax.annotate('Migration\nTriggered', xy=(10, 0.29), xytext=(10, 0.5), 
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8), 
                ha='center', fontweight='bold', color='red')
    
    ax.set_xlabel('Time Intervals (Rolling Window)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SLA Violation Probability', fontsize=12, fontweight='bold')
    ax.set_title('Fig 3: Temporal Hysteresis Filtering & Signal Debouncing', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')
    ax.set_xticks(time_steps)
    
    plt.tight_layout()
    plt.savefig('patent_fig3_hysteresis_logic.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_confusion_matrix()
    create_metrics_chart()
    create_hysteresis_chart()
    print("Successfully generated 3 patent-ready infographics.")
