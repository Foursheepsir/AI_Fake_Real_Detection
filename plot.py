import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

def plot_1(tfidf, embedding, name):
    classes = ['human-fake', 'AI-fake', 'true']

    all_scores = []
    for i in range(3): 
        all_scores.extend([
            tfidf['precision'][i], embedding['precision'][i],
            tfidf['recall'][i],    embedding['recall'][i],
            tfidf['f1'][i],        embedding['f1'][i]
        ])

    # position parameters
    group_gap = 0.2          # gap between groups of 6 bars
    bar_width = 0.1          # width of each bar
    intra_gap = 0.02         # gap between bars in the same group

    # calculate x positions for each bar
    x_positions = []
    for i in range(len(classes)):  
        group_start = i * (6 * (bar_width + intra_gap) + group_gap)
        for j in range(6): 
            pos = group_start + j * (bar_width + intra_gap)
            x_positions.append(pos)

    # set colors for each bar
    colors = [
        '#1f77b4', '#aec7e8',  # TF/EM Precision
        '#ff7f0e', '#ffbb78',  # TF/EM Recall
        '#2ca02c', '#98df8a'   # TF/EM F1
    ] * 3 

    fig, ax = plt.subplots(figsize=(7.5, 3.2))

    ax.bar(x_positions, all_scores, width=bar_width, color=colors)
    ax.tick_params(axis='y', labelsize=14)
    tick_pos = [
        i * (6 * (bar_width + intra_gap) + group_gap) + 2.5 * (bar_width + intra_gap)
        for i in range(len(classes))
    ]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(classes, fontsize=14)

    # axis settings
    ax.set_ylim(10, 100)
    ax.set_ylabel("Score (%)", fontsize=14)
    ax.set_title(name + ": TF-IDF vs Embedding", fontsize=18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    # legend settings
    legend_elements = [
        Patch(facecolor=colors[0], label='TF-IDF Precision'),
        Patch(facecolor=colors[1], label='Embedding Precision'),
        Patch(facecolor=colors[2], label='TF-IDF Recall'),
        Patch(facecolor=colors[3], label='Embedding Recall'),
        Patch(facecolor=colors[4], label='TF-IDF F1'),
        Patch(facecolor=colors[5], label='Embedding F1'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(name + "_tfidf_embed_metrics.png")  

def plot_bert_scores(bert_df, name="BERT"):
    classes = ['human-fake', 'AI-fake', 'true']
    metrics = ['precision', 'recall', 'f1']

    # obtain scores from the DataFrame
    all_scores = []
    for cls in classes:
        for metric in metrics:
            all_scores.append(bert_df.loc[metric][cls] * 100)

    # position parameters
    group_gap = 0.3
    bar_width = 0.15
    intra_gap = 0.05

    # x positions for each bar
    x_positions = []
    for i in range(len(classes)):
        group_start = i * (3 * (bar_width + intra_gap) + group_gap)
        for j in range(3):
            x_positions.append(group_start + j * (bar_width + intra_gap))

    # colors for each bar
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] * len(classes)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x_positions, all_scores, width=bar_width, color=colors)
    ax.tick_params(axis='y', labelsize=14)

    tick_pos = [
        i * (3 * (bar_width + intra_gap) + group_gap) + 1 * (bar_width + intra_gap)
        for i in range(len(classes))
    ]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(classes, fontsize=14)

    # axis settings
    ax.set_ylim(95, 100.5)
    ax.set_ylabel("Score (%)", fontsize=14)
    ax.set_title(name + " Classification Scores", fontsize=18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    # legend settings
    legend_elements = [
        Patch(facecolor=colors[0], label='Precision'),
        Patch(facecolor=colors[1], label='Recall'),
        Patch(facecolor=colors[2], label='F1 Score'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14)

    plt.tight_layout()
    plt.savefig(name + "_bert_metrics.png")

if __name__ == '__main__':
    tfidf_logreg = {
        "precision": [96.5, 97.7, 94.6],
        "recall": [94.5, 87.8, 97.8],
        "f1": [95.5, 92.5, 96.2],
    }

    embedding_logreg = {
        "precision": [93.8, 81.8, 91.4],
        "recall": [93.0, 71.4, 94.2],
        "f1": [93.4, 76.2, 92.8],
    }

    tfidf_SVM = {
        "precision": [97.1, 97.6, 95.9],
        "recall": [95.6, 93.0, 95.2],
        "f1": [96.3, 95.2, 96.9],
    }

    embedding_SVM = {
        "precision": [93.1, 84.6, 91.7],
        "recall": [93.6, 69.6, 94.2],
        "f1": [93.3, 76.4, 92.9],
    }

    tfidf_RF = {
        "precision": [94.4, 99.3, 89.7],
        "recall": [91.2, 66.2, 97.4],
        "f1": [92.7, 79.5, 93.4],
    }

    embedding_RF = {
        "precision": [86.6, 94.1, 76.8],
        "recall": [81.1, 12.8, 92.5],
        "f1": [83.8, 22.5, 83.9],
    }
    tfidf_data = [tfidf_logreg, tfidf_SVM, tfidf_RF]
    embedding_data = [embedding_logreg, embedding_SVM, embedding_RF]
    name = ['Logistic_Regression', 'SVM', 'Random_Forest']
    for i in range(3):
        plot_1(tfidf_data[i], embedding_data[i], name[i])

    bert_df = pd.DataFrame({
        'human-fake': [0.9960, 0.9974, 0.9967],
        'AI-fake': [0.9978, 0.9951, 0.9964],
        'true': [0.9952, 0.9984, 0.9968]
    }, index=['precision', 'recall', 'f1'])

    # Plot tfidf and embedding data 
    for i in range(3):
        plot_1(tfidf_data[i], embedding_data[i], name[i])

    # Plot BERT scores
    plot_bert_scores(bert_df, name="BERT")
