import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx

import nltk
from nltk import ngrams

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch

from collections import Counter
from pprint import pprint
import ast
import time
import json
import re
import os


def interactive_2d_histogram_violin(df: pd.DataFrame, numeric_column: str, category_column: str):
    # Create interactive histogram
    hist_fig = px.histogram(
        df,
        x=numeric_column,
        color=category_column,
        opacity=0.6,
        barmode='overlay',
        title=f'Distribution of {numeric_column} by {category_column}'
    )

    # Create interactive violin plot
    violin_fig = px.violin(
        df,
        x=category_column,
        y=numeric_column,
        box=True,
        points='all',
        title=f'Violin Plot of {numeric_column} by {category_column}'
    )

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Distribution of {numeric_column} by {category_column}',
            f'Violin Plot of {numeric_column} by {category_column}'
        )
    )

    # Add histogram traces to subplot 1
    for trace in hist_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Add violin plot traces to subplot 2
    for trace in violin_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        width=1200,
        height=600,
        showlegend=True,
        legend_title_text=category_column,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Update axes titles
    fig.update_xaxes(title_text=numeric_column, row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)

    fig.update_xaxes(title_text=category_column, row=1, col=2)
    fig.update_yaxes(title_text=numeric_column, row=1, col=2)

    # Show the interactive plot
    fig.show()


def top_values_by_classes(df: pd.DataFrame, values_col: str, classes_col: str):
    classes = df[classes_col].unique()
    
    # Dictionary to hold language counts per class
    class_language_counts = {}
    
    # Compute language counts per class
    for c in classes:
        class_df = df[df[classes_col] == c]
        language_counts = class_df[values_col].value_counts().reset_index()
        language_counts.columns = [values_col, 'count']
        language_counts[classes_col] = c
        class_language_counts[c] = language_counts
    
    # Prepare data for plotting
    plot_data = []
    for c in classes:
        class_df = class_language_counts[c]
        plot_data.append(class_df)
    
    # Combine all class data
    combined_df = pd.concat(plot_data)
    
    # Set up the matplotlib figure without shared y-axis
    fig, axes = plt.subplots(1, len(classes), figsize=(15, 8))
    
    # If only one class, make axes iterable
    if len(classes) == 1:
        axes = [axes]
    
    # Color palette
    palette = sns.color_palette('hsv', len(classes))
    
    # Plotting
    for idx, c in enumerate(classes):
        class_df = combined_df[combined_df[classes_col] == c]
        class_df = class_df.sort_values(by='count', ascending=True)
        ax = axes[idx]
        sns.barplot(x='count', y=values_col, data=class_df, palette=[palette[idx]]*len(class_df), ax=ax)
        ax.set_title(f'Language Distribution - {c.capitalize()}')
        ax.set_xlabel('Number of Occurrences')
        ax.set_ylabel(values_col)
        # Add counts over the bars
        for i, row in enumerate(class_df.itertuples()):
            ax.text(row.count + 0.1, i, str(row.count), va='center')
        # Invert y-axis to have the highest count at the top
        ax.invert_yaxis()
        # Ensure y-axis labels are visible
        ax.tick_params(axis='y', which='both', labelleft=True)
    
    plt.tight_layout()
    plt.show()


def top_k_tokens_by_class(df: pd.DataFrame, tokens_col: str, classes_col: str, k: int = 20):
    # Get unique classes
    classes = df[classes_col].unique()
    
    # Dictionary to hold word counts per class
    class_word_counts = {}
    
    # Process text and compute word frequencies per class
    for c in classes:
        class_df = df[df[classes_col] == c]
        all_tokens = []
        for index, row in class_df.iterrows():
            all_tokens.extend(row[tokens_col])
        word_counts = Counter(all_tokens)
        top_words = word_counts.most_common(k)
        class_word_counts[c] = top_words
    
    # Prepare data for plotting
    plot_data = []
    for c in classes:
        top_words = class_word_counts[c]
        class_df = pd.DataFrame(top_words, columns=['word', 'count'])
        class_df['class'] = c
        plot_data.append(class_df)
    
    # Combine all class data
    combined_df = pd.concat(plot_data)
    
    # Set up the matplotlib figure without shared y-axis
    fig, axes = plt.subplots(1, len(classes), figsize=(18, 8))
    
    # If only one class, make axes iterable
    if len(classes) == 1:
        axes = [axes]
    
    # Color palette
    palette = sns.color_palette('hsv', len(classes))
    
    # Plotting
    for idx, c in enumerate(classes):
        class_df = combined_df[combined_df['class'] == c]
        class_df = class_df.sort_values(by='count', ascending=True)
        ax = axes[idx]
        sns.barplot(x='count', y='word', data=class_df, palette=[palette[idx]]*len(class_df), ax=ax)
        ax.set_title(f'Top {k} Words - {c.capitalize()}')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Words')
        # Add counts over the bars
        for i, row in enumerate(class_df.itertuples()):
            ax.text(row.count + 0.1, i, str(row.count), va='center')
        # Invert y-axis to have the highest count at the top
        ax.invert_yaxis()
        # Ensure y-axis labels are visible
        ax.tick_params(axis='y', which='both', labelleft=True)
    
    plt.tight_layout()
    plt.show()


def generate_wordcloud(text):
    return WordCloud(width=400, height=400, background_color='white').generate(text)


def generate_token_cloud_by_class(df: pd.DataFrame, tokens_column: str, classes_col: str):        
    df_tokens = df[[tokens_column, classes_col]]
    df_tokens['token_words'] = df_tokens.apply(lambda row: ' '.join(row[tokens_column]), axis=1)
    
    text_col   = 'token_words'
    target_col = classes_col

    classes   = df[classes_col].unique()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate word clouds for each text and plot them
    for i, cls in enumerate(classes):
        class_words = " ".join(df_tokens[df_tokens[target_col]==cls][text_col].astype(str))
        
        class_wordcloud = generate_wordcloud(class_words)
        axes[i].imshow(class_wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'{cls} Word Cloud')
        
    plt.tight_layout()
    plt.show()


def get_ngrams(token_list: list, n: int) -> list:
    if isinstance(token_list, str):
        token_list = token_list.split(' ')
    return list(ngrams(token_list, n))

def get_top_20_ngrams(df: pd.DataFrame, text_col: str, class_col: str, n_values: list, classes: list) -> dict:
    top_ngrams_per_class_per_N = {}
    
    for N in n_values:
        top_ngrams_per_class = {}
        for cls in classes:
            # Filter the DataFrame for the current class
            class_df = df[df[class_col] == cls]
            
            # Get all token lists and flatten them
            tokens = [
                token
                for sublist in class_df[text_col]
                for token in (sublist if isinstance(sublist, list) else sublist.split())
            ]

            # Generate N-grams
            ngram_list = get_ngrams(tokens, N)
            
            # Count frequencies
            ngram_freq = Counter(ngram_list)
            
            # Get the top 20 N-grams
            top_ngrams = ngram_freq.most_common(20)
            
            top_ngrams_per_class[cls] = top_ngrams
        top_ngrams_per_class_per_N[N] = top_ngrams_per_class
    return top_ngrams_per_class_per_N
    

def plot_top_ngrams(top_ngrams_per_class_per_N: dict, n_values: list, classes: list, class_colors: dict, figsize=(6, 6)):
    # Set up the subplots grid
    nrows = len(n_values)
    ncols = len(classes)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))

    # Make axes iterable if there's only one row or column
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    # Plotting
    for i, N in enumerate(n_values):
        for j, cls in enumerate(classes):
            ax = axes[i][j]
            ngrams_data = top_ngrams_per_class_per_N.get(N, {}).get(cls, [])
            if ngrams_data:
                ngrams_list, counts = zip(*ngrams_data)
                ngrams_strings = [' '.join(ngram) for ngram in ngrams_list]  # Convert tuples to strings
                
                # Plot horizontal bar chart with class-specific color
                y_pos = range(len(ngrams_strings))
                color = class_colors.get(cls, 'skyblue')  # Default to 'skyblue' if class not in class_colors
                ax.barh(y_pos, counts, align='center', color=color)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(ngrams_strings, fontsize=9)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Frequency')
                ax.set_title(f'Top {N}-grams for {cls}')
                
                # Add counts to the bars
                for k, v in enumerate(counts):
                    ax.text(v + max(counts)*0.01, k, str(v), color='black', va='center', fontsize=8)
            else:
                ax.set_visible(False)  # Hide axes if there's no data

    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.05, right=0.95)

    # Show the plot
    plt.show()


def viz_tsne_3d_clusetrs_by_class(df: pd.DataFrame, vector_col: str, target_col: str, perplexity: int = 30, height: int = 100):
    X = np.array(df[vector_col].tolist())
    
    # Step 1: Perform t-SNE to reduce dimensions to 3
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    
    # Step 2: Create a DataFrame with the t-SNE results and the target labels
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2', 'Dim3'])
    df_tsne[target_col] = df[target_col].values
    
    # Step 3: Plot the interactive 3D scatter plot
    fig = px.scatter_3d(
        df_tsne, x='Dim1', y='Dim2', z='Dim3',
        color=target_col,
        title='3D t-SNE Visualization',
        labels={'Dim1': 'Dimension 1', 'Dim2': 'Dimension 2', 'Dim3': 'Dimension 3'}
    )

    # Adjust the layout to increase the figure's height
    fig.update_layout(
        height=height,  # Set the height here
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        margin=dict(l=0, r=0, b=50, t=50)  # Adjust margins as needed
    )
    
    fig.show()
      
def visualize_cooccurrence(cooccurrence_matrix, top_n=50, ax=None, title='', background_color='white'):
    if ax is None:
        ax = plt.gca()

    # Create a background image
    bg_array = np.full((10, 10, 3), matplotlib.colors.to_rgb(background_color))
    ax.imshow(bg_array, extent=[-1.1, 1.1, -1.1, 1.1], aspect='auto')

    # Build the graph
    G = nx.Graph()
    # Flatten co-occurrence into edge list
    edges = []
    for word, counts in cooccurrence_matrix.items():
        for context_word, count in counts.items():
            edges.append((word, context_word, count))
    # Sort edges by weight (count)
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]
    # Add weighted edges to the graph
    G.add_weighted_edges_from(edges)

    # Draw graph
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title(title, fontsize=12)
    # Remove ticks and spines
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Set limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


def visualize_pos_frequencies(df: pd.DataFrame, pos_col: str, calsses_col: str, class_label: str = None, ax=None):
    if class_label:
        df_filtered = df[df[calsses_col] == class_label]
        title = f'POS Tag Frequency in {class_label.capitalize()} Inquiries'
    else:
        df_filtered = df
        title = 'Overall POS Tag Frequency'

    all_tags = [tag for tags in df_filtered[pos_col] for tag in tags]
    pos_counts = Counter(all_tags)

    pos_df = pd.DataFrame(pos_counts.items(), columns=['POS Tag', 'Count'])
    pos_df = pos_df.sort_values(by='Count', ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(y='POS Tag', x='Count', data=pos_df, palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Count')
    ax.set_ylabel('POS Tag')
    ax.tick_params(axis='y', labelsize=10)

    # Add data labels to each bar
    for p in ax.patches:
        count = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(count + max(pos_df['Count']) * 0.01, y, int(count), va='center')


def visualize_top_n_ner_entities(df: pd.DataFrame, entities_col: str, classes_col: str, class_label: str = None, top_n: int = 10, ax=None):
    if class_label:
        entities_list = df[df[classes_col] == class_label][entities_col]
        title = f'Top {top_n} Entities in {class_label} Inquiries'
    else:
        entities_list = df[entities_col]
        title = f'Top {top_n} Entities'
    
    # Aggregate entities
    all_entities = []
    for entities in entities_list:
        all_entities.extend(entities)

    cls_label = class_label if class_label else 'the dataset'
    
    if not all_entities:
        print(f"No entities found for {cls_label}.")
        return
    
    entities_df = pd.DataFrame(all_entities)
    
    if entities_df.empty:
        print(f"No entities to display for {cls_label}.")
        return
    
    # Count entity texts
    entity_text_counts = entities_df['text'].value_counts().nlargest(top_n).reset_index()
    entity_text_counts.columns = ['Entity', 'Count']
    entity_text_counts = entity_text_counts.sort_values(by='Count', ascending=False)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(y='Entity', x='Count', data=entity_text_counts, palette='coolwarm', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Count')
    ax.set_ylabel('Entity')
    ax.tick_params(axis='y', labelsize=10)

    # Add data labels to each bar
    for p in ax.patches:
        count = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(count + max(entity_text_counts['Count']) * 0.01, y, int(count), va='center')


def visualize_top_tfidf_words(tfidf_df, class_label, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    data = tfidf_df.sort_values(by='score_diff', ascending=False)
    
    # Plot
    sns.barplot(x='score_diff', y='term', data=data, palette='viridis', ax=ax)
    ax.set_title(f'Top TF-IDF Words for {class_label.capitalize()}')
    ax.set_xlabel('TF-IDF Score Difference')
    ax.set_ylabel('Term')
    
    # Add data labels
    for p in ax.patches:
        score = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(score + max(data['score_diff']) * 0.01, y, f'{score:.4f}', va='center')


def visualize_tfidf_results(top_words_per_class):
    classes = list(top_words_per_class.keys())
    num_classes = len(classes)
    ncols = 2
    nrows = math.ceil(num_classes / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 10, nrows * 6))
    axes = axes.flatten()
    
    for idx, cls in enumerate(classes):
        ax = axes[idx]
        visualize_top_tfidf_words(top_words_per_class[cls], cls, ax=ax)
    
    # Hide any unused subplots
    for idx in range(len(classes), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution_hist(df: pd.DataFrame, classes_col: str, sentiment_col: str, max_cols: int = 2, bins: int = 20, figsize: tuple = (12, 4), palette: str = 'husl'):
    # Get unique classes and determine the grid size
    classes = df[classes_col].unique()
    classes = np.append(classes, ['all'])
    num_classes = len(classes)
    num_cols = min(max_cols, num_classes)
    num_rows = (num_classes + num_cols - 1) // num_cols  # Ceiling division

    # Create a color palette
    colors = sns.color_palette(palette, num_classes)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=True)
    axes = np.array(axes).reshape(-1)  # Flatten the axes array

    # Plot histograms for each class
    for idx, (cls, color) in enumerate(zip(classes, colors)):
        data = df if cls == 'all' else df[df[classes_col] == cls]
        ax = axes[idx]
        sns.histplot(
            data=data,
            x=sentiment_col,
            bins=bins,
            kde=False,
            color=color,
            ax=ax
        )
      
        ax.set_title(f'Class: {cls}')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.set_xlim(-1, 1)  # Assuming sentiment scores are between -1 and 1

    # Remove any unused subplots
    for idx in range(len(classes), len(axes)):
        fig.delaxes(axes[idx])

    plt.show()


def plot_category_distribution_by_class_barchart(df: pd.DataFrame, classes_col: str, categories_col: str, max_cols: int = 2, figsize: tuple = (12, 4), palette: str = 'husl'):
    # Get unique classes and determine the grid size
    classes = df[classes_col].unique()
    classes = np.append(classes, ['all'])  # Add 'all' to plot the overall distribution
    num_classes = len(classes)
    num_cols = min(max_cols, num_classes)
    num_rows = (num_classes + num_cols - 1) // num_cols  # Ceiling division

    # Create a color palette (or default if not provided)
    colors = sns.color_palette("husl", num_classes)

    # Get the unique category order from the entire dataset
    category_order = df[categories_col].unique()

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # If only one subplot, make sure axes is a list to iterate over
    if num_classes == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()  # Flatten the axes array to iterate easily

    # Plot histograms for each class
    for idx, (cls, color) in enumerate(zip(classes, colors)):
        data = df if cls == 'all' else df[df[classes_col] == cls]
        ax = axes[idx]
        
        sns.countplot(
            data=data,
            x=categories_col,
            ax=ax,
            color=color,
            order=category_order  # Set consistent category order
        )
      
        ax.set_title(f'Class: {cls}')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Frequency')
        
        # Rotate x-axis labels for readability
        ax.tick_params(axis='x', rotation=45)

    # Remove any unused subplots
    for idx in range(len(classes), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

def plot_pca_class_distribution(df: pd.DataFrame, classes_col: str, n_components: int = 2):
    # Dynamically select all columns except the target class column
    features = [col for col in df.columns if col != classes_col]
    
    # Standardize the features
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)
    
    # Perform PCA to reduce to n_components dimensions
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with the PCA components and the target class
    pca_columns = ['PC' + str(i+1) for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)
    pca_df[classes_col] = df[classes_col].values
    
    # Plotting
    if n_components == 2:
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color=classes_col,
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            title='2D PCA of Term Scores Distribution'
        )
        fig.show()
    
    elif n_components == 3:
        fig = px.scatter_3d(
            pca_df, x='PC1', y='PC2', z='PC3', color=classes_col,
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
            title='3D PCA of Term Scores Distribution'
        )
        fig.show()

    else:
        raise ValueError("n_components must be either 2 or 3")


def plot_distributions(df: pd.DataFrame, target_col: str, exclude_cols: list = []):
    cols_to_plot = [col for col in df.columns if col not in exclude_cols]
    
    n_cols = 3
    n_rows = int(np.ceil(len(cols_to_plot) / n_cols))

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(19, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        # Plot target col
        if col == target_col:
            value_counts = df[col].value_counts()
            ax.bar(value_counts.index, value_counts.values)
            ax.set_title(f'Distribution of {target_col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')

        # Plot int64 data as bar chart if unique values <= 10
        elif df[col].dtype == 'int64' and df[col].nunique() <= 10:
            value_counts = df.groupby(col)[target_col].value_counts().unstack()
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Distribution of {col} by {target_col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Plot float or other numeric data as histogram
        elif pd.api.types.is_numeric_dtype(df[col]):
            for label in df[target_col].unique():
                df[df[target_col] == label][col].plot(kind='hist', alpha=0.8, ax=ax, label=label)
            ax.set_title(f'Distribution of {col} by {target_col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')

        # Plot categorical data            
        elif isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'object':
            df.groupby(col)[target_col].value_counts(normalize=False).unstack().plot(kind='bar', stacked=False, ax=ax)
            ax.set_title(f'Distribution of {col} by {target_col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.legend(title=target_col)
    plt.show()


def scatter_plot_matrix(df: pd.DataFrame, num_features: list):
    scatter_matrix_plot = scatter_matrix(df[num_features], alpha=0.8, figsize=(40, 40), diagonal='kde')
    
    for ax in scatter_matrix_plot.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=12, rotation=45)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.show()


def correlation_heatmap_plot(df: pd.DataFrame, num_features: list, fig_size: tuple = (12, 8)):
    # Calculate correlation matrix
    corr = df[num_features].corr()

    plt.figure(figsize=fig_size)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', mask=mask)
    plt.show()



def plot_boxplots(df: pd.DataFrame, num_features: list, max_charts_per_line: int = 4, **kwargs):
    max_charts_per_line = max_charts_per_line
    num_features = list(num_features)
    
    # Calculate number of rows and columns needed
    n_cols = min(max_charts_per_line, len(num_features))
    n_rows = math.ceil(len(num_features) / max_charts_per_line)
    
    figsize = kwargs.get('figsize', (5 * n_cols, 6 * n_rows))
    title = kwargs.get('title', 'Boxplots of Numeric Features')
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else axes

    for i, feature in enumerate(num_features):
        df.boxplot(column=feature, ax=axes[i])
        axes[i].set_title(f'Boxplot of {feature}')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
