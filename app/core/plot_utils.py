import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        y_true: array-like of shape (n_samples,)
            True labels for predictions.
        y_pred: array-like of shape (n_samples,)
            Predicted labels, as returned by a classifier.
        classes: array-like of shape (n_classes,)
            List of names of the classes to index the matrix.
        normalize: bool, default=False
            Whether to normalize the confusion matrix.
        title: str, default=None
            Title for the plot.
        cmap: Colormap, default=plt.cm.Blues
            Colormap for the matrix plot.

    Returns:
        matplotlib.axes.Axes: The axes object with the confusion matrix plot.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        # Calculate normalization, handling potential division by zero for rows with no true samples
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm_normalized) # Replace NaN with 0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(8, 8)) # Adjust figure size if needed
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Set y-axis limits to avoid cutting off top/bottom cells with newer matplotlib versions
    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2. if cm.size > 0 else 0.1 # Avoid error if cm is empty
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show() # Display the plot
    return ax