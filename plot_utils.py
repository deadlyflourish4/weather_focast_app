import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure matplotlib uses a non-interactive backend suitable for background tasks
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for saving figures without display

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, output_path: Optional[Path] = None):
    """
    This function plots the confusion matrix and saves it to a file if output_path is provided.
    Normalization can be applied by setting `normalize=True`.

    Args:
        y_true: array-like of shape (n_samples,)
        y_pred: array-like of shape (n_samples,)
        classes: array-like of shape (n_classes,)
        normalize: bool, default=False
        title: str, default=None
        cmap: Colormap, default=plt.cm.Blues
        output_path: Path object or str, optional. If provided, saves the plot here.

    Returns:
        matplotlib.axes.Axes: The axes object with the confusion matrix plot.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm_normalized) # Replace NaN with 0
        logger.info("Normalized confusion matrix calculated.")
    else:
        logger.info('Confusion matrix calculated, without normalization.')

    logger.debug(f"Confusion Matrix values:\n{cm}")

    fig, ax = plt.subplots(figsize=(10, 10)) # Give it a bit more space
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2. if cm.size > 0 else 0.1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if output_path:
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix plot to {output_path}: {e}")
    else:
        logger.warning("Confusion matrix plot generated but not saved (no output_path provided).")
        # If you absolutely needed to show it interactively (not recommended in background):
        # plt.show()

    # Close the plot to free up memory, important in background tasks
    plt.close(fig)

    return ax # Returning axes might be less useful now, but keep signature consistent