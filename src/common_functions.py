def run_sweetviz_report(df, TARGET):
    
    import sweetviz as sv
    from datetime import datetime
    
    report_label = datetime.today().strftime('%Y-%m-%d_%H_%M')
    
    my_report = sv.analyze(df,target_feat=TARGET)
    my_report.show_html(filepath='SWEETVIZ_' + report_label + '.html')
    
    return

def plot_corr_barchart(df1, drop_cols, n=30):
    """
    plot a color-gradient barchart showing top n correlations between features

    """

    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    df1 = df1.drop(columns=drop_cols)
    useful_columns =  df1.select_dtypes(include=['number']).columns

    def get_redundant_pairs(df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0,df.shape[1]):
            for j in range(0,i+1):
                pairs_to_drop.add((cols[i],cols[j]))
        return pairs_to_drop

    def get_correlations(df,n=n):
        au_corr = df.corr(method = 'spearman').unstack() #spearman used because not all data is normalized
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels = labels_to_drop).sort_values(ascending=False)
        top_n = au_corr[0:n]    
        bottom_n =  au_corr[-n:]
        top_corr = pd.concat([top_n, bottom_n])
        return top_corr

    corrplot = get_correlations(df1[useful_columns])


    fig, ax = plt.subplots(figsize=(15,10))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax =1)
    colors = [plt.cm.RdYlGn(norm(c)) for c in corrplot.values]

    print(corrplot)

    corrplot.plot.barh(color=colors)
    
    return


def plot_corr_vs_target(df1, target, drop_cols, n=30):
    
    """
    plot a color-gradient barchart showing top n correlations between features and target

    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    
    target_series = df1[target]
    df1 = df1.drop(columns=drop_cols)
    
    x = df1.corrwith(target_series, method = 'spearman',numeric_only=True).sort_values(ascending=False)
    top_n = x[0:n]    
    bottom_n =  x[-n:]
    top_corr = pd.concat([top_n, bottom_n])
    x = top_corr

    print(x)

    fig, ax = plt.subplots(figsize=(15,10))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax =1)
    colors = [plt.cm.RdYlGn(norm(c)) for c in x.values]
    x.plot.barh(color=colors)
    
    return


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    return fig

def run_sweetviz_report(df, TARGET):
    
    import sweetviz as sv
    from datetime import datetime
    
    report_label = datetime.today().strftime('%Y-%m-%d_%H_%M')
    
    my_report = sv.analyze(df,target_feat=TARGET)
    my_report.show_html(filepath='SWEETVIZ_' + report_label + '.html')
    
    return