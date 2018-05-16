import numpy as np
from eval_metric import EvalMetric

class CompositeEvalMetric(EvalMetric):
    """Manages multiple evaluation metrics.

    Parameters
    ----------
    metrics : list of EvalMetric
        List of child metrics.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> eval_metrics_1 = mx.metric.Accuracy()
    >>> eval_metrics_2 = mx.metric.F1()
    >>> eval_metrics = mx.metric.CompositeEvalMetric()
    >>> for child_metric in [eval_metrics_1, eval_metrics_2]:
    >>>     eval_metrics.add(child_metric)
    >>> eval_metrics.update(labels = labels, preds = predicts)
    >>> print eval_metrics.get()
    (['accuracy', 'f1'], [0.6666666666666666, 0.8])
    """

    def __init__(self, metrics=None, name='composite',
                 output_names=None, label_names=None):
        super(CompositeEvalMetric, self).__init__(
            'composite', output_names=output_names, label_names=label_names)
        if metrics is None:
            metrics = []
        self.metrics = [create(i) for i in metrics]

    def add(self, metric):
        """Adds a child metric.

        Parameters
        ----------
        metric
            A metric instance.
        """
        self.metrics.append(metric)

    def get_metric(self, index):
        """Returns a child metric.

        Parameters
        ----------
        index : int
            Index of child metric in the list of metrics.
        """
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update_dict(self, labels, preds): # pylint: disable=arguments-differ
        if self.label_names is not None:
            labels = OrderedDict([i for i in labels.items()
                                  if i[0] in self.label_names])
        if self.output_names is not None:
            preds = OrderedDict([i for i in preds.items()
                                 if i[0] in self.output_names])

        for metric in self.metrics:
            metric.update_dict(labels, preds)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        """Returns the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            if isinstance(name, str):
                name = [name]
            if isinstance(value, (float, int, np.generic, long)):
                value = [value]
            names.extend(name)
            values.extend(value)
        return (names, values)

    def get_config(self):
        config = super(CompositeEvalMetric, self).get_config()
        config.update({'metrics': [i.get_config() for i in self.metrics]})
        return config
