from eval_metric import EvalMetric

class AccMetric(EvalMetric):
    def __init__(self, pred_names, label_names):
        super(AccMetric, self).__init__('Acc')
        self.pred_names = pred_names
        self.label_names = label_names

    def update(self, preds, labels):
        pred = preds[self.pred_names.index('cls_prob')]
        pred_label = pred.argmax(dim=1)
        self.sum_metric += (pred_label == labels).sum().cpu().numpy()
        self.num_inst += labels.shape[0]
