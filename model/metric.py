import torch
from rouge import Rouge

rouge = Rouge()
metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l
    """
    # if unit == 'word':
    #     source = jieba.cut(source, HMM=False)
    #     target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target, unit='word'):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics

def rouge_1(source, target):
    source, target = ' '.join(source), ' '.join(target)
    scores = rouge.get_scores(hyps=source, refs=target)
    return scores[0]['rouge-1']['f']

def rouge_2(source, target):
    source, target = ' '.join(source), ' '.join(target)
    scores = rouge.get_scores(hyps=source, refs=target)
    return scores[0]['rouge-2']['f']

def rouge_l(source, target):
    source, target = ' '.join(source), ' '.join(target)
    scores = rouge.get_scores(hyps=source, refs=target)
    return scores[0]['rouge-l']['f']