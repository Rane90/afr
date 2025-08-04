import torch


def cross_entropy(*_):
    def criterion(logits, y, *_):
        xe = torch.nn.functional.cross_entropy(logits, y, reduction='none')
        loss = xe.mean()
        return loss

    return criterion

def afr(*_):
    def criterion(logits, y, weights):
        return afr_fn(logits, y, weights)
    return criterion

def afr_fn(logits, y, weights):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    out = weights * ce
    return out.sum()

def get_exp_weights(logits, y, gamma):
    p = logits.softmax(-1)
    y_onehot = torch.zeros_like(logits).scatter_(-1, y.unsqueeze(-1), 1)
    p_true = (p * y_onehot).sum(-1)
    weights = (-gamma * p_true).exp()
    return weights

def focal_loss(args):
    def criterion(logits, y):
        return focal_loss_fn(logits, y, gamma=args.gamma, alpha=args.alpha_focal)
    return criterion


def focal_loss_fn(logits, y, gamma=2.0, alpha=None):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    
    y_onehot = torch.zeros_like(logits).scatter_(-1, y.unsqueeze(-1), 1)
    p_t = (probs * y_onehot).sum(dim=-1)
    log_p_t = (log_probs * y_onehot).sum(dim=-1)

    if alpha is not None:
        alpha_t = alpha[y]
        loss = -alpha_t * ((1 - p_t) ** gamma) * log_p_t
    else:
        loss = -((1 - p_t) ** gamma) * log_p_t

    return loss.mean()
