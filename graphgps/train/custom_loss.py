from itertools import product

import torch
import numpy as np
import diffsort
from torch import nn, Tensor
from torch_geometric.graphgym.config import cfg
from graphgps.train.fast_soft_sort.pytorch_ops import soft_rank
from loguru import logger


def ranking(y: torch.Tensor):
    assert y.ndim == 2, f"{y.ndim} != 2"
    ord_idx = torch.sort(y, dim=-1).indices
    rank = torch.arange(0, y.size(1)).to(y.device) # .repeat(y.size(0), 1)
    new_y = torch.zeros_like(y, dtype=rank.dtype)
    for i, order in enumerate(ord_idx):
        new_y[i][order] = rank
    return new_y


def apply_rank_loss(y_pred, y_true, train=True):
    kwargs = {
        "adaptive": cfg.train.adap_margin,
        "train": train,
        "weight_by_diff": False,
        "weight_by_diff_var": True,
    }
    
    if cfg.model.loss_fun == 'hinge':
        loss = pairwise_hinge_loss_batch(y_pred, y_true, **kwargs)
    elif cfg.model.loss_fun == 'listmle':
        loss = listMLE(y_pred, y_true, **kwargs)
    elif cfg.model.loss_fun == 'ranknet':
        loss = rankNet(y_pred, y_true, **kwargs)
    elif cfg.model.loss_fun == 'approx_ndcg':
        loss = approxNDCGLoss(y_pred, ranking(y_true).float(), **kwargs)
    elif cfg.model.loss_fun == 'neural_ndcg':
        loss = neuralNDCG(y_pred, ranking(y_true).float(), **kwargs)
    elif cfg.model.loss_fun == 'neural_sort':
        loss = neural_sort(y_pred, ranking(y_true), **kwargs)
    elif cfg.model.loss_fun == 'diffsort':
        loss = diffsort_loss(y_pred, ranking(y_true), **kwargs)
    else:
        logger.warning(f"Getting a unknown loss function setting: {cfg.model.loss_fun}, fallback to hinge loss")
        cfg.model.loss_fun = 'hinge'
        loss = pairwise_hinge_loss_batch(y_pred, y_true, **kwargs)
    return loss


def apply_regression_loss(y_pred, y_true, model):
    y_true = y_true.float()
    # if cfg.train.regression.val_min >= 0:
    #     y_true -= cfg.train.regression.val_min
    #     if cfg.train.regression.val_max > cfg.train.regression.val_min:
    #         scope = cfg.train.regression.val_max - cfg.train.regression.val_min
    #         y_true = (y_true / scope) * 100
    pred = model.reg_scale * y_pred + model.reg_offset
    return nn.functional.mse_loss(pred, y_true)


def apply_pair_rank_loss(y_pred, y_true, train=True):
    if y_true.ndim == 2:
        y_true = y_true.squeeze(0)
    mask = y_true >= 0
    loss = nn.functional.cross_entropy(y_pred[mask], y_true[mask], reduction='none')
    return loss.sum()


def pairwise_hinge_loss_batch(pred, true, base_margin=0.1, adaptive=False, **kwargs):
    # pred: (batch_size, num_preds )
    # true: (batch_size, num_preds)
    batch_size = pred.shape[0]
    num_preds = pred.shape[1]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)

    pairwise_true = true[:,i_idx] - true[:,j_idx] > 1e-9
    mask = torch.logical_and(true[:,i_idx] >= 0,  true[:,j_idx] >= 0)
    if adaptive:
        fp_true = true.float()
        step = (fp_true.var(dim=1, keepdim=True)**0.5)
        step = torch.clip(step, min=1e-5)
        step = step * 6 / 10
        pairwise_scale = nn.functional.relu(true[:,i_idx] - true[:,j_idx]) * pairwise_true / step
        pairwise_scale = torch.clip(pairwise_scale, min=0, max=10)
        margin = (pairwise_scale + 1) * base_margin
    else:
        margin = base_margin

    loss = nn.functional.relu(margin - (pred[:,i_idx] - pred[:,j_idx]))
    loss = loss * pairwise_true.float() * mask.float()
    loss = torch.sum(loss) / batch_size
    return loss


def rankNet(
        y_pred, y_true, padded_value_indicator=-1, 
        weight_by_diff=False, weight_by_diff_powed=False, weight_by_diff_var=False, **kwargs):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone().float()
    y_rank = ranking(y_true).float()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    pairs_rank = y_rank[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    rank_diffs = pairs_rank[:, :, 0] - pairs_rank[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 1e-9) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(rank_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_rank[:, :, 0], 2) - torch.pow(pairs_rank[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_var:
        fp_true = y_true.float()
        step = (fp_true.var(dim=1, keepdim=True)**0.5)
        step = torch.clip(step, min=1e-5)
        step = step* 6 / 10
        weight = torch.clip(torch.abs(true_diffs) / step, min=1, max=10)[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.functional.binary_cross_entropy_with_logits(
        pred_diffs, true_diffs, weight=weight
    )


def listMLE(y_pred, y_true, eps=1e-7, padded_value_indicator=-1, **kwargs):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    
    ref: https://github.com/allegro/allRank/blob/master/allrank/models/losses/listMLE.py
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))


def approxNDCGLoss(y_pred, y_true, eps=1e-6, padded_value_indicator=-1, alpha=1., **kwargs):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1e-6)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=1e-6)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def deterministic_neural_sort(s, tau, mask, softmax=True):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length] {batch, class, samples}
    """
    dev = s.device

    n = s.size()[1]
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))  # pairwise difference
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    one = torch.ones((n, n), dtype=torch.float32, device=dev)
    B = torch.matmul(A_s, one)  # B[i, j] = A_s[i].sum()

    temp = [
        (n - m + 1) - 2 * (torch.arange(n - m, device=dev) + 1) 
        for m in mask.squeeze(-1).sum(dim=1)
    ]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]  # padding masked part
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))  # score[i], rank multiplier scaling[j], pairwise prod

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    if softmax:
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / tau)
        return P_hat
    else:
        return P_max


def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """
    def sample_gumbel(samples_shape, device, eps=1e-10) -> torch.Tensor:
        """
        Sampling from Gumbel distribution.
        Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
        Minor modifications applied to the original code (masking).
        :param samples_shape: shape of the output samples tensor
        :param device: device of the output samples tensor
        :param eps: epsilon for the logarithm function
        :return: Gumbel samples tensor of shape samples_shape
        """
        U = torch.rand(samples_shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)
    
    dev = s.device

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=-1):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=-1, discount=True):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    if discount:
        discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0))
        discounts = discounts.to(device=true_sorted_by_preds.device)
    else:
        discounts = 1.0

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def neuralNDCG(y_pred, y_true, padded_value_indicator=-1, temperature=1., powered_relevancies=False, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True, discount=False, **kwargs):
    """
    NeuralNDCG loss introduced in "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable
    Relaxation of Sorting" - https://arxiv.org/abs/2102.07831. Based on the NeuralSort algorithm.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param temperature: temperature for the NeuralSort algorithm
    :param powered_relevancies: whether to apply 2^x - 1 gain function, x otherwise
    :param k: rank at which the loss is truncated
    :param stochastic: whether to calculate the stochastic variant
    :param n_samples: how many stochastic samples are taken, used if stochastic == True
    :param beta: beta parameter for NeuralSort algorithm, used if stochastic == True
    :param log_scores: log_scores parameter for NeuralSort algorithm, used if stochastic == True
    :return: loss value, a torch.Tensor
    """
    dev = y_pred.device

    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    # Choose the deterministic/stochastic variant
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

    # Mask P_hat and apply to true labels, ie approximately sort them
    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    if discount:
        discounted_gains = ground_truth * discounts
    else:
        discounted_gains = ground_truth

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)  # Ideal DCG
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + 1e-6)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)

    assert (ndcg < 0.).sum() >= 0, "every ndcg should be non-negative"
    if idcg_mask.all():
        return torch.tensor(0.)

    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore
    if mean_ndcg.isnan().any():
        breakpoint()
    return -1. * mean_ndcg  # -1 cause we want to maximize NDCG


def neural_sort(y_pred, y_true, padded_value_indicator=-1, **kwargs):
    # k = torch.arange(y_pred.shape[-1])
    # batch = y_pred.shape[0]
    # order = torch.sort(y_true, dim=-1).indices
    # for b in range(batch):
    #     y_pred[b] = y_pred[b][order[b]]

    mask = (y_true == padded_value_indicator)
    P_hat = deterministic_neural_sort(
        y_pred.unsqueeze(-1), 
        tau=1, 
        mask=mask, 
        softmax=False
    )

    # loss = []
    # for i in range(y_pred.shape[-1]):
    #     if i == 0: continue
    #     # cut = P_hat[:, i, :i + 1]
    #     # local_label = torch.zeros([batch], device=y_pred.device, dtype=torch.int64)
    #     cut = P_hat[:, i, ~i:]
    #     local_label = torch.zeros([batch], device=y_pred.device, dtype=torch.int64)
    #     loss.append(nn.functional.cross_entropy(cut, local_label))
    # loss = torch.stack(loss).mean()
    # return loss
    
    n = y_true.size(1)
    # P_hat = torch.permute(P_hat, [0, 2, 1])

    P_true = deterministic_neural_sort(
        y_true.unsqueeze(-1).float(),
        tau=1, 
        mask=mask, 
        softmax=True
    )
    # P_true = torch.permute(P_true, [0, 2, 1])
    # P_true[0, :, 1].argmax()
    return nn.functional.cross_entropy(P_hat, P_true)
    # return nn.functional.cross_entropy(P_hat, n - y_true - 1)


SORTERS = None

def init_sorters():
    global SORTERS
    SORTERS = {
        'train': diffsort.DiffSortNet(
            sorting_network_type='odd_even',
            size=cfg.dataset.num_sample_config,
            device=cfg.device,
            steepness=10,
            art_lambda=0.25,
        ),
        'val': diffsort.DiffSortNet(
            sorting_network_type='odd_even',
            size=cfg.dataset.eval_num_sample_config,
            device=cfg.device,
            steepness=10,
            art_lambda=0.25,
        ),
    }


def diffsort_loss(y_pred, y_true, train=True, **kwargs):
    global SORTERS
    if SORTERS is None:
        init_sorters()
    sorter = SORTERS['train'] if train else SORTERS['val']
    perm_ground_truth = torch.nn.functional.one_hot(
        torch.argsort(y_true, dim=-1)).transpose(-2, -1).float()
    _, perm_prediction = sorter(y_pred)
    
    # loss = torch.nn.BCELoss()(perm_prediction, perm_ground_truth)
    loss = nn.functional.cross_entropy(
        torch.permute(perm_prediction, [0, 2, 1]), 
        y_true,
        reduction='sum',
    )
    # loss = nn.functional.binary_cross_entropy(
    #     torch.permute(perm_prediction, [0, 2, 1]), 
    #     perm_ground_truth,
    #     reduction='sum',
    # )
    return loss