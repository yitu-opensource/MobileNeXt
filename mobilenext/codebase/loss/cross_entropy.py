import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        # import pdb;pdb.set_trace()
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LearnableLabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, n_component=3):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LearnableLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.n_classes = 1000
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.sampler=[]
        self.n_component = n_component
        for i in range(n_component+1):
            u = i/n_component
            self.sampler.append(torch.distributions.normal.Normal(u,0.1))
        # self.sampler1 = torch.distributions.normal.Normal(0,0.1)
        # self.sampler2 = torch.distributions.normal.Normal(smoothing,0.1)
        # self.sampler3 = torch.distributions.normal.Normal(-smoothing,0.1)

        self.alpha = Parameter(torch.ones(n_component, requires_grad=True))
        self.mean_pos = Parameter(torch.tensor([i/n_component for i in range(n_component+1)], requires_grad=True))
    def gen_dist(self,probs, pos):
        lr_dist = 0
        for i in range(len(probs)):
            pos_idx = int(pos[i] * self.n_component // 1)
            pos_frac = pos[i] * self.n_component - pos_idx
            # import pdb;pdb.set_trace()
            start_pos = pos_idx
            stop_pos = start_pos + 1 if start_pos < len(self.sampler) else -1
            lr_dist += probs[i] * (pos_frac * self.sampler[start_pos].sample([self.n_classes]).cuda() + (1-pos_frac) * self.sampler[stop_pos].sample([self.n_classes]).cuda())
        return lr_dist
    def forward(self, x, target):
        logsoftmax = nn.LogSoftmax()
        target = torch.unsqueeze(target, 1)
        soft_target = torch.zeros_like(x)
        soft_target.scatter_(1, target, 1)
        
        probs = F.softmax(self.alpha, dim=0)
        # lr_dist = probs[0]*self.sampler1.sample([self.n_classes]).cuda() + \
        #           probs[1]*self.sampler2.sample([self.n_classes]).cuda() + \
        #           probs[2]*self.sampler3.sample([self.n_classes]).cuda()
        lr_dist = self.gen_dist(probs, self.mean_pos)
        soft_target = soft_target * self.confidence + self.smoothing * lr_dist / self.n_classes
        # import pdb;pdb.set_trace()
        
        return torch.mean(torch.sum(- soft_target * logsoftmax(x), 1))
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
