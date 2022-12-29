class Our(nn.Module):
    def __init__(self, reduction='mean', weights=None):
        super(Our, self).__init__()
        self.reduction = reduction
        self.weights = weights

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  
            logits = logits.transpose(1, 2)  
            logits = logits.contiguous().view(-1, logits.size(2))  
        target = target.view(-1, 1)  

        log_pt = F.log_softmax(logits, 1)
        log_pt = log_pt.gather(1, target).view(-1).to(device)  
        pt = torch.exp(log_pt)
        per_cls_weights = (self.weights.to(device)).gather(0, target.data.view(-1))

        loss = - (torch.exp(- 1 * pt)) * log_pt * per_cls_weights
        loss = loss.mean()

        return loss