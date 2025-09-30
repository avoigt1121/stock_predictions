"""
Custom loss functions for financial time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Loss function that penalizes wrong directional predictions more heavily
    This is crucial for financial applications where direction matters more than magnitude
    """
    
    def __init__(self, mse_weight=0.5, directional_weight=0.5):
        super(DirectionalLoss, self).__init__()
        self.mse_weight = mse_weight
        self.directional_weight = directional_weight
        
    def forward(self, predictions, targets):
        # MSE component
        mse_loss = F.mse_loss(predictions, targets)
        
        # Directional component
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        
        # Penalty for wrong direction
        directional_accuracy = (pred_direction == target_direction).float()
        directional_loss = 1.0 - directional_accuracy.mean()
        
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.directional_weight * directional_loss)
        
        return total_loss


class HuberLoss(nn.Module):
    """
    Huber loss is less sensitive to outliers than MSE
    Good for financial data which often has extreme outliers
    """
    
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, predictions, targets):
        return F.huber_loss(predictions, targets, delta=self.delta)


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic predictions
    Useful when you want to predict confidence intervals
    """
    
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
        
    def forward(self, predictions, targets):
        errors = targets - predictions
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss focuses on hard examples
    Good when you have imbalanced data (more up days than down days)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions, targets):
        # Convert to classification problem (up/down)
        pred_prob = torch.sigmoid(predictions)
        target_class = (targets > 0).float()
        
        # Focal loss calculation
        pt = target_class * pred_prob + (1 - target_class) * (1 - pred_prob)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        bce_loss = F.binary_cross_entropy(pred_prob, target_class, reduction='none')
        
        return (focal_weight * bce_loss).mean()


def get_loss_function(loss_type='mse', **kwargs):
    """
    Factory function to get loss functions
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return HuberLoss(delta=kwargs.get('delta', 1.0))
    elif loss_type == 'directional':
        return DirectionalLoss(
            mse_weight=kwargs.get('mse_weight', 0.5),
            directional_weight=kwargs.get('directional_weight', 0.5)
        )
    elif loss_type == 'quantile':
        return QuantileLoss(quantile=kwargs.get('quantile', 0.5))
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test different loss functions
    batch_size = 32
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)
    
    print("Testing loss functions:")
    
    # MSE
    mse_loss = get_loss_function('mse')
    print(f"MSE Loss: {mse_loss(predictions, targets).item():.6f}")
    
    # Directional Loss
    dir_loss = get_loss_function('directional')
    print(f"Directional Loss: {dir_loss(predictions, targets).item():.6f}")
    
    # Huber Loss
    huber_loss = get_loss_function('huber')
    print(f"Huber Loss: {huber_loss(predictions, targets).item():.6f}")
    
    # Calculate directional accuracy
    pred_direction = torch.sign(predictions)
    target_direction = torch.sign(targets)
    directional_accuracy = (pred_direction == target_direction).float().mean()
    print(f"Directional Accuracy: {directional_accuracy.item():.4f}")
