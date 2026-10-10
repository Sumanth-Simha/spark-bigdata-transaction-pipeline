import torch
import numpy as np
from src.models.neural_net import FraudMLP
from src.train_neural_net import FocalLoss

def test_fraud_mlp_shapes():
    num_features = 30
    embedding_dims = [(24, 4)]

    model = FraudMLP(num_features, embedding_dims)

    # Batch size 16
    x_num = torch.randn(16, num_features)
    x_cat = torch.randint(0, 24, (16, 1))

    out = model(x_num, x_cat)

    # Should output logits for 16 samples
    assert out.shape == (16, 1)

def test_focal_loss_computation():
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # Predict high probability for positive class, low for negative class
    # logits: > 0 means prob > 0.5, < 0 means prob < 0.5
    inputs = torch.tensor([[5.0], [-5.0], [0.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

    loss = criterion(inputs, targets)

    # Loss should be computable and positive
    assert loss.item() > 0
    assert not torch.isnan(loss).any()

def test_threshold_logic():
    # Simulate probabilities and threshold logic
    probs = np.array([0.1, 0.4, 0.6, 0.9])

    threshold_low = 0.5
    flags_low = probs >= threshold_low
    assert np.array_equal(flags_low, [False, False, True, True])

    threshold_high = 0.8
    flags_high = probs >= threshold_high
    assert np.array_equal(flags_high, [False, False, False, True])
