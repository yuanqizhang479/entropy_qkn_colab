import math
import numpy as np
import pytest
import torch
from qknlab.evaluate import score_tokens


class Uniform(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(*x.shape, 7, device=x.device)


@pytest.mark.parametrize('length,cap,batch', [(44,None,3),(17,9,2),(6,None,8),(44,40,1)])
def test_exact_target_accounting(length, cap, batch):
    total, n = score_tokens(Uniform(), np.arange(length) % 7, 8, batch, cap)
    assert n == min(length-1, cap if cap is not None else length-1)
    assert total/n == pytest.approx(math.log(7), abs=3e-7)
