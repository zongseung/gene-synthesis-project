from src.training.trainer import cosine_warmup_lr_lambda


def test_warmup_is_linear_then_cosine_decays_to_zero():
    assert cosine_warmup_lr_lambda(0, warmup=10, max_iters=110) == 0.0
    assert cosine_warmup_lr_lambda(5, warmup=10, max_iters=110) == 0.5
    assert cosine_warmup_lr_lambda(10, warmup=10, max_iters=110) == 1.0
    assert abs(cosine_warmup_lr_lambda(60, warmup=10, max_iters=110) - 0.5) < 1e-12
    assert abs(cosine_warmup_lr_lambda(110, warmup=10, max_iters=110)) < 1e-12
