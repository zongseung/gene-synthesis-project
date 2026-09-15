import torch

from src.models.modules.conditioning import UnifiedFiLMGenerator
from src.models.modules.cnn import CNNDecoder, FiLMConvBlock
from src.models.modules.dit import DiTBlock


def test_adaln_zero_is_identity_and_attention_gate_gets_gradient():
    torch.manual_seed(0)
    block = DiTBlock(d_model=4, n_heads=1, mlp_ratio=1.0)
    x = torch.randn(2, 3, 4)
    modulation = torch.zeros(2, 24, requires_grad=True)

    output = block(x, modulation)
    output.square().mean().backward()

    assert torch.equal(output, x)
    assert modulation.grad is not None
    assert torch.count_nonzero(modulation.grad[:, 8:12]) > 0


def test_attention_weights_update_after_zero_gate_opens():
    torch.manual_seed(1)
    generator = UnifiedFiLMGenerator(
        d_model=4,
        d_time=4,
        cnn_channels=[4],
        cnn_dec_channels=[4],
        n_dit_blocks=1,
        d_dit=4,
    )
    block = DiTBlock(d_model=4, n_heads=1, mlp_ratio=1.0)
    optimizer = torch.optim.SGD(
        [*generator.parameters(), *block.parameters()], lr=0.1
    )
    pop_emb = torch.randn(2, 4)
    timesteps = torch.tensor([1, 2])
    x = torch.randn(2, 3, 4)
    target = torch.zeros_like(x)
    initial_attention = block.attn.in_proj_weight.detach().clone()
    initial_modulation = generator(pop_emb, timesteps)[2][0]

    assert torch.count_nonzero(initial_modulation) == 0

    for step in range(3):
        optimizer.zero_grad()
        modulation = generator(pop_emb, timesteps)[2][0]
        torch.nn.functional.mse_loss(block(x, modulation), target).backward()
        if step == 0:
            gate_bias_grad = generator.dit_films[0][1].bias.grad[8:12]
            assert torch.count_nonzero(gate_bias_grad) > 0
        optimizer.step()

    assert torch.count_nonzero(generator(pop_emb, timesteps)[2][0]) > 0
    assert not torch.equal(block.attn.in_proj_weight, initial_attention)


def test_encoder_downsample_forwards_the_residual_sum():
    block = FiLMConvBlock(1, 1, downsample=True)
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv1.bias.zero_()
        block.conv2.weight.zero_()
        block.conv2.bias.zero_()
        block.downsample.weight.fill_(1.0)
        block.downsample.bias.zero_()
    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    gamma = torch.zeros(1, 1)
    beta = torch.zeros(1, 1)

    output, skip = block(x, gamma, beta)

    torch.testing.assert_close(skip, x)
    torch.testing.assert_close(output, block.downsample(skip))


def test_decoder_upsamples_to_each_skip_length_without_padding():
    decoder = CNNDecoder(
        out_channels=1,
        base_channels=4,
        channel_mult=(1, 1, 2, 2, 4),
    )
    x = torch.randn(1, 16, 2)
    skips = [
        torch.randn(1, 4, 32),
        torch.randn(1, 4, 16),
        torch.randn(1, 8, 8),
        torch.randn(1, 8, 4),
        torch.randn(1, 16, 2),
    ]
    film = [
        (torch.zeros(1, channels), torch.zeros(1, channels))
        for channels in decoder.block_out_channels
    ]
    upsampled_lengths = []
    handles = [
        block.upsample.register_forward_hook(
            lambda _module, _inputs, output: upsampled_lengths.append(output.shape[-1])
        )
        for block in decoder.blocks
    ]

    try:
        output = decoder(x, skips, film)
    finally:
        for handle in handles:
            handle.remove()

    assert upsampled_lengths == [2, 4, 8, 16, 32]
    assert output.shape[-1] == 32
