import torch
import normflows as nf


class NormalizingFlow(torch.nn.Module):
    num_layers = 32
    latent_size = 2

    def __init__(self, flows=None):
        super().__init__()
        self.flows = flows if flows is not None else self._create_flows()

    def _create_flows(self):
        """Creates a flow according to Real NVP paper."""
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.latent_size)])
        flows = []
        for i in range(self.num_layers):
            s = nf.nets.MLP([self.latent_size, 16 * self.latent_size, 16 * self.latent_size, 16 * self.latent_size, self.latent_size], init_zeros=True)
            t = nf.nets.MLP([self.latent_size, 16 * self.latent_size, 16 * self.latent_size, 16 * self.latent_size, self.latent_size], init_zeros=True)
            if i % 2 == 0:
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
            flows += [nf.flows.ActNorm(self.latent_size)]

        return torch.nn.ModuleList(flows)
    
    def forward(self, x):
        for flow in self.flows:
            x = flow(x)[0]
        return x
