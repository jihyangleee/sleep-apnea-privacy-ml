import torch
from typing import List, Tuple


def additive_split(x: torch.Tensor, n: int = 3) -> List[torch.Tensor]:
    """Additive secret share: returns n shares that sum to x."""
    noise = [torch.randn_like(x) for _ in range(n - 1)]
    last  = x - sum(noise)
    return noise + [last]


def apply_dp_noise(share: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian DP noise injected into a share before transmission. sigma=0 disables."""
    if sigma <= 0.0:
        return share
    return share + torch.randn_like(share) * sigma


class BeaverProvider:
    """Simulated Beaver Triple dealer for MPC multiplication on secret shares.

    In production triples are generated offline via Oblivious Transfer (OT) or
    a trusted hardware module — no trusted dealer exists. Here all parties run
    in one process, so we simulate the dealer directly.

    Protocol for z = x * y  (Beaver 1991)
    ----------------------------------------
    Offline (dealer):
        sample random a, b;  set c = a * b
        distribute (a_i, b_i, c_i) as additive shares to each party i

    Online (n parties, each holds x_i and y_i s.t. sum = x, y):
        e_i = x_i - a_i  -->  broadcast  -->  e = sum(e_i) = x - a
        f_i = y_i - b_i  -->  broadcast  -->  f = sum(f_i) = y - b
        z_i = c_i + f*a_i + e*b_i + [i==0]*e*f

        Proof:  sum(z_i) = ab + f*a + e*b + ef
                         = ab + (y-b)*a + (x-a)*b + (x-a)*(y-b)  =  xy  ✓

    e and f are safe to reveal because a and b are random (one-time-pad argument).
    """

    def __init__(self, n_parties: int):
        self.n = n_parties

    def generate_triple(
        self, shape: tuple
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate one (a, b, c=a*b) Beaver Triple split into n additive shares."""
        a = torch.randn(shape)
        b = torch.randn(shape)
        c = a * b
        a_s = additive_split(a, self.n)
        b_s = additive_split(b, self.n)
        c_s = additive_split(c, self.n)
        return [(a_s[i], b_s[i], c_s[i]) for i in range(self.n)]

    def mul(
        self,
        x_shares: List[torch.Tensor],
        y_shares: List[torch.Tensor],
        triples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Securely compute z = x * y (element-wise) on additive shares."""
        e_shares = [x_shares[i] - triples[i][0] for i in range(self.n)]
        f_shares = [y_shares[i] - triples[i][1] for i in range(self.n)]
        e = sum(e_shares)   # x - a  (revealed; a is random so x stays hidden)
        f = sum(f_shares)   # y - b

        result = []
        for i in range(self.n):
            a_i, b_i, c_i = triples[i]
            z_i = c_i + f * a_i + e * b_i
            if i == 0:
                z_i = z_i + e * f
            result.append(z_i)
        return result

    def poly_act(
        self,
        h_shares: List[torch.Tensor],
        triples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Securely compute PolyAct(h) = h*(h+0.5) = h^2 + 0.5h.

        Uses one Beaver Triple for h^2; 0.5h is a free scalar multiply.
        sum(result_i) = PolyAct(sum(h_i))  without any party seeing h.
        """
        h2_shares = self.mul(h_shares, h_shares, triples)
        return [h2_shares[i] + 0.5 * h_shares[i] for i in range(self.n)]
