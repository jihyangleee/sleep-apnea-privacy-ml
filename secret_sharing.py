import torch
from typing import List, Tuple


def additive_split(x: torch.Tensor, n: int = 3) -> List[torch.Tensor]:
    """Additive secret share: returns n shares that sum to x."""
    noise = [torch.randn_like(x) for _ in range(n - 1)]
    last  = x - sum(noise)
    return noise + [last]


def additive_split_symmetric(x: torch.Tensor, n: int = 3) -> List[torch.Tensor]:
    """Additive secret share where EVERY share carries an equal, differentiable
    slice of x — unlike additive_split(), where only the last share is a real
    function of x and the rest are pure noise (fine when only one recipient
    ever needs to route gradient back through its share, but wrong when a
    mesh of n parties each independently receive one share and must all be
    able to route d(loss)/d(their share) back to x's owner).

    share_i = x/n + r_i,  for i < n-1  (r_i random, mean-zero across the set)
    share_{n-1} = x/n - sum(r_i)

    Every share has d(share_i)/dx = 1/n, so gradients returned by ANY subset
    of recipients accumulate correctly: sum_i d(share_i)/dx = 1.
    """
    base = x / n
    r = [torch.randn_like(x) for _ in range(n - 1)]
    last = base - sum(r)
    return [base + ri for ri in r] + [last]


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

    def sigmoid_approx(
        self,
        x_shares: List[torch.Tensor],
        triples1: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        triples2: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Securely compute σ(x) ≈ 0.5 + 0.197x - 0.004x³.

        Two Beaver Triples needed because x³ = x² * x (two multiplications):
          triples1 → x² = x * x
          triples2 → x³ = x² * x

        Only party 0 adds the constant 0.5 (same pattern as bias in logit_share).
        Gradient flows through all tensor ops for backprop.

        sum(result_i) = σ(sum(x_i))  without any party reconstructing x.
        """
        x2_shares = self.mul(x_shares, x_shares, triples1)
        x3_shares = self.mul(x2_shares, x_shares, triples2)
        result = [0.197 * x_shares[i] - 0.004 * x3_shares[i] for i in range(self.n)]
        result[0] = result[0] + 0.5
        return result
