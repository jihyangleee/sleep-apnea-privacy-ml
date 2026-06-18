"""
CKKS context shared by the patient device and all hospitals.

The patient (Galaxy Watch / user's device) holds the CKKS secret key.
Hospitals hold only the public key and model weights.
"""

import tenseal as ts


def build_he_context() -> ts.Context:
    # Level budget per hospital:
    #   mm_(W_sub)  -1  mm_(W_embed)  -1  mm_(W_top)  -1  = 3 linear levels
    #   CubicAct: z*z -1, then enc_z (level L) * enc_z2 (level L-1)
    #             → TenSEAL mod-switches enc_z down first (-1 extra) + multiply (-1) = -2 extra
    #   Total: 3 + 2 = 5 levels needed; add 2 extra primes as headroom → 7 usable 40-bit levels
    # [60,40,40,40,40,40,40,40,60] max bits = 400 < CoeffModulus::MaxBitCount(16384)=438 ✓
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60],
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    return ctx
