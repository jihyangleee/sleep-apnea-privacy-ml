"""SPU 3-party simulation smoke test for top_layer_jax.py.

Runs the same forward/backward math as top_layer_jax.py's __main__ block,
but this time each hospital's logit_share lives on a separate simulated
SecretFlow party, and the sum + sigmoid_approx + loss actually execute as
an MPC protocol on the SPU device -- not plain JAX on one process.

Requires the WSL venv (venv_wsl) with secretflow installed.
"""

import numpy as np
import jax.numpy as jnp
import secretflow as sf
from secretflow.device import SPU, SPUObject
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy

from top_layer_jax import top_layer_loss_and_grad

try:
    sf.shutdown()
except Exception:
    pass  # no prior session to tear down on a fresh process
sf.init(parties=["hospital0", "hospital1", "hospital2"], address="local")

cluster_def = sf.utils.testing.cluster_def(
    parties=["hospital0", "hospital1", "hospital2"],
)
print("=== SPU cluster_def (this is the security model actually in use) ===")
import json
print(json.dumps(cluster_def, indent=2, default=str))

# sf.SPU()'s constructor does json.dumps(cluster_def['runtime_config'])
# internally to feed a protobuf parser, but sf.utils.testing.cluster_def()
# returns raw spu.ProtocolKind/spu.FieldType enum objects there -- a
# version mismatch between this secretflow release and the auto-resolved
# spu binary, since plain json.dumps can't serialize those enums. Protobuf
# JSON accepts the enum's string name, so swap enums for their .name here
# rather than chasing an exact compatible spu pin.
cluster_def["runtime_config"] = {
    k: (v.name if hasattr(v, "name") else v)
    for k, v in cluster_def["runtime_config"].items()
}

spu = sf.SPU(cluster_def)

hospital0, hospital1, hospital2 = (
    sf.PYU("hospital0"), sf.PYU("hospital1"), sf.PYU("hospital2")
)

rng = np.random.default_rng(0)
batch = 8
local_logit_shares_np = [
    rng.normal(size=(batch, 1)).astype(np.float32) for _ in range(3)
]
y_np = (rng.random(size=(batch, 1)) > 0.5).astype(np.float32)
y_shares_np = [y_np / 3 for _ in range(3)]

# Each hospital's PYU produces its own private plaintext value; .to(spu)
# is the point where it becomes an SPU secret-shared object.
logit_share_pyu = [
    pyu(lambda x: jnp.array(x))(local_logit_shares_np[i])
    for i, pyu in enumerate([hospital0, hospital1, hospital2])
]
y_share_pyu = [
    pyu(lambda x: jnp.array(x))(y_shares_np[i])
    for i, pyu in enumerate([hospital0, hospital1, hospital2])
]

logit_shares_spu = [x.to(spu) for x in logit_share_pyu]
y_shares_spu     = [x.to(spu) for x in y_share_pyu]

loss_spu, grads_spu = spu(
    top_layer_loss_and_grad,
    num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
    user_specified_num_returns=2,
)(logit_shares_spu, y_shares_spu)

loss_revealed  = sf.reveal(loss_spu)
# grads_spu is a single SPUObject wrapping the whole [grad0, grad1, grad2]
# pytree -- reveal it as one unit, sf.reveal reconstructs the list of
# plaintext arrays, then index into that.
grads_revealed = sf.reveal(grads_spu)

print("\n=== SPU secure computation result ===")
print(f"  loss  : {float(loss_revealed):.6f}")
for i, g in enumerate(grads_revealed):
    print(f"  hospital {i} grad (first 3): {np.array(g).ravel()[:3]}")

sf.shutdown()
