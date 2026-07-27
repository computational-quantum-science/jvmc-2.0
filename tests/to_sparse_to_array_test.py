import jax.numpy as jnp

from itertools import product

import jVMC_exp.operator.discrete as op
from jVMC_exp.vqs import NQS


class FakeState:
    sampleShape = (3,)
    to_array = NQS.to_array

    def __init__(self, return_log=True):
        self.return_log = return_log

    def __call__(self, basis):
        basis = jnp.asarray(basis)

        log_values = (
            0.15 * basis[:, 0]
            - 0.07 * basis[:, 1]
            + 0.11 * basis[:, 2]
            + 0.2j * basis[:, 1]
        )

        if self.return_log:
            return log_values

        return jnp.exp(log_values)


def test_to_sparse_and_to_array():
    basis = jnp.asarray(
        list(
            product(
                (0, 1),
                repeat=3,
            )
        ),
        dtype=jnp.int32,
    )

    H = (
        0.1 * op.SigmaX(0)
        + 0.2 * op.SigmaY(1)
        + 1.3
        * op.SigmaZ(0)
        * op.SigmaZ(2)
        - 0.4
        * op.SigmaX(1)
        * op.SigmaX(2)
    )

    H_sparse = H.to_sparse(
        basis
    )

    identity = jnp.eye(
        2,
        dtype=jnp.complex128,
    )

    sigma_x = jnp.asarray(
        [[0, 1], [1, 0]],
        dtype=jnp.complex128,
    )

    sigma_y = jnp.asarray(
        [[0, -1j], [1j, 0]],
        dtype=jnp.complex128,
    )

    sigma_z = jnp.asarray(
        [[1, 0], [0, -1]],
        dtype=jnp.complex128,
    )

    def kron3(A, B, C):
        return jnp.kron(
            jnp.kron(A, B),
            C,
        )

    expected_matrix = (
        0.1
        * kron3(
            sigma_x,
            identity,
            identity,
        )
        + 0.2
        * kron3(
            identity,
            sigma_y,
            identity,
        )
        + 1.3
        * kron3(
            sigma_z,
            identity,
            sigma_z,
        )
        - 0.4
        * kron3(
            identity,
            sigma_x,
            sigma_x,
        )
    )

    assert jnp.allclose(
        jnp.asarray(
            H_sparse.toarray()
        ),
        expected_matrix,
        atol=1.0e-6,
    )

    psi_log = FakeState(
        return_log=True
    )

    statevector_log = psi_log.to_array(
        basis,
        log=True,
    )

    psi_direct = FakeState(
        return_log=False
    )

    statevector_direct = psi_direct.to_array(
        basis,
        log=False,
    )

    assert jnp.allclose(
        statevector_log,
        statevector_direct,
        atol=1.0e-6,
    )

    assert jnp.allclose(
        jnp.linalg.norm(
            statevector_log
        ),
        1.0,
        atol=1.0e-6,
    )

    energy_sparse = jnp.vdot(
        statevector_log,
        jnp.asarray(
            H_sparse @ statevector_log
        ),
    )

    energy_expected = jnp.vdot(
        statevector_log,
        expected_matrix @ statevector_log,
    )

    assert jnp.allclose(
        energy_sparse,
        energy_expected,
        atol=1.0e-6,
    )