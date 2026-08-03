import sys
sys.path.append(sys.path[0] + "/../..")

import argparse
import csv
import resource
import time

import jax
import jVMC_exp
import jVMC_exp.nets as nets
import jVMC_exp.operator.discrete as op
import jVMC_exp.propose as propose
import jVMC_exp.sampler as sampler
from jVMC_exp.vqs import NQS

def build_hamiltonian(L, J=-1.0, hx=-0.7):
    H = 0
    for i in range(L):
        H += J * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) + hx * op.SigmaX(i)
    return H

def device_memory():
    """
    (bytes_in_use, peak_bytes_in_use) for the default JAX device, or
    (None, None) if the backend doesn't report memory stats
    """
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return None, None
    return stats.get("bytes_in_use"), stats.get("peak_bytes_in_use")

def host_max_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024

def build_sampler(psi, args):
    if args.sampler == "exact":
        return sampler.ExactSampler(psi)

    proposer = propose.SpinFlip()
    return sampler.MCSampler(psi, proposer, args.seed, args.num_chains, args.num_samples, mu=1)

def run(args):
    batched = args.jacobian == "batched"

    rbm = nets.CpxRBM(numHidden=args.num_hidden, bias=False)
    psi = NQS(rbm, args.L, args.batch_size, seed=args.seed)
    mc_sampler = build_sampler(psi, args)

    hamiltonian = build_hamiltonian(args.L)
    loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=batched)
    solver = jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-8)
    stepper = jVMC_exp.stepper.Euler(timeStep=args.lr)
    opt = jVMC_exp.optimizer.SR(mc_sampler, psi, diagonalShift=args.diag_shift, solver=solver)

    n_params = psi.numParameters
    n_samples = mc_sampler.numSamples
    dense_jacobian_gib = n_samples * n_params * 16 / 2 ** 30  # complex128
    print(
        f"jacobian={args.jacobian} "
        f"n_params={n_params}  n_samples={n_samples}  batch_size={args.batch_size}  "
        f"dense_jacobian_size~{dense_jacobian_gib:.4f} GiB",
        flush=True,
    )

    rows = []
    for step in range(args.num_steps):
        t0 = time.perf_counter()
        new_parameters, _ = opt.step(0, stepper, loss_function)
        jax.block_until_ready(new_parameters)
        t1 = time.perf_counter()

        opt.sampler._samples, opt.sampler._logPsi, opt.sampler._weights = opt._sampler_out
        opt.psi.parameters = new_parameters

        bytes_in_use, peak_bytes_in_use = device_memory()
        energy = float(jax.numpy.real(opt.o_loc.mean).squeeze())

        row = dict(
            step=step,
            time_s=t1 - t0,
            device_bytes_in_use=bytes_in_use,
            device_peak_bytes_in_use=peak_bytes_in_use,
            host_max_rss_bytes=host_max_rss_bytes(),
            energy=energy,
        )
        rows.append(row)
        print(
            f"step {step:3d}  time={row['time_s']:.4f}s  "
            f"device_bytes_in_use={bytes_in_use}  "
            f"device_peak_bytes_in_use={peak_bytes_in_use}  "
            f"host_max_rss={row['host_max_rss_bytes']}  "
            f"E={energy:.6f}",
            flush=True,
        )

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out}")

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("jacobian", choices=["dense", "batched"])
    parser.add_argument("--sampler", choices=["exact", "mc"], default="mc")
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--num-hidden", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=2**12)
    parser.add_argument("--num-chains", type=int, default=2**10)
    parser.add_argument("--batch-size", type=int, default=2**12)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diag-shift", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()