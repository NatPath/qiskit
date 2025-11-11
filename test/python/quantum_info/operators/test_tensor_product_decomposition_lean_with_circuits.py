# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Robust tests for tensor_product_decomposition.
Covers:
- All bipartitions for n up to MAX_N (both search orders).
- All set partitions for n <= FULL_SET_PARTITIONS_UP_TO (both search orders).
- Sampled set partitions for FULL_SET_PARTITIONS_UP_TO < n <= MAX_N (small_to_big).
"""
import itertools
import unittest

from ddt import ddt, data
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary, Operator

from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators.tensor_product_decomposition import (
    tensor_product_decomposition,
    circuit_from_tensor_product,
)
from qiskit.quantum_info.operators.operator_schmidt_decomposition import (
    _permutation_matrix_from_qubit_order,
)
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

# =============================================================================
# ------------------------------ CONFIG KNOBS ---------------------------------
# =============================================================================
# Change MAX_N here; the rest adapts automatically.
# Don't set too high (>10) as tests will take a lot of time.
MAX_N = 7

# Enumerate ALL set partitions for n <= this value (kept modest for CI time).
FULL_SET_PARTITIONS_UP_TO = min(6, MAX_N)

# Number of random set partitions per n for FULL_SET_PARTITIONS_UP_TO < n <= MAX_N.
# (Adjust per CI/runtime needs; defaults are modest.)
SAMPLED_SET_PARTITIONS_PER_N = {
    7: 10,
    8: 10,  # useful if you bump MAX_N
    9: 10,
    10: 10,
}

# Search orders
SEARCH_MODES = ("small_to_big", "big_to_small")


# =============================================================================
# -------------------------------- HELPERS ------------------------------------
# =============================================================================
def kron_msb_to_lsb(*factors: np.ndarray) -> np.ndarray:
    """Kronecker in MSB->LSB order (left arg acts on MSB)."""
    out = factors[0]
    for fac in factors[1:]:
        out = np.kron(out, fac)
    return out


def blocks_cover_and_disjoint(blocks, n_qubits: int) -> bool:
    """Return True iff `blocks` form a disjoint partition of {0,...,n_qubits-1}."""
    seen = set()
    for blk in blocks:
        for qb in blk:
            if qb in seen:
                return False
            seen.add(qb)
    return seen == set(range(n_qubits))


def canonicalize_blocks(blocks):
    """Return blocks canonicalized (sort items in blocks; sort blocks by (len, tuple))."""
    return tuple(sorted((tuple(sorted(b)) for b in blocks), key=lambda t: (len(t), t)))


def build_product_from_blocks(n_qubits, blocks, seeds):
    """Build exact product U for `blocks` (LSB->MSB) in the original little-endian order."""
    assert len(seeds) == len(blocks)
    mats = [random_unitary(2 ** len(blk), seed=sd).data for blk, sd in zip(blocks, seeds)]
    # Build MSB->LSB Kronecker so that block 0 truly lands on LSB after permutation.
    u_perm = mats[-1]  # start from MSB block
    for mat in reversed(mats[:-1]):  # fold down to LSB
        u_perm = np.kron(u_perm, mat)
    new_order = tuple(q for blk in blocks for q in blk)  # LSB block first
    perm_matrix = _permutation_matrix_from_qubit_order(new_order, n_qubits)
    return perm_matrix.T @ u_perm @ perm_matrix


def enumerate_set_partitions(n_qubits):
    """Yield ALL set partitions of {0,...,n_qubits-1} (order-insensitive)."""

    def _backtrack(idx, acc_blocks):
        if idx == n_qubits:
            sorted_blocks = [tuple(sorted(b)) for b in acc_blocks]
            sorted_blocks = tuple(sorted(sorted_blocks, key=lambda t: (min(t), len(t), t)))
            yield sorted_blocks
            return
        # place idx into existing blocks
        for blk in acc_blocks:
            blk.append(idx)
            yield from _backtrack(idx + 1, acc_blocks)
            blk.pop()
        # start a new block with idx
        acc_blocks.append([idx])
        yield from _backtrack(idx + 1, acc_blocks)
        acc_blocks.pop()

    if n_qubits <= 0:
        return
    seen = set()
    for part in _backtrack(0, []):
        if part not in seen:
            seen.add(part)
            yield part


def sample_set_partitions(n_qubits, count, rng):
    """Sample `count` random set partitions of {0,...,n_qubits-1} via RGS sampling."""
    seen = set()
    attempts = 0
    max_attempts = max(100, 20 * count)
    while len(seen) < count and attempts < max_attempts:
        attempts += 1
        # random restricted-growth string (RGS)
        rgs = [0]
        max_label = 0
        for _ in range(1, n_qubits):
            lab = rng.integers(0, max_label + 2)
            rgs.append(lab)
            if lab == max_label + 1:
                max_label += 1
        # convert to blocks
        num_labels = max(rgs) + 1
        blocks = [[] for _ in range(num_labels)]
        for idx, lab in enumerate(rgs):
            blocks[lab].append(idx)
        part = tuple(
            sorted(
                (tuple(sorted(b)) for b in blocks),
                key=lambda t: (min(t), len(t), t),
            )
        )
        if part not in seen:
            seen.add(part)
    return list(seen)


# =============================================================================
# --------------------------- ASSERTION HELPERS --------------------------------
# =============================================================================
def _assert_diagnostics_consistent(testcase, res, u_op):
    """Common assertions for residual/relative_residual/is_exact consistency."""
    testcase.assertGreaterEqual(res.residual, 0.0)
    testcase.assertGreaterEqual(res.relative_residual, 0.0)
    denom = float(np.linalg.norm(u_op, ord="fro"))
    # relative_residual ≈ residual / ||U||_F  (handle zero-norm edge case)
    if denom == 0.0:
        testcase.assertAlmostEqual(res.relative_residual, 0.0, places=12)
    else:
        testcase.assertAlmostEqual(res.relative_residual, res.residual / denom, places=12)
    # is_exact matches tolerance predicate
    atol = float(ATOL_DEFAULT)
    rtol = float(RTOL_DEFAULT)
    testcase.assertEqual(res.is_exact, bool(res.residual <= atol + rtol * denom))


# =============================================================================
# -------------------------------- TEST SUITES --------------------------------
# =============================================================================
@ddt
class TestTPDAllBipartitionsExact(unittest.TestCase):
    """Exhaustive bipartition tests for n in [2..MAX_N] in both search orders."""

    @data(*range(2, MAX_N + 1))
    def test_all_bipartitions_exact_both_orders(self, n_qubits):
        """All S|Sc for this n: exactness, canonical blocks, permutation, reconstruction."""
        for k_size in range(1, n_qubits // 2 + 1):
            for subset_s in itertools.combinations(range(n_qubits), k_size):
                subset_sc = tuple(q for q in range(n_qubits) if q not in subset_s)
                # LSB->MSB blocks = (subset_s, subset_sc) for deterministic product-building
                blocks = (tuple(sorted(subset_s)), tuple(sorted(subset_sc)))
                for seed_offset in range(2):  # modest variety
                    seeds = tuple(1000 + seed_offset * 10 + i for i, _ in enumerate(blocks))
                    u_op = build_product_from_blocks(n_qubits, blocks, seeds)
                    for search in SEARCH_MODES:
                        res = tensor_product_decomposition(
                            u_op,
                            search=search,
                            return_operator=True,
                        )
                        with self.subTest(
                            n=n_qubits,
                            k=k_size,
                            s=subset_s,
                            seed_offset=seed_offset,
                            search=search,
                        ):
                            self.assertTrue(res.is_exact)
                            self.assertEqual(
                                canonicalize_blocks(res.blocks),
                                canonicalize_blocks(blocks),
                            )
                            self.assertTrue(blocks_cover_and_disjoint(res.blocks, n_qubits))
                            # Permutation & reconstruction
                            new_order = res.permutation["new_order"]
                            self.assertEqual(
                                new_order,
                                tuple(q for blk in res.blocks for q in blk),
                            )
                            perm_matrix = res.permutation["matrix"]
                            self.assertEqual(perm_matrix.dtype, bool)
                            np.testing.assert_allclose(
                                perm_matrix.T @ perm_matrix,
                                np.eye(2**n_qubits),
                                atol=1e-12,
                            )
                            np.testing.assert_allclose(res.reconstruction, u_op, atol=1e-12)
                            # Diagnostics
                            _assert_diagnostics_consistent(self, res, u_op)


@ddt
class TestTPDAllAndSampledSetPartitions(unittest.TestCase):
    """All set partitions for small n; sampled partitions for larger n (up to MAX_N)."""

    @data(*range(2, FULL_SET_PARTITIONS_UP_TO + 1))
    def test_all_set_partitions_exact_both_orders(self, n_qubits):
        """Exactness & equality to expected blocks for every set partition (both orders)."""
        seed_base = 3000 + n_qubits * 100
        for idx, blocks in enumerate(enumerate_set_partitions(n_qubits)):
            seeds = tuple(seed_base + idx * 10 + i for i, _ in enumerate(blocks))
            u_op = build_product_from_blocks(n_qubits, blocks, seeds)
            for search in SEARCH_MODES:
                res = tensor_product_decomposition(u_op, search=search, return_operator=True)
                with self.subTest(n=n_qubits, idx=idx, search=search):
                    self.assertTrue(res.is_exact)
                    self.assertEqual(
                        canonicalize_blocks(res.blocks),
                        canonicalize_blocks(blocks),
                    )
                    self.assertTrue(blocks_cover_and_disjoint(res.blocks, n_qubits))
                    np.testing.assert_allclose(res.reconstruction, u_op, atol=1e-12)
                    # Diagnostics
                    _assert_diagnostics_consistent(self, res, u_op)

    @data(*range(FULL_SET_PARTITIONS_UP_TO + 1, MAX_N + 1))
    def test_sampled_set_partitions_exact_small_to_big(self, n_qubits):
        """Exactness & equality for sampled partitions in small_to_big (runtime-friendly)."""
        # Might be empty if MAX_N == FULL_SET_PARTITIONS_UP_TO
        samples = SAMPLED_SET_PARTITIONS_PER_N.get(n_qubits, 0)
        if samples <= 0:
            self.skipTest(f"No samples configured for n={n_qubits}")
        rng = np.random.default_rng(4000 + n_qubits)
        parts = sample_set_partitions(n_qubits, samples, rng)
        for idx, blocks in enumerate(parts):
            seeds = tuple(5000 + n_qubits * 100 + idx * 10 + i for i, _ in enumerate(blocks))
            u_op = build_product_from_blocks(n_qubits, blocks, seeds)
            res = tensor_product_decomposition(u_op, search="small_to_big", return_operator=True)
            with self.subTest(n=n_qubits, idx=idx):
                self.assertTrue(res.is_exact)
                self.assertEqual(
                    canonicalize_blocks(res.blocks),
                    canonicalize_blocks(blocks),
                )
                self.assertTrue(blocks_cover_and_disjoint(res.blocks, n_qubits))
                np.testing.assert_allclose(res.reconstruction, u_op, atol=1e-12)
                # Diagnostics
                _assert_diagnostics_consistent(self, res, u_op)


class TestTensorProductCircuit(unittest.TestCase):
    def test_parallel_blocks_exact(self):
        """If U is an exact tensor product across blocks, the circuit should match U exactly."""
        n = 5
        blocks = ((0, 2), (4,), (1, 3))
        seeds = (10, 11, 12)
        u = build_product_from_blocks(n, blocks, seeds)

        pack = circuit_from_tensor_product(u, search="small_to_big")
        qc = pack.circuit
        self.assertTrue(pack.had_partitions)

        # Compare circuit matrix directly
        u_circ = Operator(qc).data
        np.testing.assert_allclose(u_circ, u, atol=1e-12)

        # Check decomposition metadata
        decomp = pack.decomposition
        self.assertTrue(decomp.is_exact)
        canon = lambda blks: tuple(
            sorted((tuple(sorted(b)) for b in blks), key=lambda t: (len(t), t))
        )
        self.assertEqual(canon(decomp.blocks), canon(blocks))

    def test_no_partition_returns_single_gate_and_flag(self):
        """For a generic random U, expect a single block and had_partitions=False."""
        n = 3
        u = random_unitary(2**n, seed=999).data
        pack = circuit_from_tensor_product(u, search="small_to_big")
        self.assertFalse(pack.had_partitions)

        u_circ = Operator(pack.circuit).data
        np.testing.assert_allclose(u_circ, u, atol=1e-12)
        self.assertTrue(pack.decomposition.is_exact)


if __name__ == "__main__":
    unittest.main()
