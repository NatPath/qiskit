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
"""
Tensor-product block decomposition for n-qubit operators.

Given an operator U acting on n qubits (unitary or not), this module provides a function that
finds the *most granular* tensor-product factorization over disjoint qubit sets **when it exists**.
If a subset has no exact bipartition (rank-1 operator Schmidt split), it becomes a leaf block.

It relies on the Operator Schmidt Decomposition (OSD) utility available in
``qiskit.quantum_info``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence
import itertools
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

from qiskit.quantum_info.operators.operator_schmidt_decomposition import operator_schmidt_decomposition, _permutation_matrix_from_qubit_order


# -------------------------------------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------------------------------------
@dataclass
class TensorProductDecomposition:
    """
    Structured result returned by :func:`tensor_product_decomposition`.

    Attributes
    ----------
    blocks:
        A partition of the qubit indices into non-overlapping blocks, listed in the *same order*
        used to build the final permutation (see below).

    factors:
        Local operators (np.ndarray) for each block, aligned with ``blocks``.
        With the ordering guarantees in this module, you can reconstruct the *permuted* operator as
        ``kron(factors[0], factors[1], ..., factors[m-1])`` (i.e., LSB block first in Kronecker
        order).

    operator_factors:
        Same as ``factors`` but wrapped as :class:`~qiskit.quantum_info.Operator` when
        ``return_operator=True``.

    is_exact:
        ``True`` when the factorization is exact within tolerances.

    residual, relative_residual:
        Frobenius norm of ``U - reconstruction`` and its normalized variant.

    permutation:
        Dict with:
        * ``new_order``: tuple[int] – qubit order used by the final permutation (concatenation
          of ``blocks`` in the returned order; LSB block comes first).
        * ``matrix``: np.ndarray (dtype ``bool``) – permutation matrix ``P`` such that
          ``U = P^T U_perm P`` and ``U_perm = kron(factors[0],...,factors[m-1])``.

    reconstruction:
        The reconstructed operator (np.ndarray) in the *original* qubit order, present only when
        ``return_operator=True``.
    """

    blocks: tuple[tuple[int, ...], ...]
    factors: tuple[np.ndarray, ...]
    operator_factors: tuple[Operator, ...] | None
    is_exact: bool
    residual: float
    relative_residual: float
    permutation: dict[str, Any]
    reconstruction: np.ndarray | None = None


# -------------------------------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------------------------------
def _closest_unitary(x: np.ndarray) -> np.ndarray:
    """Polar decomposition: return the unitary closest to x in Frobenius norm."""
    u_mat, _, vh_mat = np.linalg.svd(x, full_matrices=False)
    return u_mat @ vh_mat


# def _permutation_matrix_from_qubit_order(new_order: Sequence[int], n: int) -> np.ndarray:
#     """
#     Return the ``(2**n) x (2**n)`` permutation matrix ``P`` that reorders **little‑endian** qubits.

#     Little‑endian convention: qubit 0 is the **least significant bit** (LSB).
#     Mapping (bits → indices):
#     * ``new_order[k]`` gives which **original** qubit becomes bit‑position ``k`` in the **new**
#       representation (with ``k=0`` the LSB).
#     * For a computational basis state with original bitstring
#       ``b = (b_{n-1} ... b_1 b_0)`` we form the new bitstring ``b'`` by
#       ``b'_k = b_{ new_order[k] }``.
#     * Index mapping: ``i' = sum_k b'_k 2^k``.

#     Action:
#     * States: ``|psi_new> = P |psi_old>``.
#     * Operators: ``U_new = P U_old P^T`` (``P`` is real; ``P^T = P.conj().T``).

#     Raises:
#         QiskitError: If inputs are invalid.
#     """
#     if not isinstance(n, int) or n < 0:
#         raise QiskitError("`n` must be a non‑negative integer.")
#     if len(new_order) != n:
#         raise QiskitError(f"`new_order` must have length n={n}.")
#     if set(new_order) != set(range(n)):
#         raise QiskitError("`new_order` must be a permutation of range(n).")

#     dim = 2**n
#     indices = np.arange(dim, dtype=np.int64)  # original indices i
#     # Extract original bits b_q (q=0 is LSB) for each index.
#     bits = (indices[:, None] >> np.arange(n, dtype=np.int64)) & 1  # (dim, n)
#     # Reorder bits so that new bit‑position k gets original bit from new_order[k].
#     reordered_bits = bits[:, new_order]  # (dim, n)
#     # Convert reordered bits to new indices i'
#     new_indices = np.sum(reordered_bits << np.arange(n, dtype=np.int64), axis=1)
#     # Build permutation matrix with columns permuted by new_indices
#     return np.eye(dim, dtype=bool)[:, new_indices]


def _pick_bipartitions(
    m: int,
    order: Literal["small_to_big", "big_to_small"],
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Yield nontrivial bipartitions (s, sc) over local indices 0..m-1, up to symmetry.
    The order of S depends on ``order``:
    - "small_to_big": 1, 2, ..., floor(m/2)
    - "big_to_small": floor(m/2), ..., 2, 1
    """
    if m <= 1:
        return
    sizes = range(1, m // 2 + 1) if order == "small_to_big" else range(m // 2, 0, -1)
    local_qubits = tuple(range(m))
    for k_size in sizes:
        for subset in itertools.combinations(local_qubits, k_size):
            sc = tuple(q for q in local_qubits if q not in subset)
            yield tuple(sorted(subset)), tuple(sorted(sc))


# -------------------------------------------------------------------------------------------------
# Core recursion
# -------------------------------------------------------------------------------------------------
def _factorize_recursive(
    op_local: np.ndarray,
    qubits_local: tuple[int, ...],
    *,
    atol: float,
    search: Literal["small_to_big", "big_to_small"],
) -> tuple[list[tuple[int, ...]], list[np.ndarray]]:
    """
    Recursively decompose `op_local` acting on `qubits_local` into product blocks.

    Returns
    -------
    blocks, factors
      - `blocks`: list of global-qubit tuples (kept in recursion order)
      - `factors`: list of ndarray factors, aligned with `blocks`
    """
    m_size = len(qubits_local)
    if m_size == 1:
        return [qubits_local], [op_local]

    # 1) Try exact rank-1 splits first (via OSD k=1); pass LOCAL indices to OSD.
    for s_local, _ in _pick_bipartitions(m_size, search):
        res_osd = operator_schmidt_decomposition(
            op_local, qargs=s_local, k=1, return_reconstruction=False
        )
        svals = res_osd["singular_values"]
        tail = svals[1:]
        fro_err = float(np.sqrt(np.sum(tail**2))) if tail.size else 0.0
        if fro_err <= atol:
            # Accept exact split. OSD returns A on S, B on Sc in the [Sc, S] permuted basis.
            a_factor = res_osd["A_factors"][0]
            b_factor = res_osd["B_factors"][0]

            # Map LOCAL → GLOBAL qubit indices for the two sides.
            s_global = tuple(qubits_local[i] for i in s_local)
            sc_global = tuple(q for q in qubits_local if q not in s_global)

            # Recurse on each side. Convention: blocks = [S, Sc], factors = [B, A]
            blocks_a, ops_a = _factorize_recursive(
                a_factor, tuple(sorted(s_global)), atol=atol, search=search
            )
            blocks_b, ops_b = _factorize_recursive(
                b_factor, tuple(sorted(sc_global)), atol=atol, search=search
            )
            return blocks_a + blocks_b, ops_b + ops_a

    # 2) No exact split under this subset → leaf block (still exact overall).
    return [qubits_local], [op_local]


# -------------------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------------------
def tensor_product_decomposition(
    op: np.ndarray,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    search: Literal["small_to_big", "big_to_small"] = "small_to_big",
    return_operator: bool = False,
) -> TensorProductDecomposition:
    r"""
    Find the most granular tensor-product *block* decomposition of an operator over disjoint
    qubit sets.

    **Semantics and ordering.**
    We return disjoint blocks ``(S_0, S_1, ..., S_{m-1})`` and corresponding local factors
    ``(U_{S_0}, U_{S_1}, ..., U_{S_{m-1}})`` such that:
    - The concatenation ``new_order = S_0 + S_1 + ... + S_{m-1}`` (LSB block first) defines a
      permutation matrix ``P`` with ``U = P^T U_perm P``.
    - The *permuted* operator factors as a plain Kronecker product in **LSB→MSB order**:
      ``U_perm = kron(U_{S_0}, U_{S_1}, ..., U_{S_{m-1}})``.

    Internally rely on :func:`operator_schmidt_decomposition` which computes the OSD in a
    permuted basis ``[Sc, S]`` with the B-factor (``Sc``) occupying the LSB block and the A-factor
    (``S``) occupying the MSB block. We keep the recursion output ordered so that reconstruction
    needs only a *single* final permutation for ``new_order`` (no extra swaps).

    Parameters
    ----------
    op : np.ndarray
        Square matrix of shape ``(2**n, 2**n)`` in the standard little-endian basis
        (qubit 0 is LSB).
    atol : float or None
        Absolute tolerance used to accept rank‑1 (exact) operator‑Schmidt splits and in the final
        exactness check for the reconstruction. If ``None``, defaults to
        :data:`qiskit.quantum_info.operators.predicates.ATOL_DEFAULT`.
    rtol : float or None
        Relative tolerance used together with ``atol`` to accept exactness (i.e.,
        a check of the form ``residual <= atol + rtol * ||U||_F``). If ``None``, defaults to
        :data:`qiskit.quantum_info.operators.predicates.RTOL_DEFAULT`.
    search : {"small_to_big","big_to_small"}
        Controls the order of subset sizes ``S`` to try at each recursion node for exact splits.
        "small_to_big": 1, 2, ..., floor(m/2). "big_to_small": floor(m/2), ..., 2, 1.
    return_operator : bool
        If True, also supply ``operator_factors`` (as :class:`Operator`) and the full
        ``reconstruction`` matrix in the original qubit order.

    Returns
    -------
    TensorProductDecomposition
        See :class:`TensorProductDecomposition`. ``blocks`` and ``factors`` are aligned as
        described above.
    """
    if not isinstance(op, np.ndarray) or op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise QiskitError("`op` must be a square np.ndarray.")
    n_float = np.log2(op.shape[0])
    if not np.isclose(n_float, int(n_float)):
        raise QiskitError("`op` dimension must be a power of 2.")
    n_qubits = int(round(n_float))
    if n_qubits == 0:
        raise QiskitError(
            "0-qubit (1x1) operators are not supported by tensor_product_decomposition."
        )

    # Default tolerances from predicates
    if atol is None:
        atol = float(ATOL_DEFAULT)
    if rtol is None:
        rtol = float(RTOL_DEFAULT)

    qubits = tuple(range(n_qubits))
    blocks, local_ops = _factorize_recursive(op, qubits, atol=atol, search=search)

    # Keep the recursion ordering. Build U_perm as kron(factors[0],...,factors[m-1]) (LSB→MSB).
    factors_ordered = tuple(local_ops)
    blocks_ordered = tuple(blocks)

    if len(factors_ordered) == 0:
        u_perm = np.eye(op.shape[0], dtype=complex)
    else:
        u_perm = factors_ordered[0]
        for mat in factors_ordered[1:]:
            u_perm = np.kron(u_perm, mat)

    # Build final permutation new_order by concatenating the blocks (LSB block first).
    new_order = tuple(q for blk in blocks_ordered for q in blk)
    perm_matrix = _permutation_matrix_from_qubit_order(new_order, n_qubits)

    reconstruction = perm_matrix.T @ u_perm @ perm_matrix
    diff = op - reconstruction
    residual = float(np.linalg.norm(diff, ord="fro"))
    denom = float(np.linalg.norm(op, ord="fro"))
    relative_residual = (residual / denom) if denom != 0.0 else 0.0

    operator_factors = tuple(Operator(m) for m in factors_ordered) if return_operator else None

    return TensorProductDecomposition(
        blocks=blocks_ordered,
        factors=factors_ordered,
        operator_factors=operator_factors,
        is_exact=bool(residual <= atol + rtol * denom),
        residual=residual,
        relative_residual=relative_residual,
        permutation={"new_order": new_order, "matrix": perm_matrix},
        reconstruction=reconstruction if return_operator else None,
    )

# -------------------------------------------------------------------------------------------------
# Circuit construction from exact tensor-product decomposition
# -------------------------------------------------------------------------------------------------
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


@dataclass
class TensorProductCircuit:
    """
    Container for the circuit synthesized from the tensor-product decomposition.

    Attributes
    ----------
    circuit : QuantumCircuit
        Circuit that applies each local unitary on its corresponding qubit block.

    had_partitions : bool
        True iff the decomposition produced more than one block (i.e., a non-trivial partition).

    decomposition : TensorProductDecomposition
        The raw decomposition details returned by `tensor_product_decomposition`.
    """
    circuit: QuantumCircuit
    had_partitions: bool
    decomposition: TensorProductDecomposition


def circuit_from_tensor_product(
    op: np.ndarray | Operator,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    search: Literal["small_to_big", "big_to_small"] = "small_to_big",
) -> TensorProductCircuit:
    """
    Build a QuantumCircuit that realizes the operator as parallel local unitaries on the
    most granular tensor-product blocks.

    Parameters
    ----------
    op : np.ndarray | Operator
        Square matrix of shape (2**n, 2**n) or an `Operator` wrapping such a matrix.
    atol, rtol, search
        Forwarded to `tensor_product_decomposition`.

    Returns
    -------
    TensorProductCircuit
        - `circuit`: applies each local factor on its qubit block.
        - `had_partitions`: True iff more than one block was found.
        - `decomposition`: the `TensorProductDecomposition` structure for additional inspection.

    Notes
    -----
    The returned circuit uses the original qubit indexing. Each block in `decomposition.blocks`
    is a tuple of qubit indices (ascending). Each factor in `decomposition.factors` is applied
    to its block in that order. No additional permutations are required, because the recursion
    already aligned local factors with the original basis during reconstruction.
    """
    # Convert Operator → ndarray when needed
    if isinstance(op, Operator):
        op_mat = op.data
    else:
        op_mat = op

    decomp = tensor_product_decomposition(
        op_mat, atol=atol, rtol=rtol, search=search, return_operator=False
    )

    # Determine size and build a circuit over n original qubits
    n_qubits = int(round(np.log2(op_mat.shape[0])))
    qc = QuantumCircuit(n_qubits, name="tensor_product_circuit")

    # For each (block, factor), append a local unitary on those qubits in ascending order.
    for block, factor in zip(decomp.blocks, decomp.factors):
        # Defensive checks
        if factor.shape != (2 ** len(block), 2 ** len(block)):
            raise QiskitError(
                f"Inconsistent local factor shape {factor.shape} for block {block}"
            )
        gate = UnitaryGate(factor)
        qc.append(gate, qargs=list(block))

    had_partitions = len(decomp.blocks) > 1
    return TensorProductCircuit(circuit=qc, had_partitions=had_partitions, decomposition=decomp)
