# Singular Value Decomposition (SVD) — From Scratch

> Implementing the math behind machine learning by hand instead of calling library functions — because you don't really understand a tool until you've built it once.

This repo implements the SVD of a 2×2 matrix from first principles (no `numpy.linalg.svd`), as part of relearning ML from its mathematical foundations. What follows is the concept, the implementation, and the bug that taught me the most.

---

## The idea in one line

Every matrix, no matter how messy, does only three things: **rotate → stretch → rotate.**

That's SVD: any matrix `A` factors as

```
A = U Σ Vᵀ
```

- `Vᵀ` rotates the input,
- `Σ` (diagonal, non-negative) stretches along each axis,
- `U` rotates again.

The singular values on `Σ` tell you *how much* stretch happens in each principal direction — i.e. where the matrix's "energy" lives.

---

## Prerequisite: spectral decomposition

Before SVD, the simpler case. Any **symmetric** matrix decomposes as

```
A = Q Λ Qᵀ
```

where `Q` holds the eigenvectors (orthonormal) and `Λ` is the diagonal matrix of eigenvalues.

- **Orthogonal matrix:** one where `Q Qᵀ = I` (its transpose is its inverse).
- **Diagonal matrix:** non-zero entries only on the diagonal.

The catch: this only works for **symmetric** matrices — which forces them to be square. Most real data matrices aren't symmetric, and many aren't even square. That's the gap SVD fills.

---

## SVD: spectral decomposition for *any* matrix

SVD extends the idea to **any** `m × n` matrix by leaning on two symmetric matrices you *can* always build from `A`:

| Component | What it is |
|-----------|------------|
| `V` (right singular vectors) | eigenvectors of `AᵀA` |
| `U` (left singular vectors)  | eigenvectors of `AAᵀ` |
| `Σ` (singular values)        | `√(eigenvalues of AᵀA)`, sorted descending |

Because `AᵀA` and `AAᵀ` are always symmetric, spectral decomposition applies to them even when `A` itself is rectangular. That's the whole trick.

---

## The implementation

The clean pipeline:

1. Form `M = AᵀA` (symmetric).
2. Eigendecompose it with **`np.linalg.eigh`** → eigenvectors give `V`, eigenvalues give `σ²`.
3. Sort descending and take `√` of the eigenvalues to get the singular values.
4. **Derive `U` from `V`** (see below) instead of decomposing `AAᵀ` separately.
5. Return `(U, S, Vᵀ)`.

The key derivation. Starting from `A = U Σ Vᵀ` and right-multiplying by `V`:

```
A V = U Σ Vᵀ V = U Σ        (since V is orthogonal, Vᵀ V = I)
```

So each column of `A V` is `σᵢ · uᵢ`, which means:

```python
U = (A @ V) / S        # divide each column by its singular value
```

---

## The bug that taught me the most

My first version computed `U` from `AAᵀ` and `V` from `AAᵀ`'s counterpart `AᵀA` in **two independent eigendecompositions**. It passed the symmetric test case — and silently produced a *wrong* result on others, where `A ≠ U Σ Vᵀ`.

**Why:** an eigenvector is only defined up to a sign — flip it by −1 and it's still a valid eigenvector. Two independent decompositions choose those signs independently, so my `U` and `V` columns didn't line up, and the reconstruction broke. Squaring `A` into `AAᵀ` / `AᵀA` throws away the very sign information that ties `U` and `V` together.

**The fix:** never compute them independently. Get `V`, then *derive* `U` from `A V / σ` so the relationship `A V = U Σ` is enforced by construction and the signs stay consistent.

**The lesson (the part that generalizes):** when two quantities are mathematically linked, **derive one from the other** rather than computing both separately — it's the only way to guarantee they stay consistent.

One more: use **`eigh`**, not `eig`. `AᵀA` is symmetric, and `eigh` returns real eigenvalues, guaranteed-orthonormal eigenvectors, and sorted output. `eig` (the general solver) drags in complex numbers and doesn't guarantee orthogonality — wrong tool for a symmetric matrix.

---

## Why SVD actually matters

This isn't an academic exercise — the same low-rank structure SVD exposes runs through modern ML:

- **PCA** is literally SVD of your centered data (and the numerically stable way to do it).
- **LoRA**, the dominant LLM fine-tuning method, rests on the fact that weight *updates* are approximately low-rank — the exact factorization SVD produces.
- **Low-rank attention** (e.g. Linformer) approximates the attention matrix with its top singular directions to run in linear time.
- **Pseudoinverse / least squares** (`np.linalg.lstsq`) is built on SVD.
- **Condition number & numerical rank** are read directly off the singular values.

The through-line: **low-rank structure** — the top singular values hold the signal; the rest is noise.

---

## The meta-lesson

Implement from scratch to *understand* a concept; call `np.linalg.svd` to *use* it. No one hand-rolls SVD in production — LAPACK does it faster and more stably than anything I'd write. But building it once is what makes PCA, LoRA, and every low-rank method click instead of staying abstract.

---

## Run it

```bash
python svd_2x2.py
```

The function returns `(U, S, Vt)` and satisfies `A ≈ U @ diag(S) @ Vt`, verified against symmetric, non-symmetric, and sign-sensitive matrices with `np.allclose`.
