# QGSR — Query-Guided Subspace Reconstruction

A PyTorch research prototype for multi-sensor fusion using an **internal learned query** and **trainable subspace reconstruction**. Includes a compact fusion module and a report comparing QGSR with ten competitors across six datasets.

[Read the full report](QGSR_Raw_Start_Guide.pdf) · [Python module](qgsr.py) · [AGPL-3.0-only licence](LICENSE)

## Quick start

Requires PyTorch; the module was checked with PyTorch 2.9.1 on CPU.

```bash
python -m pip install torch
```

Save or clone this repository, then run from its directory:

```python
import torch
from qgsr import QGSR

fusion = QGSR(sources=3)
z = torch.randn(8, 3, 32)  # batch, sources, encoded features
y = fusion(z)             # shape: (8, 32)

valid = torch.ones(8, 3, dtype=torch.bool)
valid[:, 1] = False       # mark the second source as missing
y = fusion(z, valid)
```

Each source must first be encoded into a 32-feature vector. Source encoders and the task classifier are separate from this fusion module. The optional `valid` argument is a boolean presence mask, not an external query. Present-source inputs must be finite; an all-missing input returns zero.

## Original mathematics

For encoded source vectors `z_i`, QGSR computes:

$$
w_i=r_i e^{-\beta\lVert K(z_i-q)\rVert^2},
\qquad
s=\frac{\sum_i w_i z_i}{\epsilon+\sum_i w_i}
$$

$$
A_i=w_i U_iU_i^\top
$$

$$
y=(I+W_O)s+W_O\left(\lambda I+\sum_i A_i\right)^{-1}
\sum_i A_i(z_i-s)
$$

- `q` is a learned vector created inside the module, initially zero.
- `U_i` is directly trained, without orthonormalization or equal-strength constraints.
- `r_i` is a learned sigmoid reliability predictor for each source.
- `beta` and `lambda` are positive learned scalars; `epsilon` is positive.
- There is no standard attention layer or softmax over sources in the fusion core.

The defaults use width 32, subspace rank 4 and metric rank 8. The code solves the reconstruction system directly instead of explicitly computing an inverse. Positive `lambda` makes the system invertible for nonnegative source weights, even when the subspace factors have dependent columns.

## Reported results

The report compares QGSR with mean fusion, concatenation MLP, late prediction averaging, scalar gating, Deep Sets-style fusion, GRU fusion, query attention, a two-layer Transformer, RBF SVM and random forest.

| Dataset | QGSR clean accuracy | Observed rank |
|---|---:|---:|
| Images: Fashion-MNIST | 87.63% | 4/11 |
| Body motion: HARTH | 91.69% | 8/11 |
| Gas-sensor drift | 61.39% | 1/11 |
| EEG eye state | 14.32% | 7/11 |
| Motor-current features | 55.43% | 4/11 |
| Room occupancy | 87.47% | 7/11 |

Neural scores average three training seeds. Classical methods use one validation-selected fit per split. HARTH accuracy gives held-out participants equal weight; the other tests give records equal weight. Ranks describe these fixed configurations, not a universal ordering.

QGSR did not lead consistently. The EEG test uses one short recording, the motor-current groups are feature proxies, and the room test has incomplete class coverage. The PDF gives the splits, uncertainty intervals, class-balance metrics, fault scores, parameter counts and dataset attribution.

**Timing note:** this compact module uses a dense reference solve. The report's CPU times measured equivalent optimized solves and inference caching, so they are not timings for this listing. Full-model parameter counts also include encoders and classifiers.

The implementation was checked against the original formula and one trained fusion state from each dataset, including output/gradient checks, float32 and double precision, and missing-source cases. The complete Python listing is also printed in the PDF.

## Licence and contact

QGSR custom software is licensed under **GNU AGPL-3.0-only**. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). The identical `LICENSE-AGPL-3.0.txt` copy preserves the filename used by the module's licence header. Third-party software and datasets retain their own terms.

A separate commercial licence may be discussed with the copyright holder. Compliant AGPL use, including commercial use, does not require a separate commercial licence. The licence covers copyrightable code, not exclusive rights to the mathematical ideas or independently written implementations.

**Contact:** [rickeyernest1990@gmail.com](mailto:rickeyernest1990@gmail.com)
