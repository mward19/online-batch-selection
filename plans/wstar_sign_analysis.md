# w_star sign analysis

## Ground truth convention (the reference)

`data/make_blobs_teacher.py` line 39:
```python
centers = np.stack([+cscale * w_star, -cscale * w_star])
```
- **Label 0** → cluster centered at **+w_star** → projection z > 0
- **Label 1** → cluster centered at **−w_star** → projection z < 0

The Bayes-optimal rule is therefore: predict class 0 iff `x @ w_star > 0`.

---

## Instance-by-instance

### 1. Teacher model weights — `data/make_blobs_teacher.py` line 65 ✓

```python
'fc.weight': torch.stack([+w_hat_t, -w_hat_t])
```

- Logit 0 = `x @ (+w_hat)` → large when x is near +w_star → predicts 0 ✓  
- Logit 1 = `x @ (−w_hat)` → large when x is near −w_star → predicts 1 ✓  
- `argmax` gives class 0 when `x @ w_hat > 0`. **CORRECT.**


---

### 2. Teacher test accuracy (RhoLoss, Bayesian) — `methods/RhoLoss.py` and `methods/Bayesian.py` ✓

Both call `torch.argmax(outputs, dim=1)` on the teacher model outputs. Since the teacher weights are correct (see above), argmax produces the right label. **CORRECT.**

---

### 3. Irreducible / teacher loss caching — `methods/RhoLoss.py` line 67 ✓

```python
loss = F.cross_entropy(outputs, targets, reduction='none')
```

Teacher outputs already use the correct sign (instance 1), so cross-entropy against true labels is meaningful. **CORRECT.**

---

### 4. Bayesian selection loss caching — `methods/Bayesian.py` line 81 ✓

```python
loss = -F.cross_entropy(outputs, targets, reduction='none')
```

Negated cross-entropy (log-likelihood) used as a selection score. Teacher outputs use the correct sign, so the negation is intentional and correct for its purpose. **CORRECT.**

---

### 5. `wstar_test_acc` prediction — `data/makeblobs.py` line 142 ✗

```python
z = X_test @ w_star
preds = (z > 0).astype(np.int64)   # BUG: gives 1 when z>0, but label 0 is at z>0
```

`(z > 0)` casts `True → 1`, `False → 0`. But z > 0 means x is near +w_star → label **0**, not 1. Every prediction is the mirror image of the correct label, producing accuracy ≈ 1 − Φ(cscale) instead of Φ(cscale).

**Fix:**
```python
preds = (z <= 0).astype(np.int64)
```

[[Nice find my friend]]

---

### 6. Validation notebook — `notebooks/blobs-hyperplane-validation.ipynb` cell 6 ✓

```python
pred_wstar = (X @ w_star < 0).astype(int)
pred_what  = (X @ w_hat  < 0).astype(int)
```

`< 0` gives 1 when z < 0 (near −w_star → label 1) and 0 when z ≥ 0 (near +w_star → label 0). **CORRECT** — and the direct counterpart that exposes the bug in instance 5: the notebook achieves ~69% while `makeblobs.py` with `> 0` produces ~31%.

---

### 8. `bayes_accuracy` formula — `slurm_run_blobs_deep_linear.py` line 59 ✓

```python
cfg['bayes_accuracy'] = round(float(ndtr(cs)), 3)   # ndtr = Φ
```

`bayes_accuracy_derivation.md` derives Φ(cscale) analytically assuming the correct rule "predict 0 iff z > 0". **CORRECT** — consistent with the ground truth convention. The mismatch between `bayes_accuracy` (~69%) and the observed `wstar_test_acc` (~31%) is entirely explained by the bug in instance 5.
