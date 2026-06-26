"""NTK diagnostic (§5.8.4, Phase 5c). Holds the full ``NTKDiagnostics`` compute
engine (moved here from the deleted ``method_utils/ntk.py``) plus the thin
``NTK`` leaf. The engine keeps heavy per-step state (initial kernel, initial
eigenvectors, teacher kernel, spectrum history) and is consumed only by the NTK
leaf on its own fixed balanced subset, so nothing is shared and there is no
``NTKKernel`` dependency.
"""

import os
import tempfile
import numpy as np

import torch
import torch.nn.functional as F

from methods.method_utils.build_teacher_model import build_teacher_model
from methods.diagnostics.base import Diagnostic, DiagnosticInfo

try:
    from torch.func import functional_call, jacrev, vmap

    _HAS_TORCH_FUNC = True
except ImportError:
    _HAS_TORCH_FUNC = False

try:
    from trak import TRAKer

    _HAS_PROJECTION_NTK = True
except ImportError:
    _HAS_PROJECTION_NTK = False


class NTKDiagnostics:
    def __init__(
        self,
        logger,
        fixed_train_loader,
        project_root,
        dataset_name,
        artifact_stem,
        seed,
        config,
        num_classes,
        ntk_max_samples,
        ntk_top_k,
        ntk_variant,
        ntk_eigenvalue_concentration_checkpoints,
        enabled,
        save_spectrum=False,
    ):
        self.logger = logger
        self.project_root = project_root
        self.dataset_name = dataset_name
        self.artifact_stem = artifact_stem
        self.seed = seed
        self.config = config
        self.loader = fixed_train_loader
        self.num_classes = int(num_classes)
        self.ntk_max_samples = int(ntk_max_samples)
        self.ntk_top_k = int(ntk_top_k)
        self.ntk_variant = str(ntk_variant).strip().lower()
        # Only support the canonical variants. Projections must use the projection
        # backend via `_compute_projection_ntk`.
        if self.ntk_variant not in {'pseudo', 'trace', 'full', 'conj-kernel', 'proj-pseudo', 'proj-trace'}:
            self.logger.info(
                f"Warning: unknown diagnostics ntk_variant '{ntk_variant}'."
            )
            self.ntk_variant = None
        self.ntk_eigenvalue_concentration_checkpoints = sorted(set(ntk_eigenvalue_concentration_checkpoints)) # ensure in order
        self.enabled = bool(enabled) # True if using NTK diagnostics
        self.save_spectrum = bool(save_spectrum)
        diagnostics_cfg = self.config.get('diagnostics', {})
        # Projection config kept for the projection backend only
        self.ntk_proj_dim = int(diagnostics_cfg.get('diagnostics_ntk_proj_dim', 1024))
                                
        self._ntk_init_kernel = None
        self._ntk_init_norm = None
        self._ntk_num_samples = 0
        self._ntk_init_top_eigenvectors = None
        self._ntk_init_top_k = 0
        self._ntk_warned_fallback = False
        self._ntk_warned_default_cap = False
        self._ntk_warned_invalid_top_k = False
        self._ntk_subset_inputs_cpu = None
        self._ntk_subset_labels_cpu = None
        self._ntk_fixed_indices_cpu = None
        self._teacher_model = None
        self._teacher_kernel = None
        self._teacher_norm = None
        self._teacher_eigenvalues = None
        self._teacher_top_eigenvectors = None
        self._teacher_top_k = 0
        self._teacher_kernel_initialized = False
        self._teacher_support_warned = False
        self._disable_torch_func_path = False
        self._conj_kernel_warned_feature_fallback = False
        self._projection_ntk_warned_missing = False
        self._projection_ntk_warned_failed = False
        self._spectrum = []
        self._spectrum_steps = []
        self._spectrum_num_samples = []
        self.ntk_projection_batch_size = int(
            diagnostics_cfg.get(
                'diagnostics_ntk_projection_batch_size',
                diagnostics_cfg.get('ntk_projection_batch_size', 128),
            )
        )
        self.ntk_indices_path = os.path.join(
            self.project_root,
            'ntk_indices',
            f'{self.dataset_name}.npy',
        )
        self.spectrum_path = None
        if self.save_spectrum:
            spectrum_dir = os.path.join(
                self.project_root,
                'spectrum',
                self.dataset_name,
            )
            os.makedirs(spectrum_dir, exist_ok=True)
            self.spectrum_path = os.path.join(
                spectrum_dir,
                f'{self.artifact_stem}.p',
            )

    @staticmethod
    def _kernel_inner_product(lhs, rhs):
        return torch.sum(lhs * rhs)

    @staticmethod
    def _center_kernel(kernel):
        row_mean = kernel.mean(dim=1, keepdim=True)
        col_mean = kernel.mean(dim=0, keepdim=True)
        total_mean = kernel.mean()
        return kernel - row_mean - col_mean + total_mean

    @staticmethod
    def _strip_module_prefix(state_dict):
        if not state_dict:
            return state_dict
        if all(key.startswith('module.') for key in state_dict.keys()):
            return {key[len('module.'):]: value for key, value in state_dict.items()}
        return state_dict

    def _warn_teacher_unsupported(self, message):
        """Ensures warning only appears once"""
        if not self._teacher_support_warned:
            self.logger.info(message)
            self._teacher_support_warned = True

    def _build_teacher_model(self, device):
        source = self.config.get('diagnostics', {}).get('ntk_teacher_model', {}).get(
            'source',
            self.config.get('teacher_model_source'),
        )

        if source not in {'timm', 'local_pretrained'}:
            self._warn_teacher_unsupported(
                f'Warning: NTK teacher model source {source} is not supported; skipping teacher NTK comparisons.'
            )
            return None

        teacher_model = build_teacher_model(self.config, self.logger)
        teacher_model.to(device)
        teacher_model.eval()
        self._teacher_model = teacher_model
        return self._teacher_model

    def _sorted_eigenvalues(self, kernel):
        eigenvalues = torch.linalg.eigvalsh(kernel)
        return torch.flip(eigenvalues, dims=(0,))

    def _save_spectrum(self):
        "Saves ntk spectrum locally to specified path"
        if not self.save_spectrum or self.spectrum_path is None:
            return
        payload = {
            'eigenvalues': self._spectrum,
            'steps': self._spectrum_steps,
            'num_samples': self._spectrum_num_samples,
        }
        if self._teacher_eigenvalues is not None:
            payload['teacher_eigenvalues'] = self._teacher_eigenvalues
        torch.save(payload, self.spectrum_path)

    def _record_spectrum(self, total_step, eigenvalues_desc, num_samples):
        """Records ntk spectrum to be saved locally"""
        if not self.save_spectrum:
            return
        self._spectrum.append(eigenvalues_desc.detach().cpu().to(dtype=torch.float32))
        self._spectrum_steps.append(int(total_step))
        self._spectrum_num_samples.append(int(num_samples))
        self._save_spectrum()

    def _ensure_teacher_kernel(self, device):
        """Builds teacher model and computes ntk metrics if not already computed"""
        if self._teacher_kernel_initialized:
            return
        self._teacher_kernel_initialized = True

        teacher_model = self._build_teacher_model(device)
        if teacher_model is None:
            return

        if self.ntk_variant == 'conj-kernel':
            teacher_ntk, _ = self._compute_conj_kernel(teacher_model, device)
        elif self.ntk_variant == 'proj-pseudo':
            teacher_ntk, _ = self._compute_proj_pseudo_ntk(teacher_model, device)
        elif self.ntk_variant == 'proj-trace':
            teacher_ntk, _ = self._compute_proj_trace_ntk(teacher_model, device)
        elif self.ntk_variant == 'full':
            teacher_ntk, _ = self._compute_full_ntk(teacher_model, device)
        elif self.ntk_variant == 'pseudo':
            teacher_ntk, _ = self._compute_pseudo_ntk(teacher_model, device)
        else:
            # default / 'trace'
            teacher_ntk, _ = self._compute_trace_ntk(teacher_model, device)
        if teacher_ntk is None:
            self._warn_teacher_unsupported(
                'Warning: unable to compute teacher NTK; skipping teacher NTK comparisons.'
            )
            return

        self._teacher_kernel = teacher_ntk.detach().cpu().to(dtype=torch.float64)
        self._teacher_norm = float(torch.norm(self._teacher_kernel, p='fro').item())
        teacher_eigenvalues, teacher_eigenvectors = torch.linalg.eigh(self._teacher_kernel)
        teacher_eigenvalues_desc = torch.flip(teacher_eigenvalues, dims=(0,))
        teacher_eigenvectors_desc = torch.flip(teacher_eigenvectors, dims=(1,))
        self._teacher_eigenvalues = teacher_eigenvalues_desc.to(dtype=torch.float32)
        self._teacher_top_k = min(self.ntk_top_k, int(teacher_eigenvectors_desc.size(1)))
        if self._teacher_top_k > 0:
            self._teacher_top_eigenvectors = teacher_eigenvectors_desc[:, : self._teacher_top_k]
        self._save_spectrum()

    def _flatten_grads(self, grads, params):
        flat = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat.append(torch.zeros_like(param, memory_format=torch.contiguous_format).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        return torch.cat(flat, dim=0)

    def _select_balanced_indices_from_candidates(self, candidate_indices, index_to_label, per_class):
        """Randomly permute candidate indices, then greedily pick a class-balanced subset."""
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)

        rng = np.random.default_rng(int(self.seed))
        candidate_indices = rng.permutation(candidate_indices)

        selected_counts = np.zeros(self.num_classes, dtype=np.int64)
        selected = []
        for idx in candidate_indices:
            cls_idx = int(index_to_label.get(int(idx), -1))
            if 0 <= cls_idx < self.num_classes and selected_counts[cls_idx] < per_class:
                selected.append(int(idx))
                selected_counts[cls_idx] += 1
                if np.all(selected_counts >= per_class):
                    break

        effective_per_class = int(selected_counts.min())
        if effective_per_class == per_class: # all classes are to max capacity
            return np.asarray(selected, dtype=np.int64), effective_per_class

        trimmed = [] # need to even out class-imbalance by trimming
        trimmed_counts = np.zeros(self.num_classes, dtype=np.int64)
        for idx in selected:
            cls_idx = int(index_to_label[idx])
            if trimmed_counts[cls_idx] < effective_per_class:
                trimmed.append(idx)
                trimmed_counts[cls_idx] += 1
        return np.asarray(trimmed, dtype=np.int64), effective_per_class

    def _collect_ntk_inputs(self, device):
        """
        Either 
        (1) data already in _ntk_subset_inputs_cpu, 
        (2) data not loaded but selected indices stored from previous runs, or 
        (3) generates, loads, and stores new subset of class balanced data
        """
        if self._ntk_subset_inputs_cpu is None: # case (1)
            dataset = self.loader.dataset
            per_class = self.ntk_max_samples // self.num_classes

            if os.path.isfile(self.ntk_indices_path): #case (2)
                loaded = np.load(self.ntk_indices_path, allow_pickle=False)
                selected_indices = np.asarray(loaded, dtype=np.int64).reshape(-1)
                self._ntk_fixed_indices_cpu = selected_indices
            else: # case (3)
                dataset_indices = np.arange(len(dataset), dtype=np.int64)
                targets_source = getattr(dataset, 'targets', None)
                if targets_source is None and hasattr(dataset, 'dataset'):
                    targets_source = getattr(dataset.dataset, 'targets', None)
                if targets_source is None:
                    dataset_labels = np.asarray(
                        [int(dataset[i]['target']) for i in dataset_indices.tolist()],
                        dtype=np.int64,
                    )
                else:
                    dataset_labels = np.asarray(targets_source, dtype=np.int64).reshape(-1)

                index_to_label = {
                    int(idx): int(lbl)
                    for idx, lbl in zip(dataset_indices.tolist(), dataset_labels.tolist())
                }
                selected_indices, _ = self._select_balanced_indices_from_candidates(
                    dataset_indices,
                    index_to_label,
                    per_class,
                )
                if selected_indices is None or selected_indices.size == 0:
                    return None, 0

                indices_dir = os.path.dirname(self.ntk_indices_path)
                if indices_dir:
                    os.makedirs(indices_dir, exist_ok=True)
                np.save(self.ntk_indices_path, selected_indices.astype(np.int64, copy=False))
                self._ntk_fixed_indices_cpu = selected_indices

            selected_inputs = []
            selected_labels = []
            for idx in selected_indices.tolist():
                sample = dataset[int(idx)]
                selected_inputs.append(sample['input'].detach().cpu())
                selected_labels.append(int(sample['target']))

            self._ntk_subset_inputs_cpu = torch.stack(selected_inputs, dim=0).contiguous()
            self._ntk_subset_labels_cpu = torch.tensor(selected_labels, dtype=torch.long)

        total_samples = int(self._ntk_subset_inputs_cpu.size(0))
        self._ntk_num_samples = total_samples
        return self._ntk_subset_inputs_cpu.to(device), total_samples

    def _extract_penultimate_features(self, model, inputs):
        """Gets penultimate layer activation features for ll-NTK"""
        with torch.no_grad():
            if hasattr(model, 'feat_nograd_forward'):
                outputs = model.feat_nograd_forward(inputs)
            elif hasattr(model, 'net') and hasattr(model.net, 'feat_nograd_forward'):
                outputs = model.net.feat_nograd_forward(inputs)
            else:
                outputs = model(inputs, need_features=True)

        features = outputs[1]
        
        return features

    def _compute_conj_kernel(self, model, device):
        """Computes NTK with respect to the just the last layer gradient"""
        inputs, total_samples = self._collect_ntk_inputs(device)

        try:
            features = self._extract_penultimate_features(model, inputs)
        except Exception as exc:
            if not self._conj_kernel_warned_feature_fallback:
                self.logger.info(
                    f'Warning: failed to extract penultimate features for conj_kernel ({exc}); skipping NTK diagnostics.'
                )
                self._conj_kernel_warned_feature_fallback = True
            return None, 0

        features = features.detach().reshape(features.size(0), -1)
        feature_kernel = features @ features.t()
        return feature_kernel.detach().cpu(), total_samples


    def _compute_torch_func_kernel(self, model, device, variant):
        """
        Computes NTK variants using memory-efficient torch.func implementations.

        Supported variants:
          - 'pseudo': N x N Pseudo-NTK (sums logits before gradient). Single quick pass.
          - 'trace':  N x N Trace-NTK (sums per-class NTK matrices).
          - 'full':   NC x NC Full NTK (exact logit trajectories).
        """
        inputs, total_samples = self._collect_ntk_inputs(device)
        if inputs is None:
            return None, 0

        num_samples = inputs.shape[0]
        inputs = inputs.to(device)

        with torch.no_grad():
            sample_out = model(inputs[0:1])
            num_classes = sample_out.shape[1]

        params = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        if not params:
            return None, 0
        buffers = dict(model.named_buffers())

        if variant == 'pseudo':
            def model_pseudo(curr_params, curr_buffers, x):
                out = functional_call(model, (curr_params, curr_buffers), (x.unsqueeze(0),))
                return out.squeeze(0).sum()

            jac = vmap(jacrev(model_pseudo, argnums=0), in_dims=(None, None, 0))(params, buffers, inputs)

            ntk = None
            for param_jac in jac.values():
                j_flat = param_jac.flatten(start_dim=1)
                contribution = torch.mm(j_flat, j_flat.t())
                ntk = contribution if ntk is None else ntk + contribution

            return ntk.detach().cpu(), total_samples

        # Projections are not handled in this path. Use the projection backend
        # (`_compute_projection_ntk`) for projected trace/full kernels.
        loop_dim = num_classes

        if variant == 'full':
            ntk_dim = num_samples * loop_dim
            ntk = torch.zeros((ntk_dim, ntk_dim), device=device)
        else:
            ntk = torch.zeros((num_samples, num_samples), device=device)

        def model_scalar_logit(curr_params, curr_buffers, x, idx):
            out = functional_call(model, (curr_params, curr_buffers), (x.unsqueeze(0),)).squeeze(0)
            return out[idx]

        for i in range(loop_dim):
            get_jac = vmap(
                jacrev(lambda p, b, x: model_scalar_logit(p, b, x, i), argnums=0),
                in_dims=(None, None, 0),
            )
            jac_i = get_jac(params, buffers, inputs)
            flat_grads_i = torch.cat([j.flatten(start_dim=1) for j in jac_i.values()], dim=1)

            if variant == 'full':
                for j in range(loop_dim):
                    if j == i:
                        flat_grads_j = flat_grads_i
                    else:
                        get_jac_j = vmap(
                            jacrev(lambda p, b, x: model_scalar_logit(p, b, x, j), argnums=0),
                            in_dims=(None, None, 0),
                        )
                        jac_j = get_jac_j(params, buffers, inputs)
                        flat_grads_j = torch.cat([p.flatten(start_dim=1) for p in jac_j.values()], dim=1)

                    sub_block = torch.mm(flat_grads_i, flat_grads_j.t())
                    ntk[i::loop_dim, j::loop_dim] = sub_block
            else:
                ntk += torch.mm(flat_grads_i, flat_grads_i.t())

        return ntk.detach().cpu(), total_samples

    def _compute_trace_ntk(self, model, device):
        return self._compute_kernel_by_variant(model, device, variant='trace')

    def _compute_full_ntk(self, model, device):
        return self._compute_kernel_by_variant(model, device, variant='full')

    def _compute_pseudo_ntk(self, model, device):
        return self._compute_kernel_by_variant(model, device, variant='pseudo')

    def _compute_proj_pseudo_ntk(self, model, device):
        return self._compute_projection_ntk(model, device, task='pNTK')

    def _compute_proj_trace_ntk(self, model, device):
        return self._compute_projection_ntk(model, device, task='trNTK')

    def _compute_projection_ntk(self, model, device, task):
        """Computes projection NTK using pnnl/projection_ntk TRAKer implementation."""
        if not _HAS_PROJECTION_NTK:
            if not self._projection_ntk_warned_missing:
                self.logger.info(
                    'Warning: projection_ntk backend requested but `trak` is not installed. '
                    'Install pnnl/projection_ntk (traker) to use proj-pseudo or proj-trace variants.'
                )
                self._projection_ntk_warned_missing = True
            return None, 0

        inputs, total_samples = self._collect_ntk_inputs(device)
        if inputs is None or total_samples <= 0:
            return None, 0

        if self._ntk_subset_labels_cpu is None or self._ntk_subset_labels_cpu.numel() != total_samples:
            self.logger.info('Warning: NTK subset labels unavailable; cannot run projection NTK backend.')
            return None, 0

        batch_size = max(1, min(int(self.ntk_projection_batch_size), int(total_samples)))
        labels = self._ntk_subset_labels_cpu.to(device)

        try:
            with tempfile.TemporaryDirectory(prefix='ntk_proj_') as save_dir:
                traker_kwargs = {
                    'model': model,
                    'task': task,
                    'train_set_size': int(total_samples),
                    'save_dir': save_dir,
                    'device': str(device),
                    'proj_dim': self.ntk_proj_dim,
                    'use_half_precision': False,
                    'proj_max_batch_size': int(batch_size),
                    'logging_level': 100,
                }
                if task == 'trNTK':
                    traker_kwargs['num_classes'] = int(self.num_classes)

                traker = TRAKer(**traker_kwargs)
                traker.load_checkpoint(model.state_dict(), model_id=0)

                for start in range(0, total_samples, batch_size):
                    end = min(start + batch_size, total_samples)
                    batch_inputs = inputs[start:end]
                    batch_labels = labels[start:end]
                    traker.featurize(batch=[batch_inputs, batch_labels], num_samples=int(end - start))

                if task == 'pNTK':
                    grads = torch.as_tensor(np.asarray(traker.saver.current_store['grads']), dtype=torch.float64)
                    kernel = grads @ grads.t()
                    return kernel, total_samples

                kernel = None
                for cls_idx in range(self.num_classes):
                    key = f'grads_{cls_idx}'
                    if key not in traker.saver.current_store:
                        continue
                    grads_cls = torch.as_tensor(np.asarray(traker.saver.current_store[key]), dtype=torch.float64)
                    contrib = grads_cls @ grads_cls.t()
                    kernel = contrib if kernel is None else kernel + contrib

                if kernel is None:
                    self.logger.info('Warning: projection trNTK did not produce class-wise gradients.')
                    return None, 0
                return kernel, total_samples
        except Exception as exc:
            if not self._projection_ntk_warned_failed:
                self.logger.info(
                    f'Warning: projection NTK backend failed with error ({exc}); skipping projection NTK diagnostics.'
                )
                self._projection_ntk_warned_failed = True
            return None, 0

    

    def _compute_kernel_by_variant(self, model, device, variant):
        """Computes requested NTK variant using torch.func (fast) or autograd (slow)."""
        if not self._ntk_warned_default_cap and self.ntk_max_samples == 1000:
            self.logger.info('NTK diagnostics: using default diagnostics_ntk_max_samples=1000.')
            self._ntk_warned_default_cap = True

        if _HAS_TORCH_FUNC and not self._disable_torch_func_path:
            try:
                return self._compute_torch_func_kernel(
                    model,
                    device,
                    variant=variant,
                )
            except RuntimeError as err:
                err_msg = str(err).lower()
                if 'out of memory' in err_msg or ('cuda' in err_msg and 'memory' in err_msg):
                    self._disable_torch_func_path = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.info(
                        'Warning: torch.func NTK path ran out of memory; autograd fallback removed; skipping NTK diagnostics for this checkpoint.'
                    )
                    return None, 0
                raise
        # No autograd fallback: if torch.func path is unavailable or disabled, skip NTK diagnostics.
        if not _HAS_TORCH_FUNC:
            self.logger.info('Warning: torch.func not available and autograd fallback removed; skipping NTK diagnostics.')
            return None, 0
        if self._disable_torch_func_path:
            self.logger.info('Warning: torch.func NTK path disabled due to previous OOM; autograd fallback removed; skipping NTK diagnostics.')
            return None, 0
        # Should not reach here normally; default to skipping.
        return None, 0

    def _get_ntk_targets_one_hot(self, num_samples):
        if self._ntk_subset_labels_cpu is None:
            return None
        if self._ntk_subset_labels_cpu.numel() != num_samples:
            self.logger.info(
                'Warning: NTK subset labels do not match the empirical kernel size; skipping label-based NTK diagnostics.'
            )
            return None
        return F.one_hot(
            self._ntk_subset_labels_cpu,
            num_classes=self.num_classes,
        ).to(dtype=torch.float64)

    def _get_full_ntk_target_vector(self, kernel_size):
        if self._ntk_subset_labels_cpu is None:
            return None
        one_hot_targets = F.one_hot(
            self._ntk_subset_labels_cpu,
            num_classes=self.num_classes,
        ).to(dtype=torch.float64)
        target_vector = one_hot_targets.reshape(-1)
        if int(target_vector.numel()) != int(kernel_size):
            self.logger.info(
                'Warning: Full NTK target vector size does not match kernel size; skipping label-based NTK diagnostics.'
            )
            return None
        return target_vector

    def _compute_kernel_target_alignment_metrics(self, kernel, num_samples):
        log_data = {}
        kernel_norm = torch.linalg.norm(kernel, ord='fro')

        if self.ntk_variant == 'full':
            target_vector = self._get_full_ntk_target_vector(kernel.size(0))
            if target_vector is None:
                return log_data

            target_norm = torch.linalg.norm(target_vector).pow(2)
            kta_denom = kernel_norm * target_norm
            if kta_denom.item() > 0.0:
                kta_numerator = torch.dot(target_vector, kernel @ target_vector)
                kta = kta_numerator / kta_denom
                log_data['diagnostics/ntk_kernel_target_alignment'] = float(kta.item())

            centered_kernel = self._center_kernel(kernel)
            centered_target_vector = target_vector - target_vector.mean()
            centered_kernel_norm = torch.linalg.norm(centered_kernel, ord='fro')
            centered_target_norm = torch.linalg.norm(centered_target_vector).pow(2)
            cka_denom = centered_kernel_norm * centered_target_norm
            if cka_denom.item() > 0.0:
                cka_numerator = torch.dot(centered_target_vector, centered_kernel @ centered_target_vector)
                cka = cka_numerator / cka_denom
                log_data['diagnostics/ntk_centered_kernel_alignment'] = float(cka.item())
            return log_data

        one_hot_targets = self._get_ntk_targets_one_hot(num_samples)
        if one_hot_targets is None:
            return log_data

        target_kernel = one_hot_targets @ one_hot_targets.t()
        target_norm = torch.linalg.norm(target_kernel, ord='fro')
        kta_denom = kernel_norm * target_norm
        if kta_denom.item() > 0.0:
            kta = self._kernel_inner_product(kernel, target_kernel) / kta_denom
            log_data['diagnostics/ntk_kernel_target_alignment'] = float(kta.item())

        centered_kernel = self._center_kernel(kernel)
        centered_target_kernel = self._center_kernel(target_kernel)
        centered_kernel_norm = torch.linalg.norm(centered_kernel, ord='fro')
        centered_target_norm = torch.linalg.norm(centered_target_kernel, ord='fro')
        cka_denom = centered_kernel_norm * centered_target_norm
        if cka_denom.item() > 0.0:
            cka = self._kernel_inner_product(centered_kernel, centered_target_kernel) / cka_denom
            log_data['diagnostics/ntk_centered_kernel_alignment'] = float(cka.item())

        return log_data

    def _compute_ntk_spectrum_metrics(self, current_ntk, kernel_size, total_step, include_label_alignment=True):
        log_data = {}
        if kernel_size <= 0:
            return log_data, None, None

        if self.ntk_top_k <= 0:
            if not self._ntk_warned_invalid_top_k:
                self.logger.info('Warning: diagnostics_ntk_top_k must be positive. Top-k NTK diagnostics will be skipped.')
                self._ntk_warned_invalid_top_k = True
            return log_data, None, None

        kernel = current_ntk.detach().cpu().to(dtype=torch.float64)

        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
        eigenvalues_desc = torch.flip(eigenvalues, dims=(0,))
        eigenvectors_desc = torch.flip(eigenvectors, dims=(1,))
        self._record_spectrum(total_step, eigenvalues_desc, kernel_size)

        effective_k = min(self.ntk_top_k, int(kernel_size), int(eigenvalues_desc.numel()))
        if effective_k <= 0:
            return log_data, eigenvalues_desc, None

        top_eigenvalues = eigenvalues_desc[:effective_k]
        top_eigenvectors = eigenvectors_desc[:, :effective_k]
        nonnegative_top_eigenvalues = top_eigenvalues.clamp_min(0.0)
        nonnegative_eigenvalues = eigenvalues_desc.clamp_min(0.0)
        log_data['diagnostics/ntk_top_k'] = int(effective_k)

        total_eigenvalue_mass = nonnegative_eigenvalues.sum()
        if total_eigenvalue_mass.item() > 0.0:
            concentration = nonnegative_top_eigenvalues.sum() / total_eigenvalue_mass
            log_data['diagnostics/ntk_eigenvalue_concentration'] = float(concentration.item())

            for checkpoint_k in self.ntk_eigenvalue_concentration_checkpoints:
                effective_checkpoint_k = min(checkpoint_k, int(nonnegative_eigenvalues.numel()))
                if effective_checkpoint_k <= 0:
                    continue
                concentration_checkpoint = (
                    nonnegative_eigenvalues[:effective_checkpoint_k].sum() / total_eigenvalue_mass
                )
                log_data[f'diagnostics/ntk_eigenvalue_concentration_top_{checkpoint_k}'] = float(
                    concentration_checkpoint.item()
                )
        else:
            log_data['diagnostics/ntk_eigenvalue_concentration'] = 0.0
            for checkpoint_k in self.ntk_eigenvalue_concentration_checkpoints:
                log_data[f'diagnostics/ntk_eigenvalue_concentration_top_{checkpoint_k}'] = 0.0

        if total_eigenvalue_mass.item() > 0.0:
            mu = nonnegative_eigenvalues / total_eigenvalue_mass
            positive_mu = mu[mu > 0]
            if positive_mu.numel() > 0:
                shannon_entropy = -(positive_mu * positive_mu.log()).sum()
                effective_rank = torch.exp(shannon_entropy)
                log_data['diagnostics/ntk_effective_rank'] = float(effective_rank.item())
            else:
                log_data['diagnostics/ntk_effective_rank'] = 0.0
        else:
            log_data['diagnostics/ntk_effective_rank'] = 0.0

        if include_label_alignment:
            if self.ntk_variant == 'full':
                target_vector = self._get_full_ntk_target_vector(kernel_size)
                if target_vector is not None:
                    alignment = torch.abs(target_vector @ top_eigenvectors)
                    log_data['diagnostics/ntk_top_eigenvector_label_alignment_sum'] = float(alignment.pow(2).sum().item())
                    weighted_alignment = nonnegative_top_eigenvalues * alignment.pow(2)
                    log_data['diagnostics/ntk_top_eigenvector_label_alignment'] = float(weighted_alignment.sum().item())
            else:
                one_hot_targets = self._get_ntk_targets_one_hot(kernel_size)
                if one_hot_targets is not None:
                    projected_targets = one_hot_targets.t() @ top_eigenvectors
                    alignment = torch.linalg.norm(projected_targets, dim=0)
                    log_data['diagnostics/ntk_top_eigenvector_label_alignment_sum'] = float(alignment.pow(2).sum().item())
                    weighted_alignment = nonnegative_top_eigenvalues * alignment.pow(2)
                    log_data['diagnostics/ntk_top_eigenvector_label_alignment'] = float(weighted_alignment.sum().item())

        if self._ntk_init_top_eigenvectors is None:
            self._ntk_init_top_eigenvectors = top_eigenvectors.clone()
            self._ntk_init_top_k = effective_k
            log_data['diagnostics/ntk_top_eigenspace_overlap_init'] = 1.0
            return log_data, eigenvalues_desc, top_eigenvectors

        if self._ntk_init_top_eigenvectors.size(0) != top_eigenvectors.size(0):
            self.logger.info(
                'Warning: NTK eigenspace shape changed between initialization and current step; skipping eigenspace overlap logging.'
            )
            return log_data, eigenvalues_desc, top_eigenvectors

        overlap_k = min(self._ntk_init_top_k, effective_k)
        if overlap_k <= 0:
            return log_data, eigenvalues_desc, top_eigenvectors

        init_top_eigenvectors = self._ntk_init_top_eigenvectors[:, :overlap_k]
        current_top_eigenvectors = top_eigenvectors[:, :overlap_k]
        overlap = current_top_eigenvectors.t() @ init_top_eigenvectors
        log_data['diagnostics/ntk_top_eigenspace_overlap_init'] = float(
            overlap.norm(p='fro').pow(2).item() / overlap_k
        )
        return log_data, eigenvalues_desc, top_eigenvectors

    def _compute_relative_distance(self, reference_kernel, current_kernel, reference_norm):
        return float(torch.norm(current_kernel - reference_kernel, p='fro').item() / max(float(reference_norm), 1e-12))

    def _compute_angular_distance(self, reference_kernel, current_kernel, reference_norm, current_norm):
        cosine = float(self._kernel_inner_product(reference_kernel, current_kernel).item() / max(float(reference_norm) * float(current_norm), 1e-12))
        return float(1.0 - cosine)

    def log_metrics(self, model, device, total_step):
        if not self.enabled:
            return {}

        if self.ntk_variant == 'conj-kernel':
            current_ntk, _ = self._compute_conj_kernel(model, device)
        elif self.ntk_variant == 'proj-pseudo':
            current_ntk, _ = self._compute_proj_pseudo_ntk(model, device)
        elif self.ntk_variant == 'proj-trace':
            current_ntk, _ = self._compute_proj_trace_ntk(model, device)
        elif self.ntk_variant == 'full':
            current_ntk, _ = self._compute_full_ntk(model, device)
        elif self.ntk_variant == 'pseudo':
            current_ntk, _ = self._compute_pseudo_ntk(model, device)
        else:
            # default / 'trace'
            current_ntk, _ = self._compute_trace_ntk(model, device)
        if current_ntk is None:
            return {}

        current_ntk = current_ntk.detach().cpu().to(dtype=torch.float64)
        current_norm = float(torch.norm(current_ntk, p='fro').item())
        kernel_size = int(current_ntk.size(0))

        log_data = {
            'diagnostics/ntk_num_samples': int(self._ntk_num_samples),
            'diagnostics/ntk_kernel_size': int(kernel_size),
            'diagnostics/ntk_norm_fro': current_norm,
        }
        spectrum_metrics, _, current_top_eigenvectors = self._compute_ntk_spectrum_metrics(
            current_ntk,
            kernel_size,
            total_step,
            include_label_alignment=True,
        )
        log_data.update(spectrum_metrics)
        log_data.update(self._compute_kernel_target_alignment_metrics(current_ntk, self._ntk_num_samples))
        
        if self._ntk_init_kernel is None: # First checkpoint
            self._ntk_init_kernel = current_ntk
            self._ntk_init_norm = current_norm
            log_data['diagnostics/ntk_init_norm_fro'] = float(self._ntk_init_norm)
            log_data['diagnostics/ntk_relative_drift_init'] = 0.0
            log_data['diagnostics/ntk_angular_kernel_distance_init'] = 0.0
        else: # Otherwise
            log_data['diagnostics/ntk_init_norm_fro'] = float(self._ntk_init_norm)
            log_data['diagnostics/ntk_relative_drift_init'] = float(self._compute_relative_distance(
                self._ntk_init_kernel,
                current_ntk,
                self._ntk_init_norm,
            )
            )
            log_data['diagnostics/ntk_angular_kernel_distance_init'] = self._compute_angular_distance(
                self._ntk_init_kernel,
                current_ntk,
                self._ntk_init_norm,
                current_norm,
            )

        self._ensure_teacher_kernel(device) # Simply returns true or computes all necessary metrics if not already computed
        if self._teacher_kernel is not None:
            log_data['diagnostics/ntk_teacher_norm_fro'] = float(self._teacher_norm)
            log_data['diagnostics/ntk_relative_distance_teacher'] = self._compute_relative_distance(
                self._teacher_kernel,
                current_ntk,
                self._teacher_norm,
            )
            log_data['diagnostics/ntk_angular_kernel_distance_teacher'] = self._compute_angular_distance(
                self._teacher_kernel,
                current_ntk,
                self._teacher_norm,
                current_norm,
            )

            if current_top_eigenvectors is not None and self._teacher_top_eigenvectors is not None:
                overlap_k = min(
                    int(current_top_eigenvectors.size(1)),
                    int(self._teacher_top_eigenvectors.size(1)),
                )
                if overlap_k > 0:
                    overlap = (
                        current_top_eigenvectors[:, :overlap_k].t()
                        @ self._teacher_top_eigenvectors[:, :overlap_k]
                    )
                    log_data['diagnostics/ntk_top_eigenspace_overlap_teacher'] = float(
                        overlap.norm(p='fro').pow(2).item() / overlap_k
                    )

        return log_data

    def finalize(self):
        self._save_spectrum()


# --------------------------------------------------------------------------- #
# Logged leaf
# --------------------------------------------------------------------------- #

class NTK(Diagnostic):
    """Single logged leaf wrapping ``NTKDiagnostics``. Builds the engine from the
    manager's static context at construction; reads ``model``/``device`` from the
    merged context at run time."""

    def __init__(self, manager, builder, should_run=None, **params):
        super().__init__(manager, log_path=params.get("log_path"), should_run=should_run)
        sc = manager.static_context
        self._impl = NTKDiagnostics(
            logger=sc["logger"],
            fixed_train_loader=sc["fixed_train_loader"],
            project_root=sc["project_root"],
            dataset_name=sc["dataset_name"],
            artifact_stem=sc["artifact_stem"],
            seed=sc["seed"],
            config=sc["config"],
            num_classes=sc["num_classes"],
            ntk_max_samples=int(params.get("max_samples", 1000)),
            ntk_top_k=int(params.get("top_k", 10)),
            ntk_variant=str(params.get("variant", params.get("kernel_type", "trace"))),
            ntk_eigenvalue_concentration_checkpoints=params.get(
                "eigenvalue_concentration_checkpoints", [20, 40, 80]
            ),
            enabled=True,
            save_spectrum=bool(params.get("save_spectrum", False)),
        )

    def _run(self):
        ctx = self.get_context()
        total_steps = int(self.get_state().total_steps)
        return DiagnosticInfo("ntk", self._impl.log_metrics(ctx["model"], ctx["device"], total_steps))

    def finalize(self):
        self._impl.finalize()

    def __eq__(self, other):
        return isinstance(other, NTK)