import os

import numpy as np
import torch

# Determining inputs for a noise-label cache file (§3). The filename losslessly
# encodes each of these, so different inputs map to different files and a
# mismatched noise realization can never be silently reused.
LABEL_CACHE_KEYS = ('dataset', 'noise_percent', 'noise_seed', 'noise_algo')

# Version tag for the derangement algorithm below. Bump whenever the noise
# generation logic changes, so stale caches get a new name instead of being
# reused under the old one.
NOISE_ALGO_VERSION = 1

CACHE_LABELS_DIR = os.path.join('cache', 'labels')


def noise_cache_filename(dataset_name, noise_percent, noise_seed, noise_algo=NOISE_ALGO_VERSION):
	return (
		f'{dataset_name.lower()}_noise{noise_percent}'
		f'_nseed{int(noise_seed)}_algo{int(noise_algo)}_labels.pt'
	)


def noise_cache_path(dataset_name, noise_percent, noise_seed, noise_algo=NOISE_ALGO_VERSION):
	return os.path.join(
		CACHE_LABELS_DIR,
		noise_cache_filename(dataset_name, noise_percent, noise_seed, noise_algo),
	)


def _link_cache_into_run_dir(run_dir, labels_path):
	"""Symlink the shared cache file into the run dir as ``labels`` so the run is
	browsable as self-contained (§3). No-op if unset or already linked."""
	if not run_dir:
		return
	link = os.path.join(run_dir, 'labels')
	if os.path.lexists(link):
		return
	target = os.path.relpath(os.path.abspath(labels_path), os.path.abspath(run_dir))
	os.symlink(target, link)


def _to_numpy_labels(targets):
	if isinstance(targets, torch.Tensor):
		return targets.detach().cpu().numpy().astype(np.int64, copy=True)
	return np.asarray(targets, dtype=np.int64).copy()


def _allocate_noisy_counts(true_labels, num_noisy):
	classes, counts = np.unique(true_labels, return_counts=True)
	if num_noisy == 0:
		return classes, np.zeros_like(counts)

	expected = counts.astype(np.float64) * float(num_noisy) / float(len(true_labels))
	noisy_counts = np.floor(expected).astype(np.int64)
	remainder = int(num_noisy - noisy_counts.sum())

	if remainder > 0:
		fractions = expected - noisy_counts
		order = np.argsort(-fractions)
		for class_pos in order:
			if remainder == 0:
				break
			if noisy_counts[class_pos] >= counts[class_pos]:
				continue
			noisy_counts[class_pos] += 1
			remainder -= 1

	return classes, noisy_counts


def _sample_noisy_indices(true_labels, noise_fraction, rng):
	num_samples = len(true_labels)
	num_noisy = int(noise_fraction * num_samples)
	if noise_fraction > 0.0 and num_samples > 1:
		num_noisy = max(num_noisy, 2)
	num_noisy = min(num_noisy, num_samples)

	if num_noisy == 0:
		return np.array([], dtype=np.int64)

	classes, noisy_counts = _allocate_noisy_counts(true_labels, num_noisy)

	if noisy_counts.sum() == 0:
		noisy_counts[0] = min(1, np.sum(true_labels == classes[0]))

	active_classes = np.flatnonzero(noisy_counts > 0)
	if active_classes.size == 1 and classes.size > 1:
		dominant = active_classes[0]
		alternative_candidates = np.flatnonzero(noisy_counts < np.array([np.sum(true_labels == c) for c in classes]))
		alternative_candidates = alternative_candidates[alternative_candidates != dominant]
		if alternative_candidates.size > 0 and noisy_counts[dominant] > 1:
			noisy_counts[dominant] -= 1
			noisy_counts[alternative_candidates[0]] += 1

	noisy_indices = []
	for class_pos, class_id in enumerate(classes):
		class_count = int(noisy_counts[class_pos])
		if class_count == 0:
			continue
		class_indices = np.flatnonzero(true_labels == class_id)
		chosen = rng.choice(class_indices, size=class_count, replace=False)
		noisy_indices.append(chosen)

	if not noisy_indices:
		return np.array([], dtype=np.int64)

	noisy_indices = np.concatenate(noisy_indices, axis=0).astype(np.int64, copy=False)
	rng.shuffle(noisy_indices)
	return noisy_indices


def _sample_proportional_mismatched_labels(selected_true_labels, reference_labels, rng):
	selected_true_labels = np.asarray(selected_true_labels, dtype=np.int64)
	reference_labels = np.asarray(reference_labels, dtype=np.int64)
	if selected_true_labels.size == 0:
		return selected_true_labels.copy()

	unique_labels, reference_counts = np.unique(reference_labels, return_counts=True)
	if unique_labels.size < 2:
		raise ValueError('Cannot create noisy labels when the dataset has fewer than two classes.')

	reference_probs = reference_counts.astype(np.float64)
	reference_probs /= reference_probs.sum()
	permuted = np.empty_like(selected_true_labels)

	for class_id in np.unique(selected_true_labels):
		class_mask = selected_true_labels == class_id
		candidate_mask = unique_labels != class_id
		candidate_labels = unique_labels[candidate_mask]
		candidate_probs = reference_probs[candidate_mask]
		candidate_probs /= candidate_probs.sum()
		permuted[class_mask] = rng.choice(
			candidate_labels,
			size=int(class_mask.sum()),
			replace=True,
			p=candidate_probs,
		)

	if np.any(permuted == selected_true_labels):
		raise RuntimeError('Failed to sample mismatched noisy labels for the selected subset.')

	return permuted


def _derange_selected_labels(selected_true_labels, rng, reference_labels=None):
	selected_true_labels = np.asarray(selected_true_labels, dtype=np.int64)
	if selected_true_labels.size == 0:
		return selected_true_labels.copy()

	if reference_labels is None:
		reference_labels = selected_true_labels
	else:
		reference_labels = np.asarray(reference_labels, dtype=np.int64)

	unique_labels, counts = np.unique(selected_true_labels, return_counts=True)
	if unique_labels.size < 2:
		return _sample_proportional_mismatched_labels(selected_true_labels, reference_labels, rng)

	for _ in range(256):
		permuted = selected_true_labels[rng.permutation(selected_true_labels.size)]
		if np.all(permuted != selected_true_labels):
			return permuted

	max_count = int(counts.max())
	if max_count > selected_true_labels.size - max_count:
		return _sample_proportional_mismatched_labels(selected_true_labels, reference_labels, rng)

	sort_order = np.argsort(selected_true_labels, kind='stable')
	sorted_labels = selected_true_labels[sort_order]
	rotated = np.roll(sorted_labels, -max_count)
	inverse_order = np.empty_like(sort_order)
	inverse_order[sort_order] = np.arange(sort_order.size)
	permuted = rotated[inverse_order]

	if np.any(permuted == selected_true_labels):
		raise RuntimeError('Failed to build a derangement for the selected noisy labels.')

	return permuted


def _load_noise_payload(labels_path):
	payload = torch.load(labels_path, map_location='cpu')
	if 'true_labels' not in payload or 'noisy_labels' not in payload or 'noisy_indices' not in payload:
		raise ValueError(
			f'Noise label file at {labels_path} must contain true_labels, noisy_labels, and noisy_indices.'
		)

	true_labels = _to_numpy_labels(payload['true_labels'])
	noisy_labels = _to_numpy_labels(payload['noisy_labels'])
	noisy_indices = np.asarray(payload['noisy_indices'], dtype=np.int64)
	return true_labels, noisy_labels, noisy_indices


def _save_noise_payload(labels_path, true_labels, noisy_labels, noisy_indices):
	labels_dir = os.path.dirname(labels_path)
	if labels_dir:
		os.makedirs(labels_dir, exist_ok=True)
	torch.save(
		{
			'true_labels': torch.from_numpy(true_labels).long(),
			'noisy_labels': torch.from_numpy(noisy_labels).long(),
			'noisy_indices': torch.from_numpy(noisy_indices).long(),
		},
		labels_path,
	)


def set_dataset_targets(dataset, labels):
	labels = np.asarray(labels, dtype=np.int64)
	if isinstance(getattr(dataset, 'targets', None), torch.Tensor):
		dataset.targets = torch.from_numpy(labels).long()
	else:
		dataset.targets = labels.tolist()

	if hasattr(dataset, 'samples'):
		dataset.samples = [
			(path, int(label)) for (path, _), label in zip(dataset.samples, labels.tolist())
		]
	if hasattr(dataset, 'imgs'):
		dataset.imgs = list(dataset.samples)


def apply_or_generate_label_noise(dataset, num_classes, dataset_config, logger, dataset_name,
								  seed=None, run_dir=None):
	noise_fraction = float(dataset_config.get('noise_percent', 0.0))
	if not 0.0 <= noise_fraction <= 1.0:
		raise ValueError(f'dataset.noise_percent must be in [0, 1], got {noise_fraction}')

	rng_seed = int(dataset_config.get('noise_seed', seed if seed is not None else 0))
	labels_path = noise_cache_path(dataset_name, noise_fraction, rng_seed)

	true_labels = _to_numpy_labels(dataset.targets)
	if np.any(true_labels < 0) or np.any(true_labels >= int(num_classes)):
		raise ValueError(f'{dataset_name} labels must be in [0, {num_classes}), got invalid values.')

	if os.path.isfile(labels_path):
		cached_true_labels, noisy_labels, noisy_indices = _load_noise_payload(labels_path)
		if cached_true_labels.shape != true_labels.shape:
			raise ValueError(
				f'Cached noisy labels at {labels_path} have shape {cached_true_labels.shape}, '
				f'expected {true_labels.shape}.'
			)
		if not np.array_equal(cached_true_labels, true_labels):
			raise ValueError(
				f'Cached true labels at {labels_path} do not match the current {dataset_name} dataset.'
			)
		logger.info(f'Loaded cached noisy labels for {dataset_name} from {labels_path}')
	else:
		rng = np.random.default_rng(rng_seed)
		noisy_indices = _sample_noisy_indices(true_labels, noise_fraction, rng)
		noisy_labels = true_labels.copy()
		if noisy_indices.size > 0:
			noisy_labels[noisy_indices] = _derange_selected_labels(
				true_labels[noisy_indices],
				rng,
				reference_labels=true_labels,
			)
		if os.path.exists(labels_path):
			raise FileExistsError(
				f'Noise cache {labels_path} appeared unexpectedly; refusing to overwrite.'
			)
		_save_noise_payload(labels_path, true_labels, noisy_labels, noisy_indices)
		logger.info(f'Saved noisy labels for {dataset_name} to {labels_path}')

	_link_cache_into_run_dir(run_dir, labels_path)

	if noisy_labels.shape != true_labels.shape:
		raise ValueError(
			f'Noisy labels for {dataset_name} have shape {noisy_labels.shape}, expected {true_labels.shape}.'
		)
	if np.any(noisy_labels < 0) or np.any(noisy_labels >= int(num_classes)):
		raise ValueError(
			f'Noisy labels for {dataset_name} must be in [0, {num_classes}), got invalid values.'
		)
	if noisy_indices.size > 0 and np.any(noisy_labels[noisy_indices] == true_labels[noisy_indices]):
		raise ValueError(f'Noisy labels for {dataset_name} must differ from the true labels at noisy indices.')

	set_dataset_targets(dataset, noisy_labels)
	applied_labels = _to_numpy_labels(dataset.targets)
	if not np.array_equal(applied_labels, noisy_labels):
		raise ValueError(
			f'Applied noisy labels for {dataset_name} do not match the labels stored on the dataset object.'
		)

	diff_indices = np.flatnonzero(applied_labels != true_labels)
	sorted_noisy_indices = np.sort(noisy_indices)
	indices_match_dataset_diff = np.array_equal(diff_indices, sorted_noisy_indices)
	if not indices_match_dataset_diff:
		raise ValueError(
			f'Applied noisy labels for {dataset_name} do not match noisy_indices from {labels_path}.'
		)

	num_noisy = int(diff_indices.size)
	fraction_noisy = float(num_noisy / len(true_labels)) if len(true_labels) > 0 else 0.0
	logger.info(
		f'Noisy label summary for {dataset_name}: configured_noise_fraction={noise_fraction:.4f}, '
		f'num_noisy={num_noisy}, fraction_noisy={fraction_noisy:.4f}, '
		f'noisy_indices_match_dataset_diff={indices_match_dataset_diff}'
	)
	return {
		'true_labels': torch.from_numpy(true_labels).long(),
		'noisy_indices': torch.from_numpy(noisy_indices).long(),
	}
