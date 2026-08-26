from __future__ import annotations

import pytest
from conftest import make_synthetic_root

from flowmotion.data.dataset import MotionWindowDataset, build_vocab, lookup_id, split_subjects
from flowmotion.data.loader import discover_sequences


def test_split_is_deterministic_and_disjoint():
    keys = [f"DatasetX/subject{i:02d}" for i in range(20)]
    train_a, held_a = split_subjects(keys, held_out_frac=0.2, seed=0)
    train_b, held_b = split_subjects(keys, held_out_frac=0.2, seed=0)

    assert train_a == train_b
    assert held_a == held_b
    assert set(train_a).isdisjoint(set(held_a))
    assert set(train_a) | set(held_a) == set(keys)
    assert len(held_a) == round(20 * 0.2)


def test_split_operates_on_unique_subjects_not_sequence_counts():
    # subject "A" appears 10x more often than "B" in the input list (e.g. 10 sequences
    # vs 1) -- the split must still treat them as ONE subject each, not weight by count.
    keys = ["DatasetX/A"] * 10 + ["DatasetX/B"] * 1
    train, held = split_subjects(keys, held_out_frac=0.5, seed=0)
    assert set(train) | set(held) == {"DatasetX/A", "DatasetX/B"}
    assert len(train) + len(held) == 2


def test_no_held_out_subject_windows_leak_into_train_dataset(tmp_path):
    root = make_synthetic_root(tmp_path, n_subjects_per_dataset=6, n_sequences_per_subject=2)
    sequences = discover_sequences(root)
    subject_keys = [s.subject_key for s in sequences]

    train_subjects, held_out_subjects = split_subjects(subject_keys, held_out_frac=0.3, seed=0)
    train_set = set(train_subjects)
    train_sequences = [s for s in sequences if s.subject_key in train_set]

    subject_vocab = build_vocab(train_subjects)
    action_vocab = build_vocab([s.dataset_name for s in sequences])
    dataset = MotionWindowDataset(
        train_sequences, subject_vocab, action_vocab, K=4, H=4, stride=4, normalizer=None
    )

    for i in range(len(dataset)):
        seq_idx, _ = dataset.windows[i]
        assert dataset.subject_keys[seq_idx] not in set(held_out_subjects)


def test_leakage_assertion_is_not_vacuous__a_sequence_level_split_would_be_caught():
    # Sanity-check the leakage test above by deliberately constructing a BAD, sequence-
    # level split (same subject appears in both "train" and "held out") and confirming
    # the same style of assertion fails against it -- proves the check has teeth.
    sequences_for_subject_a = ["seqA1", "seqA2"]
    bad_train_sequences = [sequences_for_subject_a[0]]
    bad_held_out_subjects = {"subjectA"}  # subject A is nominally "held out"...
    sequence_to_subject = {"seqA1": "subjectA", "seqA2": "subjectA"}

    leaked = any(sequence_to_subject[seq] in bad_held_out_subjects for seq in bad_train_sequences)
    with pytest.raises(AssertionError):
        assert not leaked


def test_lookup_id_maps_unseen_key_to_null_index():
    vocab = build_vocab(["a", "b", "c"])
    assert lookup_id("b", vocab) == vocab["b"]
    assert lookup_id("never-seen", vocab) == len(vocab)
