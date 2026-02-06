from __future__ import annotations

from trickster.models.linear import LinearBinaryModel
from trickster.training.policy import TrainedPolicy
from trickster.training.eval import evaluate_policies, evaluate_policies_parallel
from trickster.training.model_store import default_slots, load_slot, save_latest_and_prev
from trickster.training.model_spec import ModelSpec
from trickster.training.self_play import train_self_play


def _policy_with_bias(bias: float) -> TrainedPolicy:
    m1 = LinearBinaryModel(weights={"__bias__": bias})
    m2 = LinearBinaryModel(weights={"__bias__": bias})
    return TrainedPolicy(spec=ModelSpec(kind="linear", params={}), lead_model=m1, follow_model=m2)


def test_save_latest_and_prev_rotates(tmp_path) -> None:
    p1 = _policy_with_bias(1.0)
    p2 = _policy_with_bias(2.0)

    save_latest_and_prev(p1, models_dir=tmp_path)
    slots = default_slots(tmp_path)
    assert slots.latest.exists()
    assert not slots.prev.exists()

    save_latest_and_prev(p2, models_dir=tmp_path)
    assert slots.latest.exists()
    assert slots.prev.exists()

    latest = load_slot("latest", models_dir=tmp_path)
    prev = load_slot("prev", models_dir=tmp_path)
    assert latest.lead_model.weights["__bias__"] == 2.0
    assert prev.lead_model.weights["__bias__"] == 1.0


def test_evaluate_policies_counts_deals() -> None:
    p1 = _policy_with_bias(0.0)
    p2 = _policy_with_bias(0.0)
    stats = evaluate_policies(p1, p2, games=10, seed=0)
    assert stats.deals == 10
    assert stats.a_points + stats.b_points > 0


def test_evaluate_policies_parallel_counts_deals() -> None:
    p1 = _policy_with_bias(0.0)
    p2 = _policy_with_bias(0.0)
    stats = evaluate_policies_parallel(p1, p2, games=20, seed=0, workers=2, chunk_games=5)
    assert stats.deals == 20
    assert stats.a_points + stats.b_points > 0


def test_train_self_play_resume_uses_initial_policy() -> None:
    p1 = _policy_with_bias(1.234)
    policy, stats = train_self_play(spec=ModelSpec(kind="linear", params={}), episodes=0, seed=0, initial_policy=p1)
    assert stats.episodes == 0
    assert policy.lead_model.weights["__bias__"] == 1.234

