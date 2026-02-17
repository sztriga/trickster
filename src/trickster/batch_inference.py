"""Cross-game batched inference server for GPU training.

When multiple games run concurrently (threaded self-play on a GPU
tier), their NN queries are funnelled through this server which
collects them into large batches, improving GPU utilisation.

Protocol::

    server = BatchInferenceServer(wrapper)
    server.start()
    # ... game threads call server.predict_both(), etc.
    server.stop()

Thread-safe: game threads call the public methods; an internal
daemon thread performs the actual inference.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from trickster.model import OnnxUltiWrapper, UltiNetWrapper


class BatchInferenceServer:
    """Collect per-game NN queries and run them as one GPU batch.

    The server wraps a ``UltiNetWrapper`` (or ``OnnxUltiWrapper``) and
    exposes the same public API.  Internally a daemon thread drains a
    queue of pending requests, merges them into a single
    ``predict_both_batch`` call, and distributes sliced results back
    via ``concurrent.futures.Future`` objects.

    Parameters
    ----------
    wrapper : UltiNetWrapper | OnnxUltiWrapper
        The underlying model wrapper (must already be on the target
        device).
    drain_ms : float
        After the first queued item arrives, wait up to this many
        milliseconds for more items before firing the batch.
    """

    def __init__(
        self,
        wrapper: UltiNetWrapper | OnnxUltiWrapper,
        *,
        drain_ms: float = 1.0,
    ) -> None:
        self._wrapper = wrapper
        self._drain_s = drain_ms / 1000.0
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background inference thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to finish and wait."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ── internal loop ─────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            items = self._collect()
            if items:
                self._process(items)

    def _collect(self) -> list:
        """Block for the first item, then drain for *drain_ms*."""
        items: list = []
        try:
            items.append(self._queue.get(timeout=0.01))
        except queue.Empty:
            return items

        deadline = time.perf_counter() + self._drain_s
        while time.perf_counter() < deadline:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _process(self, items: list) -> None:
        """Run one mega-batch and distribute sliced results."""
        all_feats: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        # (start_idx, end_idx, is_single_sample, future)
        slices: list[tuple[int, int, bool, Future]] = []

        offset = 0
        for kind, *payload in items:
            fut: Future = payload[-1]
            if kind == "single":
                feats, mask = payload[0], payload[1]
                all_feats.append(feats)
                all_masks.append(mask)
                slices.append((offset, offset + 1, True, fut))
                offset += 1
            else:  # "batch"
                feats_list, mask_list = payload[0], payload[1]
                n = len(feats_list)
                all_feats.extend(feats_list)
                all_masks.extend(mask_list)
                slices.append((offset, offset + n, False, fut))
                offset += n

        try:
            values, probs = self._wrapper.predict_both_batch(
                all_feats, all_masks,
            )
        except Exception as exc:
            for _, _, _, fut in slices:
                if not fut.done():
                    fut.set_exception(exc)
            return

        for start, end, is_single, fut in slices:
            if is_single:
                fut.set_result((float(values[start]), probs[start]))
            else:
                fut.set_result((values[start:end], probs[start:end]))

    # ── public API (called from game threads) ─────────────────────

    def predict_both(
        self,
        state_feats: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        fut: Future = Future()
        self._queue.put(("single", state_feats, mask, fut))
        return fut.result()

    def predict_both_batch(
        self,
        feats_list: list[np.ndarray],
        mask_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not feats_list:
            ad = getattr(self._wrapper, "_action_dim", None)
            if ad is None:
                ad = self._wrapper.net.action_dim
            return (
                np.empty(0, dtype=np.float32),
                np.zeros((0, ad), dtype=np.float32),
            )
        fut: Future = Future()
        self._queue.put(("batch", feats_list, mask_list, fut))
        return fut.result()

    def predict_value(self, state_feats: np.ndarray) -> float:
        dummy_mask = np.ones(self._wrapper.net.action_dim, dtype=np.bool_)
        val, _ = self.predict_both(state_feats, dummy_mask)
        return val

    def predict_policy(
        self,
        state_feats: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        _, probs = self.predict_both(state_feats, mask)
        return probs

    # ── pass-through (called rarely, no batching needed) ──────────

    def batch_value(self, states: np.ndarray) -> np.ndarray:
        return self._wrapper.batch_value(states)

    def predict_auction(self, state_feats: np.ndarray) -> np.ndarray:
        return self._wrapper.predict_auction(state_feats)

    def predict_auction_components(
        self,
        state_feats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._wrapper.predict_auction_components(state_feats)

    def sync_weights(self, net) -> None:
        if hasattr(self._wrapper, "sync_weights"):
            self._wrapper.sync_weights(net)

    # ── compatibility properties ──────────────────────────────────

    @property
    def net(self):
        return self._wrapper.net

    @property
    def device(self):
        return self._wrapper.device
