"""Separates biometric matching decisions from the Streamlit UI."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import DEFAULT_THRESHOLD
from .store import FaceStore
from .vision import FaceObservation, normalize_feature


class Extractor(Protocol):
    model_id: str
    def extract(self, image: np.ndarray, *, annotate: bool = True) -> FaceObservation: ...


@dataclass(frozen=True)
class Match:
    name: str | None
    person_id: str | None
    similarity: float | None
    accepted: bool
    threshold: float
    elapsed_ms: float
    observation: FaceObservation
    matches: tuple[tuple[str, float], ...] = ()
    detail: str = ""


class FaceAuthService:
    def __init__(self, engine: Extractor, store: FaceStore) -> None:
        if engine.model_id != store.model_id:
            raise RuntimeError("特征提取模型与身份库版本不匹配")
        self.engine = engine
        self.store = store

    @staticmethod
    def _threshold(threshold: float) -> float:
        score = float(threshold)
        if not 0.0 < score <= 1.0:
            raise ValueError("相似度阈值应在 (0,1] 范围内。")
        return score

    def register(self, name: str, image: np.ndarray, *, consent: bool) -> tuple[dict, FaceObservation]:
        if not consent:
            raise PermissionError("请先取得注册人员的知情同意。")
        face = self.engine.extract(image)
        return self.store.enroll(
            name, face.embedding, detection_score=face.detection_score, consent=consent
        ), face

    def verify(self, name: str, image: np.ndarray, *, threshold: float = DEFAULT_THRESHOLD) -> Match:
        threshold = self._threshold(threshold)
        t0 = time.perf_counter()
        person = self.store.person_by_name(name)
        if not person:
            # Do not infer other people's names in a claimed-identity check.
            face = self.engine.extract(image)
            self.store.record_event("verify", subject=name.strip(), accepted=False,
                                    similarity=None, threshold=threshold, note="identity_not_enrolled")
            return Match(None, None, None, False, threshold, (time.perf_counter()-t0)*1000,
                         face, detail="所选身份未注册。")
        face = self.engine.extract(image)
        samples = self.store.templates_for(person["person_id"])
        if not samples:
            raise RuntimeError("身份没有可用特征样本。")
        # Templates are already L2-normalized at enrollment; a matrix product
        # handles multiple samples without a Python cosine loop.
        matrix = np.stack([vector for _, _, vector in samples]).astype(np.float32, copy=False)
        best = float(np.clip((matrix @ normalize_feature(face.embedding)).max(), -1.0, 1.0))
        accepted = best >= threshold
        self.store.record_event("verify", subject=person["name"], accepted=accepted,
                                similarity=best, threshold=threshold,
                                note="match" if accepted else "below_threshold")
        return Match(person["name"], person["person_id"], best, accepted, threshold,
                     (time.perf_counter()-t0)*1000, face,
                     detail="验证通过" if accepted else "验证未通过")

    def identify(
        self, image: np.ndarray, *, threshold: float = DEFAULT_THRESHOLD, top_k: int = 3
    ) -> Match:
        threshold = self._threshold(threshold)
        top_k = max(1, min(int(top_k), 10))
        t0 = time.perf_counter()
        face = self.engine.extract(image)
        samples = self.store.templates_for()
        scores: dict[str, tuple[str, float]] = {}
        if samples:
            matrix = np.stack([vector for _, _, vector in samples]).astype(np.float32, copy=False)
            similarities = np.clip(matrix @ normalize_feature(face.embedding), -1.0, 1.0)
            for (pid, name, _), value in zip(samples, similarities):
                score = float(value)
                old = scores.get(pid)
                if old is None or score > old[1]:
                    scores[pid] = (name, score)
        ranked = sorted(
            ((pid, name, score) for pid, (name, score) in scores.items()),
            key=lambda item: item[2], reverse=True,
        )
        best = ranked[0] if ranked else None
        accepted = best is not None and best[2] >= threshold
        # Unknown queries do not disclose names of below-threshold people.
        public_matches = tuple((name, sim) for _, name, sim in ranked if sim >= threshold)[:top_k]
        self.store.record_event(
            "identify", subject=best[1] if accepted else None, accepted=accepted,
            similarity=best[2] if best else None, threshold=threshold,
            note="match" if accepted else ("below_threshold" if best else "empty_gallery"),
        )
        return Match(
            name=best[1] if accepted else None,
            person_id=best[0] if accepted else None,
            similarity=best[2] if best else None,
            accepted=bool(accepted), threshold=threshold,
            elapsed_ms=(time.perf_counter()-t0)*1000,
            observation=face,
            matches=public_matches,
            detail="已找到已注册候选" if accepted else "未找到达到阈值的已注册身份",
        )
