from __future__ import annotations

import itertools
import logging
import math
import re
from typing import Any

import numpy as np

logger = logging.getLogger("analysis_dashboard.xai_engine")


def _tokenise(text: str) -> list[str]:
    raw = re.split(r"[\s,;:.!?()\[\]{}\"\'/\\]+", text.lower())
    return [t for t in raw if t]


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _build_fixed_vocabulary(documents: list[str]) -> tuple[dict[str, int], np.ndarray]:
    tokenised_docs = [_tokenise(doc) for doc in documents]
    vocab_set = {token for docs in tokenised_docs for token in docs}
    vocab = sorted(vocab_set)
    vocab_index = {w: i for i, w in enumerate(vocab)}
    
    n_docs = len(documents)
    n_vocab = len(vocab)
    
    if n_vocab == 0:
        return {"<empty>": 0}, np.array([1.0], dtype=np.float32)

    df = np.zeros(n_vocab, dtype=np.float32)
    for docs in tokenised_docs:
        unique_tokens = set(docs)
        for token in unique_tokens:
            df[vocab_index[token]] += 1

    idf = np.log((n_docs + 1) / (df + 1)) + 1.0
    return vocab_index, idf


def _vectorise_text_fixed(tokens: list[str], vocab_index: dict[str, int], idf: np.ndarray) -> np.ndarray:
    tf = np.zeros(len(vocab_index), dtype=np.float32)
    if not tokens:
        return tf
        
    for token in tokens:
        if token in vocab_index:
            tf[vocab_index[token]] += 1
            
    tf /= len(tokens)
    return tf * idf


def _anomaly_score(feature_vector: np.ndarray, baseline: np.ndarray) -> float:
    if baseline.shape[0] < 2:
        return 0.0
    mean = baseline.mean(axis=0)
    std  = baseline.std(axis=0) + 1e-8
    z    = (feature_vector - mean) / std
    return float(np.sqrt(np.sum(z ** 2)))


def run_proxy_lime(
    log_entries: list[dict],
    graph_context: str = "",
    n_perturbations: int = 30,
    top_k_tokens: int = 10,
) -> dict[str, Any]:
    logger.info("run_proxy_lime: %d entries", len(log_entries))

    if not log_entries:
        return {"entries": [], "method": "proxy_lime", "error": "No entries provided."}

    results = []
    corpus = [str(e.get("content", "")) for e in log_entries]
    if graph_context:
        corpus.append(graph_context)
        
    vocab_index, idf = _build_fixed_vocabulary(corpus)

    for entry_idx, entry in enumerate(log_entries):
        content = str(entry.get("content", "")).strip()
        if not content:
            continue

        entry_id = entry.get("id") or entry.get("entry_id") or f"entry_{entry_idx}"
        interaction_type = entry.get("mcp_interaction_type") or entry.get("interaction_type", "unknown")

        tokens = _tokenise(content)
        if len(tokens) < 2:
            results.append({
                "entry_id": entry_id,
                "interaction_type": interaction_type,
                "token_importances": [],
                "top_tokens": [],
                "content_preview": content[:120],
                "note": "Content too short for LIME perturbation.",
            })
            continue

        ref_vec = _vectorise_text_fixed(tokens, vocab_index, idf)

        token_importance_sum = {t: 0.0 for t in set(tokens)}
        token_perturbation_count = {t: 0 for t in set(tokens)}
        rng = np.random.default_rng(seed=42 + entry_idx)

        for p_idx in range(n_perturbations):
            mask_ratio = 0.1 + (0.4 * p_idx / max(n_perturbations - 1, 1))
            n_to_mask = max(1, int(len(tokens) * mask_ratio))

            masked_positions = set(rng.choice(len(tokens), size=n_to_mask, replace=False))
            masked_tokens = {tokens[i] for i in masked_positions}

            masked_tokens_list = [tokens[i] for i in range(len(tokens)) if i not in masked_positions]
            masked_vec = _vectorise_text_fixed(masked_tokens_list, vocab_index, idf)

            similarity = _cosine_similarity(ref_vec, masked_vec)
            confidence_drop = 1.0 - similarity

            for token in masked_tokens:
                if token in token_importance_sum:
                    token_importance_sum[token] += confidence_drop
                    token_perturbation_count[token] += 1

        raw_importances = {}
        for token in token_importance_sum:
            count = token_perturbation_count[token]
            if count > 0:
                raw_importances[token] = token_importance_sum[token] / count

        if raw_importances:
            max_val = max(abs(v) for v in raw_importances.values()) or 1.0
            normalised = {t: round(v / max_val, 4) for t, v in raw_importances.items()}
        else:
            normalised = {}

        sorted_tokens = sorted(normalised.items(), key=lambda x: abs(x[1]), reverse=True)
        token_importances = [{"token": t, "importance": imp} for t, imp in sorted_tokens]
        top_tokens = [item["token"] for item in token_importances[:top_k_tokens]]

        results.append({
            "entry_id": entry_id,
            "interaction_type": interaction_type,
            "token_importances": token_importances[:top_k_tokens],
            "top_tokens": top_tokens,
            "content_preview": content[:120],
            "n_tokens": len(tokens),
            "n_perturbations": n_perturbations,
        })

    return {
        "entries": results,
        "method": "proxy_lime",
        "graph_context_used": bool(graph_context),
        "total_entries": len(results),
    }


_SHAP_FEATURES = ["latency_ms", "token_count", "content_length"]


def _extract_feature_vector(entry: dict) -> np.ndarray:
    content = str(entry.get("content", ""))
    return np.array([
        float(entry.get("latency_ms") or 0.0),
        float(entry.get("token_count") or 0.0),
        float(len(content)),
    ], dtype=np.float64)


def _shapley_values(
    feature_vector: np.ndarray,
    baseline_matrix: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    k = len(feature_names)
    all_indices = list(range(k))
    shap_values = {name: 0.0 for name in feature_names}

    if baseline_matrix.shape[0] < 2:
        return shap_values

    for size in range(k):
        for subset in itertools.combinations(all_indices, size):
            subset_set = set(subset)
            weight = (
                math.factorial(len(subset)) *
                math.factorial(k - len(subset) - 1)
            ) / math.factorial(k)

            for i in range(k):
                if i in subset_set:
                    continue

                baseline_mean = baseline_matrix.mean(axis=0)

                vec_s = baseline_mean.copy()
                for j in subset:
                    vec_s[j] = feature_vector[j]

                vec_s_i = vec_s.copy()
                vec_s_i[i] = feature_vector[i]

                score_s_i = _anomaly_score(vec_s_i, baseline_matrix)
                score_s   = _anomaly_score(vec_s,   baseline_matrix)
                marginal   = score_s_i - score_s

                shap_values[feature_names[i]] += weight * marginal

    return {k: round(v, 4) for k, v in shap_values.items()}


def run_proxy_shap(
    log_entries: list[dict],
    graph_context: str = "",
) -> dict[str, Any]:
    logger.info("run_proxy_shap: %d entries", len(log_entries))

    if not log_entries:
        return {
            "feature_contributions": {},
            "per_entry_shap":        [],
            "anomaly_count":         0,
            "dominant_feature":      None,
            "method":                "proxy_shap",
            "error":                 "No entries provided.",
        }

    feature_matrix = np.array(
        [_extract_feature_vector(e) for e in log_entries],
        dtype=np.float64,
    )

    anomaly_flags = np.zeros(len(log_entries), dtype=bool)
    try:
        from sklearn.ensemble import IsolationForest
        if feature_matrix.shape[0] >= 4:
            iso = IsolationForest(
                n_estimators=100,
                contamination=0.2,
                random_state=42,
            )
            preds = iso.fit_predict(feature_matrix)
            anomaly_flags = preds == -1
            logger.info(
                "IsolationForest: %d/%d entries flagged as anomalous.",
                anomaly_flags.sum(), len(log_entries),
            )
        else:
            anomaly_flags[:] = True
            logger.info("Too few samples for IsolationForest — treating all as anomalous.")
    except ImportError:
        logger.warning("sklearn not available — using z-score anomaly detection.")
        if feature_matrix.shape[0] >= 2:
            means = feature_matrix.mean(axis=0)
            stds  = feature_matrix.std(axis=0) + 1e-8
            z_scores = np.abs((feature_matrix - means) / stds)
            anomaly_flags = z_scores.max(axis=1) > 2.0
        else:
            anomaly_flags[:] = True

    per_entry_shap: list[dict] = []
    all_shap_values: list[dict[str, float]] = []

    for idx, entry in enumerate(log_entries):
        entry_id = entry.get("id") or entry.get("entry_id") or f"entry_{idx}"
        fvec     = feature_matrix[idx]
        a_score  = _anomaly_score(fvec, feature_matrix)

        shap_vals = _shapley_values(
            feature_vector=fvec,
            baseline_matrix=feature_matrix,
            feature_names=_SHAP_FEATURES,
        )
        all_shap_values.append(shap_vals)

        per_entry_shap.append({
            "entry_id":        entry_id,
            "interaction_type": (
                entry.get("mcp_interaction_type") or
                entry.get("interaction_type", "unknown")
            ),
            "is_anomalous":    bool(anomaly_flags[idx]),
            "anomaly_score":   round(a_score, 4),
            "shap_values":     shap_vals,
            "feature_values":  {
                _SHAP_FEATURES[i]: round(float(fvec[i]), 2)
                for i in range(len(_SHAP_FEATURES))
            },
        })

    feature_contributions: dict[str, float] = {}
    for fname in _SHAP_FEATURES:
        vals = [sv.get(fname, 0.0) for sv in all_shap_values]
        feature_contributions[fname] = round(float(np.mean(vals)), 4)

    dominant_feature = max(
        feature_contributions,
        key=lambda f: abs(feature_contributions[f]),
        default=None,
    )

    logger.info(
        "run_proxy_shap complete. anomaly_count=%d dominant_feature=%s",
        int(anomaly_flags.sum()), dominant_feature,
    )

    return {
        "feature_contributions": feature_contributions,
        "per_entry_shap":        per_entry_shap,
        "anomaly_count":         int(anomaly_flags.sum()),
        "dominant_feature":      dominant_feature,
        "method":                "proxy_shap",
        "features_used":         _SHAP_FEATURES,
        "graph_context_used":    bool(graph_context),
        "total_entries":         len(log_entries),
    }