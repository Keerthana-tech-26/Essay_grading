from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    import language_tool_python
    _LT = language_tool_python.LanguageToolPublicAPI('en-US')
except Exception:
    _LT = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

_ANALYZER = SentimentIntensityAnalyzer()

ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "essay_scorer.pkl")
def readability_metrics(text: str) -> Dict[str, float]:
    """Return common readability indices scaled into 0–100 where higher=better."""
    if not text or len(text.split()) < 5:
        return {"flesch": 0.0, "grade_level": 0.0, "readability_score": 0.0}

    flesch = textstat.flesch_reading_ease(text)
    flesch_scaled = max(0, min(100, flesch))

    grade = textstat.text_standard(text, float_output=True)
    grade_scaled = 100 - max(0, min(100, (grade - 5) * (100 / 15.0)))
    readability = 0.6 * flesch_scaled + 0.4 * grade_scaled
    return {
        "flesch": round(flesch, 2),
        "grade_level": round(grade, 2),
        "readability_score": round(readability, 2),
    }

def sentiment_score(text: str) -> Dict[str, float]:
    if not text or len(text.split()) < 3:
        return {"compound": 0.0, "positivity": 50.0}
    s = _ANALYZER.polarity_scores(text)
    comp = s["compound"]
    positivity = round((comp + 1) * 50, 2)
    return {"compound": round(comp, 3), "positivity": positivity}

def grammar_suggestions(text: str, max_issues: int = 20) -> Dict[str, object]:
    issues: List[Dict[str, str]] = []
    corrected = text
    if _LT:
        try:
            matches = _LT.check(text)
            for m in matches[:max_issues]:
                rep = ", ".join(m.replacements[:3]) if m.replacements else ""
                issues.append({
                    "message": m.message,
                    "context": text[max(0, m.offset-20): m.offset + m.errorLength + 20],
                    "suggest": rep
                })
            corrected = language_tool_python.utils.correct(text, matches)
        except Exception:
            pass
    if not issues:
        words = text.split()
        repeats = []
        for a, b in zip(words, words[1:]):
            if a.lower() == b.lower():
                repeats.append(a)
        if repeats:
            issues.append({
                "message": "Repeated word(s) detected",
                "context": " ... ".join(repeats[:5]),
                "suggest": "Remove duplicates"
            })
        if "  " in text:
            issues.append({
                "message": "Multiple consecutive spaces",
                "context": "Contains double spaces",
                "suggest": "Use single spaces"
            })

    score_penalty = min(30, len(issues) * 2)
    grammar_score = max(0, 100 - score_penalty)
    return {"issues": issues, "corrected_text": corrected, "grammar_score": grammar_score}

def topic_relevance(text: str, topic: Optional[str] = None, keywords: Optional[List[str]] = None) -> float:
    """
    Compute a simple topic relevance (0–100) by TF-IDF cosine similarity
    between essay and topic+keywords.
    """
    if not topic and not keywords:
        return 50.0

    target = (topic or "") + " " + (" ".join(keywords) if keywords else "")
    docs = [text, target]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X[0], X[1])[0][0] 
    return round(sim * 100, 2)

def _extract_features(text: str) -> Dict[str, float]:
    words = text.split()
    sentences = [s for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    uniq = len(set(w.lower().strip(".,;:!?\"'()[]{}")) for w in words) if words else 0
    avg_sent_len = (len(words) / len(sentences)) if sentences else 0
    type_token_ratio = (uniq / len(words) * 100) if words else 0

    read = readability_metrics(text)
    sent = sentiment_score(text)
    gram = grammar_suggestions(text)

    return {
        "word_count": len(words),
        "avg_sentence_len": avg_sent_len,
        "type_token_ratio": type_token_ratio,
        "readability": read["readability_score"],
        "sentiment": sent["positivity"],
        "grammar": gram["grammar_score"],
        "issue_count": len(gram["issues"]),
    }


def _features_to_vector(feat: Dict[str, float]) -> list[float]:
    return [
        feat["word_count"],
        feat["avg_sentence_len"],
        feat["type_token_ratio"],
        feat["readability"],
        feat["sentiment"],
        feat["grammar"],
        feat["issue_count"],
    ]


def ml_score(text: str) -> Tuple[float, Dict[str, float]]:
    feat = _extract_features(text)
    if os.path.exists(ML_MODEL_PATH):
        try:
            model = joblib.load(ML_MODEL_PATH)
            x = [_features_to_vector(feat)]
            y = float(model.predict(x)[0])
            return max(0.0, min(100.0, y)), feat
        except Exception:
            pass
    richness = min(100.0, feat["type_token_ratio"])
    length_score = min(100.0, feat["word_count"] / 400.0 * 100.0)
    structure = min(100.0, feat["avg_sentence_len"] / 25.0 * 100.0) if feat["avg_sentence_len"] else 60.0

    score = (
        0.30 * feat["readability"] +
        0.30 * feat["grammar"] +
        0.10 * feat["sentiment"] +
        0.15 * richness +
        0.10 * length_score +
        0.05 * structure
    )
    return round(score, 2), feat
