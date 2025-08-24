import re
import math
from collections import Counter
from typing import Dict, Any, Optional, List
from .ai import readability_metrics, sentiment_score, grammar_suggestions, topic_relevance, ml_score

SENTENCE_SPLIT = re.compile(r'[.!?]+(?=\s|$)')
WORD_RE = re.compile(r"[A-Za-z']+")

COMMON_MISSPELLINGS = {
    'teh': 'the',
    'recieve': 'receive',
    'definately': 'definitely',
    'occured': 'occurred',
    'seperate': 'separate',
    'wich': 'which',
    'adress': 'address',
    'becuase': 'because',
    'enviroment': 'environment',
    'goverment': 'government',
}

HEDGING_WORDS = {
    'maybe', 'perhaps', 'somewhat', 'kinda', 'sort of', 'sorta',
    'i think', 'i believe', 'i guess'
}

def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]

def words(text: str):
    return WORD_RE.findall(text.lower())

def flesch_kincaid_proxy(total_words, total_sentences, syllables_estimate):
    if total_sentences == 0 or total_words == 0:
        return 0.0
    ASL = total_words / total_sentences
    ASW = syllables_estimate / total_words
    score = 206.835 - 1.015 * ASL - 84.6 * ASW
    return max(0.0, min(100.0, score))

def estimate_syllables(w):
    vowels = "aeiouy"
    w = w.lower()
    count, prev_is_vowel = 0, False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def grade_text(text: str):
    sents = split_sentences(text)
    tokens = words(text)
    total_words = len(tokens)
    total_sents = len(sents)
    length_score = min(100.0, (total_words / 150.0) * 100.0)
    avg_sent_len = (total_words / total_sents) if total_sents else 0
    if avg_sent_len <= 20:
        clarity_score = 100.0
    else:
        clarity_score = max(0.0, 100.0 - (avg_sent_len - 20) * 3.0)
    unique = len(set(tokens))
    ttr = (unique / total_words) if total_words else 0
    vocab_score = min(100.0, ttr * 200.0) 
    syllables = sum(estimate_syllables(w) for w in tokens)
    readability_score = flesch_kincaid_proxy(total_words, total_sents, syllables)
    passive_hits = len(re.findall(r'\b(am|is|are|was|were|be|been|being)\b\s+\b\w+ed\b\s*(?:by\b)?', text, flags=re.I))
    hedges = [h for h in HEDGING_WORDS if h in text.lower()]
    counts = Counter(tokens)
    repeated = [w for w, c in counts.items() if c >= 5 and len(w) > 3]
    miss = [(w, COMMON_MISSPELLINGS[w]) for w in tokens if w in COMMON_MISSPELLINGS]
    overall = round(
        0.30 * length_score +
        0.30 * clarity_score +
        0.20 * vocab_score +
        0.20 * readability_score, 2
    )
    feedback_lines = []
    feedback_lines.append(f"Words: {total_words}, Sentences: {total_sents}, Avg sentence length: {avg_sent_len:.1f}")
    if total_words < 150:
        feedback_lines.append("• Expand your essay to at least ~150 words to cover the topic more fully.")
    if avg_sent_len > 24:
        feedback_lines.append("• Consider splitting long sentences for clarity.")
    if ttr < 0.4:
        feedback_lines.append("• Try to vary your vocabulary to avoid repetition.")
    if readability_score < 60:
        feedback_lines.append("• Simplify sentence structure and prefer familiar words to improve readability.")
    if passive_hits > 0:
        feedback_lines.append(f"• Detected {passive_hits} possible passive constructions; prefer active voice when appropriate.")
    if hedges:
        feedback_lines.append(f"• Hedging language found ({', '.join(hedges)}); be more confident and precise.")
    if repeated:
        feedback_lines.append(f"• Repeated words appear often: {', '.join(sorted(repeated)[:8])}. Consider synonyms.")
    if miss:
        corrections = ', '.join(f"{a}→{b}" for a, b in miss[:10])
        feedback_lines.append(f"• Possible misspellings: {corrections}.")
    if not feedback_lines[1:]:
        feedback_lines.append("Great job! Clear, varied, and readable.")
    ai_analysis = {}
    
    try:
        ai_analysis['readability'] = readability_metrics(text).get("readability_score", 0)
    except:
        ai_analysis['readability'] = readability_score

    try:
        ai_analysis['sentiment'] = sentiment_score(text).get("positivity", 0)
    except:
        ai_analysis['sentiment'] = 0
    try:
        grammar_res = grammar_suggestions(text)
        ai_analysis['grammar'] = {
            "issues": grammar_res.get("issues", []),
            "grammar_score": grammar_res.get("grammar_score", 0)
        }
    except:
        ai_analysis['grammar'] = {"issues": [], "grammar_score": 0}

    try:
        ai_analysis['topic_relevance'] = topic_relevance(text, "")
    except:
        ai_analysis['topic_relevance'] = 0

    try:
        ml_overall, feat = ml_score(text)
        ai_analysis['ml_overall'] = ml_overall
        ai_analysis['ml_features'] = feat
    except:
        ai_analysis['ml_overall'] = overall
        ai_analysis['ml_features'] = {}

    return {
        'length_score': round(length_score, 2),
        'clarity_score': round(clarity_score, 2),
        'vocab_score': round(vocab_score, 2),
        'readability_score': round(readability_score, 2),
        'overall': overall,
        'feedback': "\n".join(feedback_lines),
        'stats': {
            'total_words': total_words,
            'total_sentences': total_sents,
            'avg_sentence_len': avg_sent_len,
        },
        'meta': {
            'passive_hits': passive_hits,
            'hedges': hedges,
            'repeated': repeated,
            'misspellings': miss, 
            'ttr': round(ttr, 3),
        },
        'ai': ai_analysis
    }