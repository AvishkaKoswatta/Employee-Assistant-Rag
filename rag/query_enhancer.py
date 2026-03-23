# type: ignore
# query_enhancer.py
#
# Drop-in for your RAG pipeline.
# query.py calls:  from query_enhancer import enhance_query
#
# enhance_query(raw: str) -> str
#   Returns an expanded, reformulated query string ready for
#   Pinecone similarity search.

import re
from typing import List, Dict

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "my", "me", "we", "our", "you", "your", "it", "its",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
    "about", "what", "how", "when", "where", "which", "who", "that",
    "this", "there", "their", "they", "from", "have", "has", "had",
    "not", "no", "any", "all", "more", "some", "get", "use", "need",
    "want", "go", "like", "just", "so", "as",
}

# ---------------------------------------------------------------------------
# Handbook-specific synonym / expansion map
# ---------------------------------------------------------------------------

SYNONYM_MAP: Dict[str, List[str]] = {
    # Leave & time off
    "leave":        ["paid time off", "absence", "vacation", "holiday", "day off"],
    "pto":          ["paid time off", "leave", "vacation", "annual leave"],
    "vacation":     ["paid time off", "leave", "holiday", "annual leave"],
    "sick":         ["sick leave", "illness", "medical leave", "unwell"],
    "parental":     ["parental leave", "maternity", "paternity", "birth", "child"],
    "maternity":    ["parental leave", "maternity leave", "birth", "child"],
    "paternity":    ["parental leave", "paternity leave", "birth", "child"],
    "unpaid":       ["unpaid leave", "leave without pay"],

    # Work arrangements
    "remote":       ["remote work", "work from home", "wfh", "telecommute"],
    "wfh":          ["work from home", "remote work", "telecommute"],
    "flexible":     ["flexible hours", "flextime", "flexible schedule"],
    "hours":        ["working hours", "schedule", "shift", "start time"],

    # Compensation & pay
    "salary":       ["pay", "compensation", "wage", "payroll"],
    "pay":          ["salary", "compensation", "wage", "payroll"],
    "overtime":     ["extra hours", "additional hours", "ot"],
    "payroll":      ["pay", "salary", "payment", "compensation"],
    "reimbursement":["expenses", "refund", "travel expenses", "receipt"],
    "bonus":        ["reward", "incentive", "referral reward"],

    # Conduct & policies
    "harassment":   ["harassment policy", "bullying", "misconduct", "unwanted advance", "sexual harassment"],
    "discrimination":["equal opportunity", "bias", "prejudice", "unfair treatment"],
    "violence":     ["workplace violence", "assault", "threat", "physical harm"],
    "safety":       ["workplace safety", "hazard", "emergency", "fire safety", "risk assessment"],
    "smoking":      ["smoking policy", "cigarette", "tobacco", "designated smoking area"],
    "drugs":        ["drug-free workplace", "substance", "alcohol", "narcotics"],
    "alcohol":      ["drug-free workplace", "substance", "drugs"],
    "dress":        ["dress code", "attire", "clothing", "uniform", "professional dress"],
    "attire":       ["dress code", "clothing", "uniform", "professional dress"],
    "internet":     ["internet usage", "web browsing", "online", "websites"],
    "email":        ["corporate email", "mail", "messaging"],
    "phone":        ["cell phone usage", "mobile", "smartphone"],
    "social media": ["social media policy", "facebook", "twitter", "linkedin"],
    "conflict":     ["conflict of interest", "personal interest", "competing interest"],

    # Employment
    "hire":         ["recruitment", "hiring", "job posting", "interview"],
    "fired":        ["termination", "dismissed", "let go"],
    "termination":  ["fired", "dismissal", "exit", "let go"],
    "resign":       ["resignation", "quit", "notice period", "two weeks notice"],
    "quit":         ["resignation", "resign", "notice period"],
    "notice":       ["resignation", "two weeks notice", "notice period"],
    "probation":    ["probation period", "trial period", "new employee confirmation"],
    "promotion":    ["promotion policy", "career growth", "advancement", "internal role"],
    "performance":  ["performance review", "appraisal", "evaluation", "pip"],
    "pip":          ["performance improvement plan", "performance issues", "improvement goals"],
    "training":     ["training and development", "learning", "course", "conference", "upskilling"],
    "referral":     ["employee referral", "refer a candidate", "referral reward"],
    "background":   ["background check", "screening", "criminal check", "verification"],
    "onboarding":   ["new employee onboarding", "orientation", "induction", "joining"],
    "offboarding":  ["offboarding procedure", "exit process", "handover", "leaving"],

    # IT & security
    "password":     ["password security", "credentials", "authentication", "mfa", "account security"],
    "software":     ["acceptable software", "licensed software", "install", "application"],
    "backup":       ["data backup", "cloud storage", "file storage", "data storage"],
    "security":     ["incident reporting", "access control", "breach", "unauthorized access"],
    "access":       ["access control", "credentials", "permissions", "system access"],
    "breach":       ["security incident", "data breach", "incident reporting"],

    # Travel
    "travel":       ["business travel", "business trip", "travel reimbursement", "expenses"],
    "trip":         ["business travel", "travel policy", "travel reimbursement"],

    # Ethics
    "ethics":       ["code of ethics", "integrity", "honesty", "ethical behavior"],
    "gift":         ["gifts and hospitality", "bribe", "token", "benefit"],
    "bribery":      ["anti-corruption", "corruption", "unethical", "bribe"],
    "corruption":   ["anti-corruption policy", "bribery", "unethical practices"],

    # HR / general
    "hr":           ["human resources", "people team", "HR department"],
    "manager":      ["supervisor", "direct manager", "team lead"],
    "policy":       ["rule", "guideline", "procedure", "handbook"],
    "report":       ["notify", "inform", "raise concern", "escalate", "complaint"],
    "confidential": ["data protection", "confidentiality", "sensitive", "private"],
    "grievance":    ["grievance handling", "complaint", "concern", "dispute"],
    "whistleblower":["whistleblower policy", "report misconduct", "anonymous report"],
    "late":         ["late arrival", "tardiness", "lateness", "arrive late"],
    "attendance":   ["attendance policy", "punctuality", "working hours", "schedule"],
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in STOPWORDS]


def _expand(tokens: List[str]) -> List[str]:
    """
    For each token (and adjacent bigram) look up synonyms and collect
    all expansion terms. Returns a deduplicated list: original tokens
    first, then expansion terms.
    """
    extra: List[str] = []
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

    for term in tokens + bigrams:
        if term in SYNONYM_MAP:
            extra.extend(SYNONYM_MAP[term])

    seen = set(tokens)
    result = list(tokens)
    for e in extra:
        if e not in seen:
            seen.add(e)
            result.append(e)

    return result


# ---------------------------------------------------------------------------
# Public API  — this is what query.py imports
# ---------------------------------------------------------------------------

def enhance_query(raw_query: str) -> str:
    """
    Expand a raw user query with handbook-specific synonyms and related terms.

    Parameters
    ----------
    raw_query : str
        The original question from the user.

    Returns
    -------
    str
        An enriched query string for use with a vector-store retriever.
        The original phrasing is preserved at the front; expansion terms
        are appended so semantic embeddings capture related concepts.

    Example
    -------
    >>> enhance_query("Can I work from home?")
    'work home remote work work from home wfh telecommute'

    >>> enhance_query("How many days of sick leave do I get?")
    'days sick leave sick leave illness medical leave unwell paid time off absence vacation holiday day off'
    """
    tokens   = _tokenise(raw_query)
    expanded = _expand(tokens)
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# Optional: CLI smoke-test  (python query_enhancer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    queries = sys.argv[1:] or [
        "Can I work from home?",
        "How many days of sick leave do I get?",
        "What happens if I am always late?",
        "How do I report harassment?",
        "What is the password policy?",
        "How do I resign from my job?",
        "Can I accept a gift from a client?",
        "How long is the probation period?",
        "What are the travel reimbursement rules?",
        "How do I refer a friend for a job?",
    ]

    print(f"{'ORIGINAL QUERY':<48} ENHANCED QUERY")
    print("-" * 110)
    for q in queries:
        print(f"{q:<48} {enhance_query(q)}")