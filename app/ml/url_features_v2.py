"""URL v2 helpers (importable so joblib artefacts unpickle outside the training script)."""
from app.ml import features as F


def host_only(u: str) -> str:
    """Preprocessor for the host-only char-ngram model."""
    return F.url_host(F.normalise_url(u))


def parse_psl(text: str) -> frozenset[str]:
    """Public Suffix List rules (ICANN + private sections), comments stripped."""
    return frozenset(l.strip().lower() for l in text.splitlines() if l.strip() and not l.startswith("//"))


def registrable_domain(host: str, rules: frozenset[str]) -> str:
    """eTLD+1 per the Public Suffix List (wildcard and exception rules supported).

    Shared-hosting platforms (pages.dev, github.io, blogspot.com ...) are in the PSL private
    section, so 'foo.pages.dev' is its own registrable domain and a popular platform does not
    vouch for everything hosted on it.
    """
    labels = host.split(".")
    if len(labels) < 2 or all(p.isdigit() for p in labels):
        return host
    suffix_len = 1
    for i in range(len(labels)):
        cand = ".".join(labels[i:])
        wild = ".".join(["*"] + labels[i + 1:])
        if "!" + cand in rules:
            suffix_len = len(labels) - i - 1
            break
        if cand in rules or wild in rules:
            suffix_len = len(labels) - i
            break
    n = min(len(labels), suffix_len + 1)
    return ".".join(labels[-n:])
