"""Optional local document search (no embeddings needed; works with any provider).

Indexes .txt/.md (and .pdf if ``pypdf`` is installed) files in a folder the
user chooses, splits them into ~1,200-character chunks and ranks chunks by BM25.
Documents never leave the machine unless a cloud provider is selected, in which
case only the top passages returned to the model are sent.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+")


def _read(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    return p.read_text(errors="ignore")


class DocIndex:
    def __init__(self, folder: str | Path, chunk: int = 1200):
        self.chunks: list[tuple[str, str]] = []
        for p in sorted(Path(folder).rglob("*")):
            if p.suffix.lower() in {".txt", ".md", ".pdf"}:
                text = _read(p)
                for i in range(0, len(text), chunk):
                    self.chunks.append((f"{p.name}#{i // chunk}", text[i:i + chunk]))
        self.toks = [Counter(TOKEN.findall(c[1].lower())) for c in self.chunks]
        self.df = Counter(t for tk in self.toks for t in tk)
        self.avg = sum(sum(t.values()) for t in self.toks) / max(len(self.toks), 1)

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = TOKEN.findall(query.lower())
        n = len(self.chunks)
        scores = []
        for i, tk in enumerate(self.toks):
            L = sum(tk.values())
            s = 0.0
            for t in q:
                if t in tk:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tk[t] * 2.2 / (tk[t] + 1.2 * (0.25 + 0.75 * L / max(self.avg, 1)))
            scores.append(s)
        top = sorted(range(n), key=lambda i: -scores[i])[:k]
        return [{"source": self.chunks[i][0], "score": round(scores[i], 3), "text": self.chunks[i][1]} for i in top if scores[i] > 0]
