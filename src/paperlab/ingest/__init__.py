from . import arxiv, openalex
from .base import run_search, store_papers

FETCHERS = {
    "arxiv": arxiv.search,
    "openalex": openalex.search,
}

__all__ = ["FETCHERS", "run_search", "store_papers"]
