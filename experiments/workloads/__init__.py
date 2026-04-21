"""
Graph workload templates for experiments.
Contains implementations of standard graph algorithms used in the paper.
"""

from .workloads import (
    query_wcc,
    query_sssp,
    query_pr,
    query_bfs,
    query_subiso,
    query_gar_match
)

__all__ = [
    'query_wcc',
    'query_sssp',
    'query_pr',
    'query_bfs',
    'query_subiso',
    'query_gar_match'
]