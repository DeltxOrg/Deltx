"""Pydantic data models for the AI authorship detection pipeline."""

import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class ParsedSource(BaseModel):
    """Structured representation of a parsed Python source file.

    Produced by :class:`~deltx.detection.parser.PythonSourceParser` and consumed
    by all three feature families (perplexity, stylometric, distribution).
    """

    model_config = {"arbitrary_types_allowed": True}

    source_code: str
    file_path: Path
    tokens: list[str]
    ast_tree: ast.Module | None
    identifiers: list[str]
    lines_of_code: int
    comment_lines: int
    total_lines: int
    indent_levels: list[int]
    ast_node_types: list[str]
    ast_depths: list[int]
    is_valid: bool


class TokenSequence(BaseModel):
    """Tokenized representation of a Python source file."""

    tokens: list[str]
    token_ids: list[int]
    source_file: Path


class SurprisalTrace(BaseModel):
    """Per-token surprisal values from a language model."""

    surprisal_values: list[float]
    token_count: int
    model_name: str


class FeatureVector(BaseModel):
    """16-dimensional feature vector extracted from a Python source file."""

    model_config = {"arbitrary_types_allowed": True}

    f1_mean_surprisal: float
    f2_surprisal_variance: float
    f3_sequence_perplexity: float
    f4_max_surprisal: float
    f5_low_surprisal_ratio: float
    f6_surprisal_slope: float
    f7_avg_identifier_length: float
    f8_identifier_diversity: float
    f9_whitespace_consistency: float
    f10_comment_to_code_ratio: float
    f11_ast_depth_mean: float
    f12_ast_node_type_diversity: float
    f13_shannon_entropy: float
    f14_zipf_coefficient_deviation: float
    f15_bigram_repetition_rate: float
    f16_hapax_legomena_ratio: float

    def to_array(self) -> npt.NDArray[np.float64]:
        """Return all 16 features as a numpy array in F1–F16 order."""
        return np.array([
            self.f1_mean_surprisal,
            self.f2_surprisal_variance,
            self.f3_sequence_perplexity,
            self.f4_max_surprisal,
            self.f5_low_surprisal_ratio,
            self.f6_surprisal_slope,
            self.f7_avg_identifier_length,
            self.f8_identifier_diversity,
            self.f9_whitespace_consistency,
            self.f10_comment_to_code_ratio,
            self.f11_ast_depth_mean,
            self.f12_ast_node_type_diversity,
            self.f13_shannon_entropy,
            self.f14_zipf_coefficient_deviation,
            self.f15_bigram_repetition_rate,
            self.f16_hapax_legomena_ratio,
        ], dtype=np.float64)

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return the 16 feature field names in F1–F16 order."""
        return [
            "f1_mean_surprisal",
            "f2_surprisal_variance",
            "f3_sequence_perplexity",
            "f4_max_surprisal",
            "f5_low_surprisal_ratio",
            "f6_surprisal_slope",
            "f7_avg_identifier_length",
            "f8_identifier_diversity",
            "f9_whitespace_consistency",
            "f10_comment_to_code_ratio",
            "f11_ast_depth_mean",
            "f12_ast_node_type_diversity",
            "f13_shannon_entropy",
            "f14_zipf_coefficient_deviation",
            "f15_bigram_repetition_rate",
            "f16_hapax_legomena_ratio",
        ]


class FileAnalysisResult(BaseModel):
    """Detection result for a single Python source file."""

    file_path: Path
    feature_vector: FeatureVector
    ai_confidence: float
    lines_of_code: int
    is_parseable: bool = True
    error_message: str | None = None


class CommitAnalysisResult(BaseModel):
    """Aggregated detection result for an entire commit."""

    commit_hash: str
    timestamp: datetime
    ai_confidence_pct: float
    file_results: list[FileAnalysisResult]
    total_files_analyzed: int
    total_files_skipped: int

    @classmethod
    def aggregate(cls, file_results: list[FileAnalysisResult]) -> float:
        """Compute LOC-weighted average ai_confidence as a percentage.

        Returns:
            ai_confidence_pct in [0, 100].

        Raises:
            ValueError: If file_results is empty.
        """
        if not file_results:
            raise ValueError("Cannot aggregate empty file results")
        total_loc = sum(r.lines_of_code for r in file_results)
        if total_loc == 0:
            raise ValueError("Total lines of code is zero")
        weighted_sum = sum(
            r.ai_confidence * r.lines_of_code for r in file_results
        )
        return (weighted_sum / total_loc) * 100
