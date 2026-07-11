"""End-to-end integration tests for the AI detection module.

Exercises the full Stage 2 flow — synthetic source → FeatureExtractionPipeline
→ DetectionClassifier → AIDetectionInference — with a mocked perplexity family
so no language model is downloaded. The stylometric (F7–F12) and distribution
(F13–F16) families run for real over the synthetic samples; only F1–F6 are
substituted with deterministic values.

The mock is intentionally class-correlated: samples containing a docstring (the
AI-style corpus) receive low surprisal values, docstring-free samples (the
human-style corpus) high ones. That mirrors the real signal — LLM output is
low-surprisal — and gives the classifier something learnable, so the round trip
through training and inference is meaningful rather than noise-on-noise.

All tests here are marked ``integration``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from deltx.common.config import DeltxConfig
from deltx.detection.classifier import DetectionClassifier
from deltx.detection.inference import AIDetectionInference
from deltx.detection.models import FeatureVector, FileAnalysisResult
from deltx.detection.pipeline import FeatureExtractionPipeline

pytestmark = pytest.mark.integration

_TS = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)

# -- synthetic corpus ----------------------------------------------------------
# Human-style: short names, inconsistent formatting, inline comments, varied
# structure, no docstrings.

HUMAN_SAMPLES: list[str] = [
    """def f(x, y):
    # quick add
    r = x + y
    return r

z = f(1,2)
print_val = z * 3
""",
    """import os

def chk(p):
    if os.path.exists(p): return True
    return False

# main-ish
res = chk('/tmp')
""",
    """cnt = 0
for i in range(10):
    if i%2==0:
        cnt+=1  # evens
    else:
        pass
tot = cnt
""",
    """def g(a,b=None):
    if b is None: b = []
    b.append(a)
    return b

x1 = g(1)
x2 = g(2, x1)
""",
    """d = {}
ks = ['a','b','c']
for k in ks:
    d[k] = len(k)+1  # why not
v = d.get('a', 0)
""",
    """class T:
    def __init__(s, n):
        s.n = n
    def dbl(s):
        return s.n*2

t = T(5)
o = t.dbl()
""",
    """import sys

def m():
    args = sys.argv[1:]
    if not args:
        return 1
    # print each
    for a in args: pass
    return 0
""",
    """s = 'hello'
r = ''
for ch in s:
    r = ch + r
# reversed now
n = len(r)
""",
    """def fib(n):
    a,b = 0,1
    out=[]
    while a<n:
        out.append(a)
        a,b = b,a+b
    return out
""",
    """nums = [3,1,4,1,5]
mx = nums[0]
for x in nums[1:]:
    if x>mx: mx=x  # track max
mn = min(nums)
""",
]

# AI-style: descriptive names, consistent formatting, docstrings, uniform
# structure.

AI_SAMPLES: list[str] = [
    '''def calculate_sum(first_number: int, second_number: int) -> int:
    """Calculate the sum of two integers.

    Args:
        first_number: The first operand.
        second_number: The second operand.

    Returns:
        The sum of the two operands.
    """
    result = first_number + second_number
    return result
''',
    '''def check_path_exists(file_path: str) -> bool:
    """Check whether the given file path exists on disk.

    Args:
        file_path: The path to check.

    Returns:
        True if the path exists, False otherwise.
    """
    import os

    return os.path.exists(file_path)
''',
    '''def count_even_numbers(upper_bound: int) -> int:
    """Count even numbers in the range [0, upper_bound).

    Args:
        upper_bound: The exclusive upper bound of the range.

    Returns:
        The number of even integers in the range.
    """
    even_count = 0
    for current_value in range(upper_bound):
        if current_value % 2 == 0:
            even_count += 1
    return even_count
''',
    '''def append_to_list(item: int, target_list: list | None = None) -> list:
    """Append an item to a list, creating the list if necessary.

    Args:
        item: The item to append.
        target_list: The list to append to, or None to create a new one.

    Returns:
        The list containing the appended item.
    """
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list
''',
    '''def build_length_mapping(keys: list[str]) -> dict[str, int]:
    """Build a mapping from each key to its character length.

    Args:
        keys: The keys to measure.

    Returns:
        A dictionary mapping each key to its length.
    """
    length_mapping: dict[str, int] = {}
    for current_key in keys:
        length_mapping[current_key] = len(current_key)
    return length_mapping
''',
    '''class ValueDoubler:
    """A helper class that doubles a stored numeric value."""

    def __init__(self, initial_value: int) -> None:
        """Initialize the doubler with an initial value.

        Args:
            initial_value: The value to store.
        """
        self.stored_value = initial_value

    def double(self) -> int:
        """Return twice the stored value."""
        return self.stored_value * 2
''',
    '''def process_arguments(argument_list: list[str]) -> int:
    """Process command-line arguments and return an exit code.

    Args:
        argument_list: The command-line arguments to process.

    Returns:
        Zero on success, one when no arguments were provided.
    """
    if not argument_list:
        return 1
    for current_argument in argument_list:
        _ = current_argument
    return 0
''',
    '''def reverse_string(input_string: str) -> str:
    """Reverse the characters of the input string.

    Args:
        input_string: The string to reverse.

    Returns:
        The reversed string.
    """
    reversed_result = ""
    for character in input_string:
        reversed_result = character + reversed_result
    return reversed_result
''',
    '''def generate_fibonacci_sequence(limit: int) -> list[int]:
    """Generate Fibonacci numbers strictly below the given limit.

    Args:
        limit: The exclusive upper bound of the sequence.

    Returns:
        A list of Fibonacci numbers below the limit.
    """
    previous_value, current_value = 0, 1
    sequence: list[int] = []
    while previous_value < limit:
        sequence.append(previous_value)
        previous_value, current_value = (
            current_value,
            previous_value + current_value,
        )
    return sequence
''',
    '''def find_maximum_value(numbers: list[int]) -> int:
    """Find the maximum value in a non-empty list of integers.

    Args:
        numbers: The list of integers to search.

    Returns:
        The largest integer in the list.
    """
    maximum_value = numbers[0]
    for current_number in numbers[1:]:
        if current_number > maximum_value:
            maximum_value = current_number
    return maximum_value
''',
]


class MockPerplexityExtractor:
    """Deterministic stand-in for the language-model-backed F1–F6 family.

    Docstring-bearing (AI-style) sources score low surprisal, docstring-free
    (human-style) sources high — the direction of the real signal — with a
    small length-derived wobble so no two files are byte-identical in
    feature space.
    """

    def __call__(self, source_code: str) -> dict[str, float]:
        """Return the six perplexity features for ``source_code``."""
        wobble = (len(source_code) % 17) / 100.0
        if '"""' in source_code:  # AI-style
            base = 1.5 + wobble
        else:  # human-style
            base = 5.5 + wobble
        return {
            "f1_mean_surprisal": base,
            "f2_surprisal_variance": base / 2.0,
            "f3_sequence_perplexity": 2.0**base,
            "f4_max_surprisal": base * 2.5,
            "f5_low_surprisal_ratio": max(0.0, 1.0 - base / 8.0),
            "f6_surprisal_slope": -0.01 * base,
        }


@pytest.fixture
def mocked_pipeline(config: DeltxConfig) -> FeatureExtractionPipeline:
    """A real pipeline whose perplexity family is replaced by the mock."""
    pipeline = FeatureExtractionPipeline(config)
    pipeline.perplexity = MockPerplexityExtractor()  # type: ignore[assignment]
    return pipeline


def _extract_corpus(
    pipeline: FeatureExtractionPipeline,
) -> tuple[list[FileAnalysisResult], list[int]]:
    """Run all 20 synthetic samples through the pipeline, with labels."""
    results: list[FileAnalysisResult] = []
    labels: list[int] = []
    for index, source in enumerate(HUMAN_SAMPLES):
        results.append(
            pipeline.extract_file_features(source, Path(f"human_{index}.py"))
        )
        labels.append(0)
    for index, source in enumerate(AI_SAMPLES):
        results.append(pipeline.extract_file_features(source, Path(f"ai_{index}.py")))
        labels.append(1)
    return results, labels


def _train_classifier(
    config: DeltxConfig,
    results: list[FileAnalysisResult],
    labels: list[int],
) -> tuple[DetectionClassifier, float]:
    """Train on 16 samples (8 per class), return the model and test accuracy.

    Sample order is human[0..9] then ai[0..9]; the last two of each class are
    held out, giving a balanced 16/4 split.
    """
    matrix = np.vstack([r.feature_vector.to_array() for r in results])
    y = np.asarray(labels, dtype=int)
    train_idx = list(range(0, 8)) + list(range(10, 18))
    test_idx = [8, 9, 18, 19]

    classifier = DetectionClassifier(config)
    classifier.train(matrix[train_idx], y[train_idx], tune_hyperparameters=False)
    predictions = classifier.predict(matrix[test_idx])
    accuracy = float((predictions == y[test_idx]).mean())
    return classifier, accuracy


def test_corpus_extraction_populates_all_16_features(
    mocked_pipeline: FeatureExtractionPipeline,
) -> None:
    """Every synthetic sample parses and yields a fully populated 16-D vector."""
    results, _ = _extract_corpus(mocked_pipeline)

    assert len(results) == 20
    for result in results:
        assert result.is_parseable, f"{result.file_path} failed to parse"
        assert result.error_message is None
        vector = result.feature_vector.to_array()
        assert vector.shape == (16,)
        assert np.all(np.isfinite(vector))
        # The three families each produced signal, not a zeroed fallback.
        assert np.any(vector[0:6] != 0.0)  # perplexity (mocked)
        assert np.any(vector[6:12] != 0.0)  # stylometric (real)
        assert np.any(vector[12:16] != 0.0)  # distribution (real)


def test_full_pipeline_end_to_end(
    config: DeltxConfig, mocked_pipeline: FeatureExtractionPipeline
) -> None:
    """Extract → train (16) → test (4) → analyze a 3-file commit."""
    results, labels = _extract_corpus(mocked_pipeline)
    classifier, accuracy = _train_classifier(config, results, labels)

    # The mocked perplexity signal is cleanly separable; the held-out four
    # samples should all classify correctly.
    assert classifier.is_fitted
    assert accuracy == pytest.approx(1.0)

    detector = AIDetectionInference(mocked_pipeline, classifier)
    commit_files = {
        Path("src/alpha.py"): AI_SAMPLES[0],
        Path("src/beta.py"): HUMAN_SAMPLES[0],
        Path("src/gamma.py"): AI_SAMPLES[1],
    }
    result = detector.analyze_commit(commit_files, "feedc0de1234", _TS, author="tester")

    # e. valid ai_confidence_pct in [0, 100]
    assert 0.0 <= result.ai_confidence_pct <= 100.0
    assert result.total_files_analyzed == 3
    assert result.total_files_skipped == 0
    assert result.author == "tester"

    # f. all 16 features populated for each file
    for file_result in result.file_results:
        vector = file_result.feature_vector.to_array()
        assert vector.shape == (16,)
        assert np.all(np.isfinite(vector))
        assert file_result.is_parseable
        assert 0.0 <= file_result.ai_confidence <= 1.0

    # g. LOC-weighted aggregation matches a hand computation over the results
    total_loc = sum(r.lines_of_code for r in result.file_results)
    assert total_loc > 0
    expected_pct = (
        sum(r.ai_confidence * r.lines_of_code for r in result.file_results)
        / total_loc
        * 100.0
    )
    assert result.ai_confidence_pct == pytest.approx(expected_pct)


def test_commit_with_mixed_authorship_scores_between_extremes(
    config: DeltxConfig, mocked_pipeline: FeatureExtractionPipeline
) -> None:
    """A pure-AI commit scores above a pure-human commit; both are in range."""
    results, labels = _extract_corpus(mocked_pipeline)
    classifier, _ = _train_classifier(config, results, labels)
    detector = AIDetectionInference(mocked_pipeline, classifier)

    ai_commit = detector.analyze_commit(
        {Path("a.py"): AI_SAMPLES[2], Path("b.py"): AI_SAMPLES[3]}, "aa11", _TS
    )
    human_commit = detector.analyze_commit(
        {Path("c.py"): HUMAN_SAMPLES[2], Path("d.py"): HUMAN_SAMPLES[3]}, "bb22", _TS
    )

    assert 0.0 <= human_commit.ai_confidence_pct <= 100.0
    assert 0.0 <= ai_commit.ai_confidence_pct <= 100.0
    assert ai_commit.ai_confidence_pct > human_commit.ai_confidence_pct


def test_trained_model_round_trips_through_from_config(
    config: DeltxConfig,
    mocked_pipeline: FeatureExtractionPipeline,
    tmp_path: Path,
) -> None:
    """Save the trained classifier, reload via from_config, and re-analyze."""
    results, labels = _extract_corpus(mocked_pipeline)
    classifier, _ = _train_classifier(config, results, labels)

    config.classifier_path = tmp_path / "detector.joblib"
    classifier.save(config.classifier_path)

    detector = AIDetectionInference.from_config(config)
    detector.pipeline = mocked_pipeline  # keep the LM mocked

    result = detector.analyze_file(AI_SAMPLES[4], Path("reloaded.py"))
    assert result.is_parseable
    assert 0.0 <= result.ai_confidence <= 1.0
    assert set(result.feature_vector.model_dump()) == set(
        FeatureVector.feature_names()
    )
