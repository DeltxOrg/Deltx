"""Custom exception hierarchy for Deltx."""


class DeltxError(Exception):
    """Base exception for all Deltx errors."""


class ParsingError(DeltxError):
    """AST or tokenization failure."""


class FeatureExtractionError(DeltxError):
    """Feature computation failure."""


class ModelNotLoadedError(DeltxError):
    """Language model or classifier not initialized."""


class DatasetError(DeltxError):
    """Dataset download or processing failure."""


class ClassifierError(DeltxError):
    """Classifier training, evaluation, or persistence failure."""
