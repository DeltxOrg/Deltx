"""Infrastructure settings, kept separate from research hyperparameters."""

from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict

from deltx.common.settings import PrefixedSettings


class SonarConfig(PrefixedSettings):
    """One local server URL, shared by the API, Compose and scanner."""

    model_config = SettingsConfigDict(
        env_prefix="SONAR_",
        env_file=".env",
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )
    host_url: str
    token: SecretStr = Field(default_factory=lambda: SecretStr(""))
    scanner_image: str = "sonarsource/sonar-scanner-cli:latest"
    expected_version: str | None = None
    compose_file: Path = (
        Path(__file__).resolve().parents[4] / "docker/sonarqube/compose.yaml"
    )
    request_timeout: float = Field(default=30, gt=0)
    startup_timeout: float = Field(default=300, gt=0)
    compute_timeout: float = Field(default=600, gt=0)
    scan_timeout: float = Field(default=900, gt=0)
    poll_interval: float = Field(default=2, gt=0)
    page_size: int = Field(default=500, ge=1, le=500)

    @model_validator(mode="after")
    def validate_local(self) -> "SonarConfig":
        parsed = urlsplit(self.host_url)
        # Accessing port validates malformed/out-of-range ports immediately.
        _ = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "SONAR_HOST_URL must be a local HTTP(S) URL without credentials"
            )
        image_name = self.scanner_image.rsplit("/", 1)[-1]
        tag = image_name.rsplit(":", 1)[-1] if ":" in image_name else ""
        if not tag or any(c.isspace() for c in self.scanner_image):
            raise ValueError("SonarScanner image must have a tag or digest")
        if self.expected_version is not None and not self.expected_version.strip():
            raise ValueError("SONAR_EXPECTED_VERSION cannot be empty")
        return self
