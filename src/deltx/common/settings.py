"""Keep independent settings namespaces compatible with a shared .env file."""

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from pydantic_settings.sources import DotEnvSettingsSource


class PrefixedSettings(BaseSettings):
    """Read only this settings model's prefix from dotenv; still reject typos."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        if isinstance(dotenv_settings, DotEnvSettingsSource):
            prefix = dotenv_settings.env_prefix
            if not dotenv_settings.case_sensitive:
                prefix = prefix.lower()
            dotenv_settings.env_vars = {
                key: value
                for key, value in dotenv_settings.env_vars.items()
                if key.startswith(prefix)
            }
        return init_settings, env_settings, dotenv_settings, file_secret_settings
