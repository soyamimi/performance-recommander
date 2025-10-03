from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=["conf/global.toml", "conf/.secrets/secrets.toml", ".env"],
    load_dotenv=True,
    environment=True,
    envvar_prefix="PERFORMANCE_RECOMMANDER",
    env_switcher="PERFORMANCE_RECOMMANDER_ENV",
)
