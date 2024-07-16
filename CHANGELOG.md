# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Types of Changes:
- **Added**: for new features.
- **Changed**: for changes in existing functionality.
- **Deprecated**: for soon-to-be removed features.
- **Removed**: for now removed features.
- **Fixed**: for any bug fixes.
- **Security**: in case of vulnerabilities.

## [Unreleased]

## [v0.16.5] - 2024-07-16

## ADDED

- Support for `nullable` attributes in OpenAPI schemas [#11](https://github.com/dantegates/porter/pull/11)

## [v0.16.5] - 2024-07-12

## Fixed

- Register feature schema when `feature_columns` are explicitly passed to `PredictionService` [#10](https://github.com/dantegates/porter/pull/10)

## [v0.16.4] - 2024-07-12

### Added
- Provisional hook for formatting the response data from `PredictionService` [#9](https://github.com/dantegates/porter/pull/9)

## [v0.16.3] - 2024-07-11

### Added
- `feature_columns` can be passed explicitly to `PredictionService` or can be (optionally) inferred from `feature_schema` [#8](https://github.com/dantegates/porter/pull/8)

### Changed
- `keras` support now comes from [`keras` 3](https://keras.io/getting_started/#installing-keras-3), which supports multiple backends.

### Fixed
- Failing unit tests [#7](https://github.com/dantegates/porter/pull/7)

## [v0.16.2] - 2023-11-06

### Fixed
- Package build was not including `porter.schemas` [#6](https://github.com/dantegates/porter/pull/6)

## [v0.16.0] - 2023-10-25

### Changed
- Migrated fully from `setup.py` to `pyproject.toml` [#5](https://github.com/dantegates/porter/pull/5)
- Use name `porter-schmorter` for PyPI [#5](https://github.com/dantegates/porter/pull/5)
- Removed explicit references to `Python 3.6` [#5](https://github.com/dantegates/porter/pull/5)

### Fixed
- Sphinx build: use `add_stylesheet` -> `add_css_file` [#5](https://github.com/dantegates/porter/pull/5)

### Changed
- `porter`'s dependencies have been upgraded to `Flask>=3.0.0`
- Removed references to `CadentTech` and updated `LICENSE`

### Added
- GitHub actions for unit tests
