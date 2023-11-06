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
