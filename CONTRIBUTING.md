# Contributing to AION Analytics

Thank you for your interest in contributing to AION Analytics! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/Aion_Analytics.git
cd Aion_Analytics
```

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Development Environment

### 1. Set Up Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 4. Create Required Directories

```bash
mkdir -p da_brains/intraday ml_data ml_data_dt data logs
```

## Code Style

### Python Style Guide

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 100)
- Use type hints for function signatures
- Write docstrings for public functions and classes

### Formatting with Black

```bash
# Format all code
black backend/ dt_backend/ tests/

# Check formatting without changes
black --check backend/ dt_backend/
```

### Linting with Ruff

```bash
# Lint all code
ruff check backend/ dt_backend/ tests/

# Auto-fix issues
ruff check --fix backend/ dt_backend/
```

### Type Checking (Optional)

```bash
mypy dt_backend/ backend/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_emergency_stop.py

# Run with coverage
pytest tests/ --cov=dt_backend --cov=backend --cov-report=html

# Run only unit tests
pytest tests/unit/ -m unit

# Run tests verbosely
pytest tests/ -v
```

### Writing Tests

- Place tests in `tests/unit/` or `tests/integration/`
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `tests/conftest.py`
- Aim for >75% code coverage for new code

Example test:

```python
def test_my_feature(mock_env):
    """Test description."""
    from module import function
    
    result = function()
    
    assert result == expected_value
```

## Pull Request Process

### 1. Make Your Changes

- Write clear, focused commits
- Add tests for new features
- Update documentation if needed
- Ensure all tests pass
- Run linters and formatters

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your branch
- Fill in the PR template:
  - **Title**: Brief description of changes
  - **Description**: Detailed explanation
  - **Related Issues**: Link any related issues
  - **Testing**: Describe how you tested
  - **Checklist**: Complete the checklist

### 4. Review Process

- Address review comments promptly
- Make requested changes in new commits
- Re-request review after changes
- Squash commits before merge (if requested)

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated (if applicable)
- [ ] No breaking changes (or documented)
- [ ] Commit messages follow guidelines
- [ ] PR description is clear and complete

## Commit Message Guidelines

### Format

```
type(scope): brief description

Detailed explanation of changes (optional)

Fixes #issue-number
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Build process or auxiliary tool changes

### Examples

```
feat(risk): add weekly drawdown protection

Implement weekly drawdown cap at 8% to prevent excessive losses
over a week-long period. Tracks week_peak_equity and compares
against current equity.

Fixes #123
```

```
fix(emergency-stop): handle missing stop file gracefully

Emergency stop check now returns False if file doesn't exist
instead of raising an exception.
```

## Project-Specific Guidelines

### Trading Safety

- ALWAYS test with `DT_DRY_RUN=1` first
- Never commit API keys or secrets
- Document any risk-related changes thoroughly
- Add validation for numerical parameters

### Risk Management Code

- Changes to risk rails require extra scrutiny
- Must include tests demonstrating safety
- Document assumptions and edge cases
- Consider failure modes

### Performance

- Avoid blocking operations in trading loops
- Use caching where appropriate
- Profile code for performance-critical paths
- Document any performance implications

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@example.com (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the project's license.

---

Thank you for contributing to AION Analytics! 🚀
