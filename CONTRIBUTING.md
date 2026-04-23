# Contributing

Contributions are welcome! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Lizatol/access_lib.git
cd access_lib
pip install -e ".[full,dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Type hints for function signatures
- Docstrings for public functions (NumPy style)

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to your fork: `git push origin feature/my-feature`
5. Open a Pull Request

## Reporting Issues

Please use [GitHub Issues](https://github.com/Lizatol/access_lib/issues) and include:
- Python version and OS
- Minimal reproducible example
- Expected vs actual behaviour
