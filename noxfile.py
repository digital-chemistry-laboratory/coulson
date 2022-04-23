"""Automated testing linting and formatting apparatus."""
# external
import nox
from nox.sessions import Session

package = "coulson"
nox.options.sessions = "lint", "tests", "mypy"  # default session
locations = "coulson", "tests", "noxfile.py"  # Linting locations
pyversions = ["3.8", "3.9", "3.10"]


# Testing
@nox.session(venv_backend="mamba", python=pyversions)
def tests(session: Session) -> None:
    """Run tests."""
    args = session.posargs or ["--cov", "--import-mode=importlib", "-s"]
    session.conda_install("pytest", "pytest-cov")
    session.conda_install("networkx", "numpy", "rdkit", "scipy")
    session.install(".", "--no-deps")
    session.run("pytest", *args)


# Linting
@nox.session(python="3.9")
def lint(session: Session) -> None:
    """Lint code."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
        "flake8-annotations",
        "flake8-docstrings",
        "darglint",
    )
    session.run("flake8", *args)


# Code formatting
@nox.session(python="3.9")
def black(session: Session) -> None:
    """Format code."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


# Static typing
@nox.session(venv_backend="mamba", python="3.9")
def mypy(session: Session) -> None:
    """Run the static type checker."""
    args = session.posargs or locations
    session.conda_install("mypy")
    session.conda_install("networkx", "numpy", "rdkit", "scipy")
    session.run("mypy", *args)
