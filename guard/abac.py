"""Attribute-based access control (ABAC) for RAG retrieval.

Policies are data (a condition tree), and two backends are generated from the
same tree so they can never drift:

- ``evaluate(cond, subject, resource)`` — pure-Python verdict, used by the
  differential tests and any per-chunk re-check.
- ``compile_sql(cond, subject)`` — a SQLAlchemy clause over the typed
  ``Chunk.sensitivity`` / ``Chunk.doc_type`` columns, used inside the
  retrieval ``WHERE`` clause (pre-filter, deny-by-default).

Subject attributes are resolved server-side from the DB user row on every
request (never from JWT claims). Attributes with a ``subject.`` prefix fold
into SQL boolean literals at compile time; ``resource.`` attributes compile
to column comparisons, which stay dialect-free and indexable.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import ColumnElement, and_, false, or_, true

from guard.db import Chunk, User

POLICY_VERSION = "1"


@dataclass(frozen=True)
class Eq:
    attr: str
    value: str


@dataclass(frozen=True)
class In:
    attr: str
    values: tuple[str, ...]

    def __init__(self, attr: str, values: Iterable[str]):
        object.__setattr__(self, "attr", attr)
        object.__setattr__(self, "values", tuple(values))


@dataclass(frozen=True)
class And:
    conditions: tuple


@dataclass(frozen=True)
class Or:
    conditions: tuple


P1_PUBLIC_PERMIT = Eq("resource.sensitivity", "public")
P2_ADMIN_PERMIT = Eq("subject.role", "admin")

#: Retrieval policy: OR of permits; deny-by-default when nothing matches.
RETRIEVAL_POLICY = Or((P1_PUBLIC_PERMIT, P2_ADMIN_PERMIT))

_RESOURCE_COLUMNS = {
    "resource.sensitivity": Chunk.sensitivity,
    "resource.doc_type": Chunk.doc_type,
}


def _resolve(attr: str, subject: dict, resource: dict):
    scope, _, name = attr.partition(".")
    if scope == "subject":
        return subject.get(name)
    if scope == "resource":
        return resource.get(name)
    raise ValueError(f"unknown attribute scope in {attr!r}; expected 'subject.' or 'resource.'")


def evaluate(cond, subject: dict, resource: dict) -> bool:
    """Pure-Python verdict for one condition tree against subject/resource attrs."""
    if isinstance(cond, Eq):
        return _resolve(cond.attr, subject, resource) == cond.value
    if isinstance(cond, In):
        return _resolve(cond.attr, subject, resource) in cond.values
    if isinstance(cond, And):
        return all(evaluate(child, subject, resource) for child in cond.conditions)
    if isinstance(cond, Or):
        return any(evaluate(child, subject, resource) for child in cond.conditions)
    raise TypeError(f"unknown condition node: {cond!r}")


def compile_sql(cond, subject: dict) -> ColumnElement[bool]:
    """SQLAlchemy clause for the condition tree; ``subject`` folds into literals."""
    if isinstance(cond, Eq):
        column = _RESOURCE_COLUMNS.get(cond.attr)
        if column is not None:
            return column == cond.value
        return true() if evaluate(cond, subject, {}) else false()
    if isinstance(cond, In):
        column = _RESOURCE_COLUMNS.get(cond.attr)
        if column is not None:
            return column.in_(cond.values)
        return true() if evaluate(cond, subject, {}) else false()
    if isinstance(cond, And):
        return and_(*(compile_sql(child, subject) for child in cond.conditions))
    if isinstance(cond, Or):
        return or_(*(compile_sql(child, subject) for child in cond.conditions))
    raise TypeError(f"unknown condition node: {cond!r}")


def subject_attributes(user: User) -> dict:
    """Resolve ABAC subject attributes from the DB user row (role + attributes)."""
    return {"role": user.role, **(user.attributes or {})}


def resource_attributes(chunk: Chunk) -> dict:
    """Resolve ABAC resource attributes for one chunk row."""
    return {"sensitivity": chunk.sensitivity, "doc_type": chunk.doc_type}


def retrieval_predicate(subject: dict) -> ColumnElement[bool]:
    """Combined retrieval WHERE clause for the given subject; deny-by-default."""
    return compile_sql(RETRIEVAL_POLICY, subject)
