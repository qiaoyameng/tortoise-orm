.. _identity_scope:

==============
Identity Scope
==============

By default every Tortoise ORM query materializes brand new model instances,
so two queries for the same row return two different Python objects.
``identity_scope()`` activates an explicit, opt-in **identity map**: while a
scope is active, all rows materialized from the database — regular queries,
``.only()`` partial queries, ``select_related`` and ``prefetch_related`` —
are keyed by ``(connection name, model, primary key)`` and the *same* Python
object is returned for the same database row.

This is useful per logical request (e.g. per HTTP request or per task), where
repeatedly querying the same row should yield one shared object.

Usage
=====

.. code-block:: python3

    from tortoise import identity_scope

    async with identity_scope():
        a = await Tournament.get(id=1)
        b = await Tournament.get(id=1)
        assert a is b

        event = await Event.get(name="Final").select_related("tournament")
        assert event.tournament is a

The scope works as both a sync and an async context manager, and can be
nested: an inner scope sees everything registered in outer scopes, while
instances first loaded in the inner scope are discarded when it exits.

The active scope is stored in a :class:`contextvars.ContextVar`, so concurrent
async tasks never share a scope unless one is explicitly propagated. Instances
loaded for one tenant/request are therefore never reused by another.

Without an active scope nothing changes: instance construction behaves
exactly as before.

Merge rules
===========

When a query returns a row that is already present in the scope, the new row
is merged into the existing instance instead of creating a new one:

- Fields that have not been loaded yet (e.g. after an ``.only()`` query) are
  filled in.
- Fields whose local value still matches the value loaded from the database
  may be refreshed with newer database values.
- Fields that were **modified locally** are never overwritten.

Rows fetched without their primary key (e.g. ``.only()`` that does not
include the pk) cannot be keyed and are not tracked.

Save, delete, refresh and transactions
======================================

- ``save()`` registers the instance in the active scope (if not present) and
  resets the modification baseline: values are considered unmodified again
  after a successful save.
- ``delete()`` on an instance evicts it from all scopes in the active chain.
  Bulk ``QuerySet.delete()`` does **not** evict (the affected primary keys are
  unknown without an extra query); call ``IdentityScope.clear()`` or re-fetch
  after a bulk delete.
- ``refresh_from_db()`` always bypasses the identity map for its own query,
  forces the fresh database values onto the instance (discarding local
  modifications for the refreshed fields) and resets the baseline.
- The scope is **not transaction aware**. After a transaction rollback,
  cached instances may reflect state that was rolled back — exit the scope or
  call ``IdentityScope.clear()`` after a rollback.

API
===

.. autoclass:: tortoise.identity.IdentityScope
   :members:

.. autofunction:: tortoise.identity.identity_scope

.. autofunction:: tortoise.identity.current_scope
