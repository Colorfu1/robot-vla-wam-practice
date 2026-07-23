# Rerun Layout Persistence Design

**Date:** 2026-07-23
**Status:** Reviewed and implemented

## 1. Problem

The project embeds a programmatic Rerun blueprint in each generated recording. It is
currently sent with Rerun's default activation settings:

```python
rr.send_blueprint(blueprint)
```

In Rerun 0.26.2 this is equivalent to:

```python
rr.send_blueprint(
    blueprint,
    make_active=True,
    make_default=True,
)
```

The viewer does persist edited blueprints when launched with `--persist-state`. Evidence
is present under `~/.local/share/rerun/blueprints/`. However, opening one of the project's
recordings processes its embedded `make_active=True` activation command and replaces the
persisted active blueprint. The user therefore sees the generated default layout again.

## 2. Desired Behavior

Layouts are shared by visualization type through the existing Rerun application IDs:

- `robot_practice_act_libero` for LIBERO closed-loop evaluation;
- `robot_practice_act_offline` for offline ACT evaluation;
- `robot_practice_aloha_dataset` for ALOHA dataset visualization.

The behavior for each application ID is:

1. On first open, when no persisted layout exists, use the project's generated blueprint.
2. After a user edits the layout and closes the viewer, a later
   `rerun --persist-state ...` invocation restores that edited layout.
3. A recording must not force its embedded blueprint over an already persisted active
   blueprint.
4. The generated blueprint remains the default/reset target.
5. Layouts from different application IDs remain independent.

## 3. Chosen Design

Send every project-generated blueprint as a default template without forcing it active:

```python
rr.send_blueprint(
    blueprint,
    make_active=False,
    make_default=True,
)
```

Rerun documents that `make_default=True` can still activate the blueprint when no other
active blueprint exists. Therefore the first-open experience is preserved. When a
persisted blueprint has already been loaded for the application ID, `make_active=False`
prevents the embedded default from replacing it.

The change applies at the two blueprint emission points:

- the shared offline/LIBERO recording setup in
  `src/robot_practice/evaluation/rerun_visualization.py`;
- the ALOHA dataset recording setup in `src/robot_practice/visualize_aloha.py`.

Blueprint construction, application IDs, view contents, timelines, metrics, camera data,
and robot visualization remain unchanged.

## 4. Compatibility and Lifecycle

- `--persist-state` remains required when opening Rerun if layout persistence is desired.
- Persistence is keyed by application ID, not by individual `.rrd` path.
- Existing `.rrd` files already contain `make_active=True`; they cannot be changed by this
  source update and must be regenerated.
- New recordings retain an embedded default blueprint, so they remain usable on machines
  without any persisted state.
- Users can return to the project layout with Rerun's Reset Blueprint action because the
  embedded blueprint remains the default.

## 5. Failure Handling

No new runtime failure path is introduced. If persisted viewer state is absent, disabled,
invalid, or removed, Rerun falls back to the embedded default blueprint. Recording
generation must not depend on the presence of viewer state.

## 6. Verification

Automated tests will verify:

1. shared offline/LIBERO recording setup calls `rr.send_blueprint` with
   `make_active=False` and `make_default=True`;
2. ALOHA recording setup uses the same activation flags;
3. generated `.rrd` files pass `rerun rrd verify`;
4. printed RRD metadata contains a blueprint activation command with
   `make_active: false` and `make_default: true`;
5. existing blueprint ordering guarantees remain intact: the file sink is configured
   before the blueprint is sent.

Manual acceptance uses two newly generated recordings with the same application ID:

1. open the first with `rerun --persist-state`;
2. change the layout and close the viewer normally;
3. open the second with `rerun --persist-state`;
4. confirm the edited layout is restored;
5. reset the blueprint and confirm the generated project layout is restored.
