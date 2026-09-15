# Architecture

The shared source repository lives at <JANGO_REPO> and contains only source, tests, scripts, configs, and documentation. The named Conda environment lives separately at <CONDA_ROOT>/envs/jango.

~~~text
jango/
+-- src/
|   +-- nbia/          # Nanobody Interface Atlas engine and preserved CLI
|   +-- jango/         # DELPHI/Jango orchestration and run layout
+-- scripts/
|   +-- nbia/          # operational NBIA-origin scripts
|   +-- jango/         # operational Jango scripts
+-- configs/           # environment and runtime examples
+-- tests/
+-- docs/
+-- pyproject.toml
+-- README.md
+-- CHANGELOG.md
+-- LICENSE
+-- .gitignore
~~~

src/jango/paths.py is the canonical run-layout implementation. Jango workflow commands derive generated paths from one explicit --output-root. NBIA command names and arguments are preserved, with a guard that rejects generated outputs inside <JANGO_REPO>.
