# Compatibility

New code should import jango for DELPHI/Jango orchestration and nbia for Nanobody Interface Atlas scientific primitives.

The nbia package and nbia CLI keep their command names and arguments. Direct NBIA commands refuse generated output paths inside <JANGO_REPO>; pass explicit paths under a user run root instead.

The historical DELPHI compatibility shim has been removed from the canonical pre-testing repository. Historical imports are preserved only in Git history and local snapshots.
