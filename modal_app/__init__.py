"""Modal orchestration for BrainPatch.

Every module in this package is imported *locally* by the `modal` CLI in order
to build the App graph. That means **no module here may import torch,
transformers or datasets at module scope** -- the local machine deliberately
does not run the ML stack. Heavy imports go inside function bodies, where they
execute only in the remote container.

Entry points are re-exported from :mod:`modal_app.app`, so every workflow is
reachable as::

    modal run modal_app/app.py::<function>
"""
