"""Research toolkit: how BrainPatches are *created*.

Everything here is for patch **authors** -- activation extraction, SAE training,
feature discovery, causal validation, patch search. It needs torch and is
installed by the ``research`` extra::

    pip install "brainpatch[research]"

**The runtime never imports this.** A user applying a patch needs none of it: no
SAE, no activation corpus, no training code. That separation is the product.

Where the compute runs is an author's choice. This repository's own experiments
used Modal (see ``modal_app/``) because the development machine deliberately
carries no ML stack, but nothing here requires it -- a local GPU, a cluster, or
a notebook works equally well.
"""
