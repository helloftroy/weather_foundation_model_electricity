"""Inspect a loaded Prithvi-WxC model to identify candidate embedding tensors.

Not part of the pretraining/inference pipeline itself -- this is a one-off
introspection script for Phase 1 step 2 (documenting where an embedding could
be extracted later). Prints module structure and, if given a sample batch,
the shape of every named submodule's output via forward hooks.

Run only after src/prithvi/setup_prithvi_env.sh has installed Prithvi-WxC and
weights have been downloaded. Fill in the model construction call below once
the repo's actual model-loading API is confirmed (see the repo's own
basic-inference notebook for the canonical way to instantiate the model from
the downloaded config + checkpoint).
"""
import argparse

import torch


def register_shape_hooks(model: torch.nn.Module, max_depth: int = 3) -> dict:
    shapes = {}

    def make_hook(name):
        def hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                shapes[name] = tuple(out.shape)
            elif isinstance(out, (tuple, list)):
                shapes[name] = [tuple(o.shape) for o in out if isinstance(o, torch.Tensor)]
        return hook

    for name, module in model.named_modules():
        depth = name.count(".")
        if 0 < depth <= max_depth:
            module.register_forward_hook(make_hook(name))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Downloaded prithvi.wxc.2300m.v1 snapshot dir")
    parser.add_argument("--print-modules", action="store_true", help="Print full named_modules() tree")
    args = parser.parse_args()

    # TODO: replace with the repo's actual model-loading call once confirmed
    # from examples/PrithviWxC_inference.ipynb, e.g. something like:
    #   from PrithviWxC.model import PrithviWxC
    #   model = PrithviWxC.from_pretrained(args.checkpoint_dir)
    raise NotImplementedError(
        "Fill in model construction using the repo's own basic-inference "
        "notebook (examples/PrithviWxC_inference.ipynb) as reference, then "
        "run this against a real (or dummy) input batch and record the "
        "resulting shapes in docs/prithvi_embedding_candidates.md."
    )


if __name__ == "__main__":
    main()
