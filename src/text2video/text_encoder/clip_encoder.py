"""Frozen CLIP text encoder.

We use CLIP ViT-B/32's text tower to turn captions into 512-d embeddings, and keep it
**frozen** -- no gradients, never trained. Two reasons:

1. Training a language model is not what this project is about. Borrowing a pretrained
   encoder is standard practice and lets all the learning capacity go into the
   generative and temporal parts.
2. The same frozen model is reused at evaluation time to compute CLIPSIM (does the
   generated video match the caption?). One model, two roles.

Known limitation, stated up front: because CLIP both *conditions* the generators and
*scores* them, CLIPSIM is not a fully independent measure of alignment. That is exactly
why the structured-grounding metric exists -- it checks the generated motion against
ground truth without CLIP in the loop at all.

Also note CLIP was trained on natural images and web captions, not 64x64 grayscale
moving digits. Its embeddings still separate our captions usefully (verified by
`scripts/build_text_embeddings.py --verify`), but CLIPSIM values here are only
meaningful for *relative* comparison between our own models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# CLIP ViT-B/32 projects both towers into a shared 512-d space.
EMBED_DIM = 512

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class CLIPTextEncoder(nn.Module):
    """Wraps open_clip's text tower with a simple `encode(list[str]) -> (N, 512)` API.

    The module is put in eval mode and all parameters have `requires_grad=False`, so it
    contributes no optimizer state and no gradient memory.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | torch.device = "cpu",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        import open_clip

        self.device = torch.device(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Drop the image tower: we only need text here. The evaluation code loads its
        # own full CLIP model when it needs image embeddings for CLIPSIM.
        self.model = model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._use_truncated = self._check_truncation_is_exact()

    # -- fast path ----------------------------------------------------------

    def _encode_truncated(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode with padding tokens after EOT dropped.

        CLIP's tokenizer pads every caption to 77 tokens. Ours are ~20 tokens, so ~70%
        of the compute is spent on padding. That padding is provably irrelevant: the
        text transformer uses a *causal* attention mask (position i attends only to
        positions <= i) and the sentence embedding is read out at the EOT position, so
        nothing after EOT can influence the output.

        Truncating to the longest EOT index in the batch is therefore exact, not an
        approximation -- and `_check_truncation_is_exact` verifies that at construction
        time against the reference implementation rather than trusting the reasoning.
        """
        model = self.model
        # EOT is the highest token id in the vocabulary, so argmax locates it.
        max_len = int(tokens.argmax(dim=-1).max().item()) + 1
        tokens = tokens[:, :max_len]

        x = model.token_embedding(tokens)
        x = x + model.positional_embedding[:max_len].to(x.dtype)
        x = model.transformer(x, attn_mask=model.attn_mask[:max_len, :max_len])
        x = model.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
        return x @ model.text_projection

    @torch.no_grad()
    def _check_truncation_is_exact(self) -> bool:
        """Verify the fast path against `encode_text` before relying on it.

        open_clip's internals differ between versions (batch-first conventions, whether
        `text_projection` is a parameter or a module), so this is a runtime self-check
        rather than an assumption. On any mismatch or error we silently fall back to the
        reference implementation -- slower, but never wrong.
        """
        probes = [
            "The digit 3 moves to the right.",
            "Two digits, 4 and 7, drift slowly across the frame.",
        ]
        try:
            tokens = self.tokenizer(probes).to(self.device)
            reference = self.model.encode_text(tokens).float()
            fast = self._encode_truncated(tokens).float()
            return bool(torch.allclose(reference, fast, atol=1e-4, rtol=1e-3))
        except Exception:  # noqa: BLE001 - any failure just means "use the slow path"
            return False

    @torch.no_grad()
    def encode(self, captions: list[str], batch_size: int = 256) -> torch.Tensor:
        """Encode captions to (N, 512) float32 on CPU.

        L2-normalised when `normalize=True`, which makes cosine similarity a plain dot
        product and keeps the conditioning signal at a consistent scale -- otherwise
        embedding magnitude varies with caption length and acts as noise on the
        conditioning pathway.

        Duplicate captions are encoded once and shared. Our templated captions repeat
        (about 25% of the 20k training captions are duplicates), and CLIP is
        deterministic, so re-encoding them would be pure waste.
        """
        if not captions:
            return torch.zeros(0, EMBED_DIM)

        unique_captions: list[str] = []
        index_of: dict[str, int] = {}
        positions: list[int] = []
        for caption in captions:
            if caption not in index_of:
                index_of[caption] = len(unique_captions)
                unique_captions.append(caption)
            positions.append(index_of[caption])

        outputs: list[torch.Tensor] = []
        for start in range(0, len(unique_captions), batch_size):
            chunk = unique_captions[start : start + batch_size]
            tokens = self.tokenizer(chunk).to(self.device)
            if self._use_truncated:
                features = self._encode_truncated(tokens).float()
            else:
                features = self.model.encode_text(tokens).float()
            if self.normalize:
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            outputs.append(features.cpu())

        unique_embeddings = torch.cat(outputs, dim=0)
        return unique_embeddings[torch.tensor(positions, dtype=torch.long)]

    @torch.no_grad()
    def encode_numpy(self, captions: list[str], batch_size: int = 256) -> np.ndarray:
        """Same as `encode`, returning a float32 numpy array for on-disk caching."""
        return self.encode(captions, batch_size=batch_size).numpy().astype(np.float32)

    def forward(self, captions: list[str]) -> torch.Tensor:
        return self.encode(captions)


def cosine_similarity_matrix(embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
    """Pairwise cosine similarity, used to sanity-check that embeddings are meaningful."""
    tensor = torch.as_tensor(embeddings, dtype=torch.float32)
    tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (tensor @ tensor.T).numpy()
