"""
FAST action tokenizer using pretrained weights from physical-intelligence/fast.
Reference: https://huggingface.co/physical-intelligence/fast
"""

import json
import hashlib
import math
from pathlib import Path
import numpy as np
from typing import Optional
from scipy.fft import dct, idct
from tokenizers import Tokenizer


# Pretrained FAST tokenizer config
# From processor_config.json of physical-intelligence/fast
DEFAULT_SCALE = 10.0
DEFAULT_MIN_TOKEN = -354  # discovered during BPE training of the pretrained model


class FASTTokenizer:
    """
    FAST action tokenizer: DCT + BPE compression for robot actions.
    Uses pretrained weights from physical-intelligence/fast.

    Actions are normalized to [-1, 1] before tokenization.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        scale: float = DEFAULT_SCALE,
        min_token: int = DEFAULT_MIN_TOKEN,
        action_dim: Optional[int] = None,
        time_horizon: int = 16,
    ):
        self.bpe_tokenizer = tokenizer
        self.scale = float(scale)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("FAST scale must be finite and positive")
        self.min_token = int(min_token)
        self.action_dim = action_dim
        self.time_horizon = time_horizon

    @classmethod
    def from_pretrained(cls, pretrained_path: str = "physical-intelligence/fast", **kwargs):
        """Load BPE and processor metadata together without executing remote code.

        FAST+ is the universal tokenizer distributed at physical-intelligence/fast
        (Apache-2.0). A local directory must contain tokenizer.json and
        processor_config.json; Hub loading supports a pinned revision.
        """
        # The retained dormant FAST head needs its original vocabulary when
        # constructing the complete checkpoint graph, without a network read.
        from prism_vla.modeling.release_assets import asset_path
        pretrained_path = str(asset_path("fast"))
        revision = kwargs.pop("revision", None)
        local_files_only = kwargs.pop("local_files_only", False)
        path = Path(pretrained_path).expanduser()
        if path.exists():
            directory = path if path.is_dir() else path.parent
            tokenizer_file = directory / "tokenizer.json"
            processor_file = directory / "processor_config.json"
            if not tokenizer_file.is_file() or not processor_file.is_file():
                raise FileNotFoundError(
                    f"FAST directory {directory} requires tokenizer.json and processor_config.json"
                )
        elif path.is_absolute() or pretrained_path.startswith(("./", "../", "~")):
            raise FileNotFoundError(f"FAST tokenizer path does not exist: {path}")
        else:
            from huggingface_hub import hf_hub_download
            tokenizer_file = Path(hf_hub_download(
                repo_id=pretrained_path, filename="tokenizer.json",
                revision=revision, local_files_only=local_files_only,
            ))
            processor_file = Path(hf_hub_download(
                repo_id=pretrained_path, filename="processor_config.json",
                revision=revision, local_files_only=local_files_only,
            ))
        config = json.loads(processor_file.read_text())
        for key in ("scale", "min_token"):
            if key not in config:
                raise ValueError(f"FAST processor_config.json is missing {key}")
        bpe_tokenizer = Tokenizer.from_file(str(tokenizer_file))
        declared_vocab = config.get("vocab_size")
        actual_vocab = bpe_tokenizer.get_vocab_size()
        if declared_vocab is not None and int(declared_vocab) != actual_vocab:
            raise ValueError(
                f"FAST processor vocabulary {declared_vocab} differs from BPE vocabulary {actual_vocab}"
            )
        scale = kwargs.pop("scale", config["scale"])
        min_token = kwargs.pop("min_token", config["min_token"])
        return cls(
            tokenizer=bpe_tokenizer,
            scale=scale,
            min_token=min_token,
            **kwargs,
        )

    @staticmethod
    def _load_bpe_tokenizer(pretrained_path: str) -> Tokenizer:
        """Compatibility helper; metadata is still validated by from_pretrained."""
        return FASTTokenizer.from_pretrained(pretrained_path).bpe_tokenizer

    def encode(
        self,
        action_chunk: np.ndarray,
        action_horizon: Optional[int] = None,
    ) -> list[int]:
        """
        Encode action chunk to token IDs.

        Args:
            action_chunk: Action array, shape (action_horizon, action_dim)
                         Values should be in [-1, 1]
            action_horizon: Optional horizon (if different from self.time_horizon)

        Returns:
            List of token IDs
        """
        action_chunk = np.asarray(action_chunk)
        action_horizon = self.time_horizon if action_horizon is None else int(action_horizon)
        if action_chunk.ndim != 2 or action_chunk.shape[0] != action_horizon:
            raise ValueError(f"FAST expected [{action_horizon}, A], got {action_chunk.shape}")
        token_str = self.encode_chunk_to_string(action_chunk)
        encoded = self.bpe_tokenizer.encode(token_str, add_special_tokens=False)
        tokens = encoded.ids
        if self.bpe_tokenizer.decode(tokens) != token_str:
            raise ValueError("FAST BPE vocabulary cannot represent all quantized coefficients losslessly")
        return tokens

    def decode(
        self,
        token_ids: list[int],
        action_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
        *,
        strict: bool = False,
    ) -> np.ndarray:
        """
        Decode token IDs back to action array.

        Args:
            token_ids: List of token IDs
            action_horizon: Number of timesteps
            action_dim: Dimension of action space

        Returns:
            Action array, shape (action_horizon, action_dim), values in [-1, 1]
        """
        action_horizon = self.time_horizon if action_horizon is None else int(action_horizon)
        action_dim = self.action_dim if action_dim is None else int(action_dim)
        if action_dim is None or action_horizon <= 0 or action_dim <= 0:
            raise ValueError("FAST decoding requires positive action_horizon and action_dim")

        # Decode BPE tokens to string
        decoded_str = self.bpe_tokenizer.decode(token_ids)

        # Convert characters back to DCT coefficients
        expected_len = action_horizon * action_dim
        try:
            decoded_dct = np.array(list(map(ord, decoded_str))) + self.min_token
            if len(decoded_dct) != expected_len:
                raise ValueError(
                    f"FAST decoded {len(decoded_dct)} coefficients, expected {expected_len}; "
                    "tokens were incomplete or malformed"
                )
            decoded_dct = decoded_dct.reshape(action_horizon, action_dim)
        except (ValueError, OverflowError):
            if strict:
                raise
            # Match the upstream fail-closed policy for malformed generated
            # chunks. Never truncate or zero-pad a partial trajectory.
            decoded_dct = np.zeros((action_horizon, action_dim))

        # Inverse DCT
        actions = idct(decoded_dct / self.scale, axis=0, norm="ortho")
        return actions

    def encode_chunk_to_string(self, action_chunk: np.ndarray) -> str:
        """Encode action chunk to a BPE-tokenized string (for computing token IDs)."""
        action_chunk = np.asarray(action_chunk)
        if action_chunk.ndim != 2 or 0 in action_chunk.shape:
            raise ValueError(f"FAST expected a nonempty 2D action chunk, got {action_chunk.shape}")
        if not np.isfinite(action_chunk).all():
            raise ValueError("FAST action labels must be finite")
        if np.max(np.abs(action_chunk)) > 1.0 + 1e-6:
            raise ValueError("FAST action labels must be normalized to [-1, 1]")
        dct_coeff = dct(action_chunk, axis=0, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.scale)
        chars = dct_coeff.flatten() - self.min_token
        if np.any(chars < 0) or np.any(chars > 0x10FFFF):
            raise ValueError("FAST coefficients exceed the tokenizer character range")
        token_str = "".join(map(chr, chars.astype(int)))
        return token_str

    def get_vocab_size(self) -> int:
        """Return the BPE vocabulary size."""
        return self.bpe_tokenizer.get_vocab_size()


class FASTActionTokenizer:
    """
    Higher-level wrapper that handles action normalization + FAST tokenization.
    Designed to match the interface expected by the VLA model.

    The EOS token (ID = bpe_vocab_size) is appended after the real FAST tokens
    during encoding. The total vocab_size = bpe_vocab_size + 1.
    """

    def __init__(
        self,
        fast_tokenizer: FASTTokenizer,
        action_dim: int,
        action_horizon: int = 16,
    ):
        self.fast = fast_tokenizer
        if action_dim is None or int(action_dim) <= 0 or int(action_horizon) <= 0:
            raise ValueError("FAST requires positive action_dim and action_horizon")
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.eos_token_id = self.fast.get_vocab_size()  # EOS = bpe_vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_path: str = "physical-intelligence/fast", **kwargs):
        """Load pretrained FAST tokenizer and create wrapper."""
        action_dim = kwargs.pop("action_dim", None)
        action_horizon = kwargs.pop("action_horizon", 16)
        fast = FASTTokenizer.from_pretrained(pretrained_path, **kwargs)
        return cls(fast_tokenizer=fast, action_dim=action_dim, action_horizon=action_horizon)

    def encode(self, actions: np.ndarray) -> list[int]:
        """
        Encode normalized actions [-1, 1] to token IDs (with EOS appended).
        Actions shape: (action_horizon, action_dim)
        """
        if np.asarray(actions).shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"FAST expected [{self.action_horizon}, {self.action_dim}], got {np.asarray(actions).shape}"
            )
        tokens = self.fast.encode(actions, action_horizon=self.action_horizon)
        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int], *, strict: bool = False) -> np.ndarray:
        """
        Decode token IDs to normalized actions [-1, 1].
        Strips EOS token and everything after it before BPE decoding.
        Returns shape: (action_horizon, action_dim)
        """
        # Strip EOS and everything after
        if self.eos_token_id in token_ids:
            token_ids = token_ids[:token_ids.index(self.eos_token_id)]
        if any(token < 0 or token >= self.eos_token_id for token in token_ids):
            raise ValueError("FAST action token IDs are outside the BPE vocabulary")
        return self.fast.decode(
            token_ids,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            strict=strict,
        )

    @property
    def vocab_size(self) -> int:
        return self.fast.get_vocab_size() + 1  # +1 for EOS token

    @property
    def provenance(self) -> dict:
        """Semantic tokenizer identity, independent of cache path/file formatting.

        The BPE model includes its vocabulary, merges and text transforms. Scale,
        offset and physical-time shape are equally necessary to interpret a
        token chunk. Source artifact hashes/revisions remain separate evidence.
        """
        bpe = json.dumps(
            json.loads(self.fast.bpe_tokenizer.to_str()),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        return {
            "schema": "prism_fast_dct_bpe_v1",
            "bpe_sha256": hashlib.sha256(bpe).hexdigest(),
            "scale": self.fast.scale,
            "min_token": self.fast.min_token,
            "action_dim": self.action_dim,
            "physical_time_horizon": self.action_horizon,
            "eos_token_id": self.eos_token_id,
        }

    @property
    def fingerprint(self) -> str:
        """Hash all token-to-action semantics for checkpoint compatibility."""
        serialized = json.dumps(self.provenance, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def action_token_length(self) -> int:
        """Number of tokens per action chunk (varies due to BPE)."""
        # For a typical action chunk, BPE compression gives variable length
        # We compute this by encoding a sample action
        sample_actions = np.zeros((self.action_horizon, self.action_dim))
        tokens = self.encode(sample_actions)
        return len(tokens)
