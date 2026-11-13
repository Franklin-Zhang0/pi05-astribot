# Validation status

The standalone implementation passes 64 CPU checks covering physical
normalization/decoding, routes and availability, prompt bytes, recurrent session
ownership, error retirement, WebSocket compatibility, portable bundle integrity,
strict mixed-dtype restoration on a fixture, and qualification comparison gates.
Ruff, Python source parsing, lockfile validation and wheel packaging pass. The
wheel contains the selected inference runtime, provenance and license assets;
it contains no Trainer or training entrypoints.

A real checkpoint-2000 bundle has been exported with its complete 13.42 GB model
payload and local Qwen/C-RADIO/FAST assets. Its weight SHA-256 is
`28aad7f18088e30c8f403ff896432e556b5e1d1ccd481b59da121595fd35e96a`.

The **complete real checkpoint restores offline on CPU**: 1,707 state tensors,
strict names/shapes, 1,162 FP32 tensors, 544 BF16 tensors and one integer tensor.
The model has 4,986,981,802 unique parameters, matching training, with the Qwen
embedding/output weight identity preserved and no remaining meta tensors.
Memory-mapped loading with empty parameter construction takes about 87 seconds
and peaks at 2.91 GiB resident memory in the development container. It does not
globally round FP32 weights. See [the CPU receipt](full_checkpoint_cpu.json).
The same complete restore also passed from a **clean Git archive**, with no
imports from the private repository; see [the clean-checkout receipt](clean_checkout_cpu.json).

Pixel preprocessing matches the private training-loader/deterministic-evaluation
reference **bit for bit** for six tested geometry/resolution combinations; see
[the pixel receipt](preprocessing_reference.json). Separate adapter comparisons
matched state/action normalization bitwise and canonical JSON byte-for-byte
across seven route/hand-availability cases.

**Full-model GPU inference parity is pending an available GPU.** The temporary
validation allocation was rejected by the provider's resource quota; no instance
was allocated and no existing training was interrupted. No GPU parity
pass or policy-quality result is claimed here. The GPU fixture covers 19
cases/94 observations, reset replay, history across the eight-observation window,
and interleaved sessions with different routes/prompts. Final qualification
receipts will identify their exact checkpoint, fixture, compute environment and
numerical tolerances. Existing training allocations were not interrupted.
