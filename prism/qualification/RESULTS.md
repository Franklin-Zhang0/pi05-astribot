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

**Full-model H100 GPU inference parity passes.** The standalone and original
graphs ran in separate CUDA processes with the same checkpoint-2000 weights,
inputs and explicit noise. Across 19 synthetic cases/94 observations, all **995
captured arrays matched with maximum absolute difference 0.0**, at declared
`atol=rtol=1e-5`. Both graphs also passed 326 total reset-replay checks and
30 interleaved-client checks each, including different routes/prompts and
history across the eight-observation window. The cases cover all four routes,
plain/JSON text, head-only/both/asymmetric wrists and missing hand feedback.

See [the complete sanitized GPU receipt](gpu_parity.json), which records the
tested runtime commit, training-source revision, weight/fixture identities,
per-array differences, reset checks, and interleaved-session results.
The test used an H100 80 GB, Torch 2.11.0+cu128, Transformers 5.4.0, Triton 3.3.1,
FLA 0.5.1, causal-conv1d 1.6.2.post1, NumPy 2.2.6 and OpenCV 4.11.0.
The temporary validation job completed successfully after about 11.25 minutes;
existing training allocations were not interrupted.

The GPU comparison isolates model-graph portability using the same serving
adapter. The separate CPU checks above establish normalization, physical action
decoding and preprocessing agreement with the private reference. This is runtime
qualification on a synthetic fixture, not closed-loop robot policy evaluation.
