# Inference runtime provenance

`src/prism_vla` contains the selected model construction, perception, action
geometry, recurrent cache and inference dependencies of the trained PRISM
snapshot. It contains no Trainer, dataset reader, training entrypoint, launch
script, credentials, private repository URL, or Git history.

`runtime_source_manifest.json` records the source revision and SHA-256 of every
selected source file and its released counterpart. The constructor, recurrent
inference, attention, action heads and perception operations retain the trained
implementation. The release adaptations are:

- Training forward entrypoints and trainer checkpoint serialization are omitted.
- Qwen and C-RADIO architectures are constructed from bundle-local configuration;
  no redundant base-model weights or network access are needed.
- The retained dormant FAST head reads bundle-local vocabulary metadata.
- Legacy source-label construction helpers and training initialization helpers
  are omitted while their geometry/configuration validators are retained.

The bundle loader restores the complete trained state strictly with
`assign=True`, preserving mixed BF16/FP32 storage. It moves devices without a
global dtype conversion. The policy runtime executes CUDA inference under BF16
autocast, following training's computation policy.

NVIDIA C-RADIO Python assets are copied from the pinned upstream model snapshot
when exporting a bundle. Their copyright and Apache-2.0 headers are preserved,
and the bundle includes the Apache-2.0 license text. Qwen tokenizer/config assets
retain their upstream license. FAST tokenizer metadata is from the
Apache-2.0 Physical Intelligence FAST distribution; its neural decoder weights
are already part of the complete trained checkpoint.
