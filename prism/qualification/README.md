# Runtime parity qualification

Run the original private model graph and the standalone graph in **separate
processes**, using the same final checkpoint, prepared observations, masks,
tokenizer, preprocessing, BF16 autocast policy and explicit flow noise. Both
processes expose `prism_vla`, so importing them together is not a valid comparison.

`capture.py` accepts a portable fixture JSON plus an adjacent NPZ. The JSON has
schema `prism_engram_parity_fixture_v1` and `cases`, each with a unique `name`,
optional `seed`, and `requests` list. A request value `{"array":"head_rgb"}`
loads the corresponding numeric NPZ array. Fixtures should cover all four
embodiments, head-only/both-wrist/asymmetric-wrist observations, JSON and plain
prompts, available and unavailable hand feedback, and at least nine sequential
10 Hz observations. Each case is reset and its first observation replayed with
identical noise. The fixture and checkpoint identity are included in receipts.
After the isolated cases, capture also interleaves two independent sessions as
`A0, B0, A1`, choosing different robot/prompt identities. It stores those outputs
and compares them against isolated `A0/A1/B0` execution with identical flow
noise. A session-isolation failure stops qualification. The comparator verifies
the numeric probe arrays itself; CUDA qualification requires this proof.

`make_fixture.py --bundle /models/engram --output /checks/cases.json` generates
19 cases/94 observations with synthetic deterministic RGB patterns, saved mean
joint states and nonidentity EEF anchors. It covers all four routes, prompt
formats, optional cameras and hand availability without opening any training
dataset. This is a numerical graph-portability fixture, not recorded policy
evaluation data. External recorded-observation fixtures can use the same schema.

The standalone capture uses the public loader:

```bash
uv run qualification/capture.py --bundle /models/engram --fixture /checks/cases.json \
  --output /checks/standalone.json
```

For the reference capture, supply `--factory original_graph:load_bundle` and set
`PRISM_PARITY_REFERENCE_SOURCE` to the original repository's `src` directory and
`PRISM_PARITY_REFERENCE_CHECKPOINT` to the original complete checkpoint. Set
`HF_HUB_OFFLINE=1` so missing original architecture assets fail locally. This
developer-only factory constructs the original private graph, loads the same final
weights strictly with their original storage dtypes, and returns an object with
the public loaded-bundle fields (`model`, `tokenizer`, `config`,
`normalization`, `manifest`, `path`). It does not substitute randomly initialized
heads, globally round FP32 weights to BF16, or use training-only future labels.
The factory requires access to the original code and locally cached base assets;
normal serving has no such dependency. No private source or asset paths are
included in capture receipts. The public serving adapter and runtime remain the
same. This comparison isolates graph portability; independent adapter
tests establish normalization and physical action-decoding correctness.

```bash
uv run qualification/compare.py --reference /checks/reference.json \
  --candidate /checks/standalone.json --require-gpu --output /checks/parity.json
```

The comparator requires matching input/weight identities, tensor inventories,
shapes and dtypes, finite values, and the declared numeric tolerance. A passing
CPU fixture comparison never claims CUDA checkpoint qualification. Numerical
runtime parity is separate from closed-loop policy quality.

The capture tool needs an idle GPU. It does not allocate, stop, resume or alter
ACP jobs, and must not be run on a GPU occupied by active training.

`run.py` orchestrates both isolated processes, refuses an occupied GPU, holds a
local lock and bounds total capture time. Its output directory must be new:

```bash
uv run qualification/run.py --bundle /models/engram --fixture /checks/cases.json \
  --reference-source /original-repository/src \
  --reference-checkpoint /original-checkpoint --gpu 0 \
  --output-dir /checks/qualification-attempt-1
```

Only developers with access to the original source need this qualification
command. Normal users need only the standalone server and exported bundle.
