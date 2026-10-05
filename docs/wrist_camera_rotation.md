# Astribot wrist RGB rotation

Each wrist image in the Orin observation may carry integer `rotation_degrees`
equal to0 or180. The adapter applies that transform once after RGB decoding.
Invalid strings, booleans, floats and other angles fail validation.

Explicit metadata overrides the default. Without metadata, native HKU
stock-gripper serving (`end_effector=gripper`) keeps wrist images unrotated,
matching the actual training exporter and video. ApexHand keeps its legacy
180-degree default. Head stereo cropping is unchanged.

The Orin Policy Status Feedback panel exposes independent left/right options
and freezes them while feedback is initialized. Camera preview rotation is
separate from policy input; use the policy controls to match checkpoint and
physical mounting. This change does not alter absolute action20 semantics or
add first-target continuity/rate limits. Physical policy tests remain suspended
after the Task1 incident until those protections and robot inspection are done.
