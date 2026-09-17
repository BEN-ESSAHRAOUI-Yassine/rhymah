## Note Geometry and Composite Events

The game may display notes as individual objects or as connected
multi-lane structures.

The detector MUST NOT assume that every note consists of a single
circular head.

Supported visual structures include:

1. Single short note
2. Single-lane vertical hold note
3. Multi-lane simultaneous notes
4. Multi-lane connected notes
5. Multi-lane sustained/connected events

A visual note event may contain:

- one or more note heads
- one or more lanes
- one or more connecting bars
- a start position
- an end position
- a hit time
- a release time

The detector must first detect visual primitives:

- circular note heads
- elongated vertical bars
- elongated horizontal bars
- lane positions

It must then group related primitives into a single logical
NoteEvent.

The system must preserve relationships between note heads.
Two heads connected by the same visual structure must not
automatically be treated as two unrelated notes.

Composite events must be represented explicitly rather than
flattened into independent notes.
