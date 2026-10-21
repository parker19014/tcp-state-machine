# TCP State Machine

A small, dependency-free library that tracks a single TCP connection through the
subset of the TCP state diagram an observer of sent/received segments can
infer: `CLOSED → SYN_SENT → ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 →
TIME_WAIT → CLOSED`.

```python
from tcp_state_machine import TCPStateMachine, TCPState, Event

sm = TCPStateMachine(time_wait_duration=30.0)
sm.handle(Event.SYN)        # -> SYN_SENT
sm.handle(Event.SYN_ACK)    # -> ESTABLISHED
sm.handle(Event.FIN)        # -> FIN_WAIT_1
sm.handle(Event.FIN_ACK)    # -> FIN_WAIT_2
sm.handle(Event.FIN)        # -> TIME_WAIT
# later, when time has passed:
sm.tick()                   # -> CLOSED
```

## Why this exists

The library answers one narrow question: given a stream of segment events for
one connection, what state is it in? It is the kind of thing you need when
writing a connection monitor, a teaching tool, or a test double for code that
otherwise reads live socket state. The trade-off is that it models **no
sequence numbers, no window math, and no retransmissions** — only the
state transitions themselves. If you need congestion control or
retransmission behaviour, this is the wrong layer.

## The awkward edge

Simultaneous close (both ends sending FIN at once, which in real TCP collapses
`FIN_WAIT_1` straight to `CLOSING`) is **not modelled**. Distinguishing a
simultaneous close from an ordered close without sequence numbers is
unreliable, so rather than guess, the machine treats a FIN received while in
`FIN_WAIT_1` as an ordered close and moves directly to `TIME_WAIT`. If you
need `CLOSING`, this library will not give it to you.

Time-dependent behaviour is driven by an injectable `clock` callable and a
`time_wait_duration` argument; `tick()` is what advances `TIME_WAIT` toward
`CLOSED`. Tests inject a fake clock and never sleep.

## Exported names

`TCPStateMachine`, `TCPState`, `Event`, `InvalidTransition`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

