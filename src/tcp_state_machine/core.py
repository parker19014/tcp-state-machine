"""Core implementation of a simplified TCP connection state machine.

The model tracks a single TCP connection through the portion of the TCP
state diagram that a passive observer of sent/received segments can infer:

    CLOSED -> SYN_SENT -> ESTABLISHED -> FIN_WAIT_1 -> FIN_WAIT_2
                                                          -> TIME_WAIT -> CLOSED

Design decisions
----------------
* Only one direction of close is tracked.  Real TCP supports simultaneous
  close (both ends FIN at once), which collapses FIN_WAIT_1 straight to
  CLOSING.  That path is intentionally NOT modelled here because distinguishing
  it reliably from an ordered close requires sequence-number awareness, which
  this library deliberately does not have.  See README for the full rationale.
* TIME_WAIT duration is a constructor argument (``time_wait_duration``) and the
  current time is supplied by an injectable ``clock`` callable.  This keeps the
  library deterministic and testable without sleeping.
* RST handling: an RST in any state returns the machine to CLOSED.  An RST in
  CLOSED is a no-op rather than an error, matching real stacks.
* Unexpected segments (e.g. a FIN in SYN_SENT) raise ``InvalidTransition``
  rather than being silently dropped, so callers can decide how loud to be.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional


class TCPState(Enum):
    CLOSED = "CLOSED"
    SYN_SENT = "SYN_SENT"
    SYN_RECEIVED = "SYN_RECEIVED"
    ESTABLISHED = "ESTABLISHED"
    FIN_WAIT_1 = "FIN_WAIT_1"
    FIN_WAIT_2 = "FIN_WAIT_2"
    TIME_WAIT = "TIME_WAIT"


class Event(Enum):
    SYN = "SYN"
    SYN_ACK = "SYN_ACK"
    FIN = "FIN"
    FIN_ACK = "FIN_ACK"
    RST = "RST"
    DATA = "DATA"


class InvalidTransition(Exception):
    """Raised when a segment arrives that has no valid transition from the
    current state.

    A genuinely unexpected segment usually indicates a buggy peer or a buggy
    caller; raising lets the caller choose to retry, log, or tear down.
    """


class TCPStateMachine:
    """A single TCP connection tracked as a finite-state automaton.

    Parameters
    ----------
    time_wait_duration:
        How long, in seconds, the connection must remain in TIME_WAIT before
        it transitions back to CLOSED.  Defaults to 30, not 2*MSL, because the
        library does not presume a particular network configuration; callers
        who care should pass an explicit value.
    clock:
        A zero-argument callable returning the current time as a number.  Time
        is only consulted on entry to and ticks within TIME_WAIT, so any
        monotonic counter works.  Injected rather than read from ``time.time``
        so tests never depend on wall-clock behaviour.
    """

    def __init__(
        self,
        time_wait_duration: float = 30.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if time_wait_duration < 0:
            raise ValueError("time_wait_duration must be non-negative")
        self._time_wait_duration = time_wait_duration
        self._clock = clock if clock is not None else _default_clock
        self._state = TCPState.CLOSED
        self._time_wait_entered_at: Optional[float] = None
        self._last_transition_reason: Optional[Event] = None

    # ------------------------------------------------------------------
# Properties
    # ------------------------------------------------------------------
    @property
    def state(self) -> TCPState:
        return self._state

    @property
    def last_transition_reason(self) -> Optional[Event]:
        """The segment event that caused the most recent state change, or
        ``None`` if no transition has occurred yet."""
        return self._last_transition_reason

    @property
    def time_wait_remaining(self) -> Optional[float]:
        """Seconds left in TIME_WAIT, or ``None`` when not in TIME_WAIT.

        Clamped at zero so a slow consumer never observes a negative value.
        """
        if self._state is not TCPState.TIME_WAIT or self._time_wait_entered_at is None:
            return None
        elapsed = self._clock() - self._time_wait_entered_at
        remaining = self._time_wait_duration - elapsed
        return remaining if remaining > 0 else 0.0

    # ------------------------------------------------------------------
# Public API
    # ------------------------------------------------------------------
    def handle(self, event: Event) -> TCPState:
        """Process one segment event and return the resulting state.

        Raises ``InvalidTransition`` for segments that cannot legally arrive
        in the current state.  RST is the one exception: it always resets to
        CLOSED and never raises, even from CLOSED.
        """
        if not isinstance(event, Event):
            raise TypeError(f"event must be an Event, got {type(event).__name__}")

        # RST is universal: tear everything down, silently if already closed.
        if event is Event.RST:
            self._reset()
            return self._state

        new_state = self._transition(self._state, event)
        if new_state is None:
            raise InvalidTransition(
                f"{event.value} is not valid in state {self._state.value}"
            )

        self._state = new_state
        self._last_transition_reason = event
        if new_state is TCPState.TIME_WAIT:
            self._time_wait_entered_at = self._clock()
        else:
            self._time_wait_entered_at = None
        return self._state

    def tick(self) -> Optional[TCPState]:
        """Advance time-dependent state.

        The only state with a time component is TIME_WAIT.  When the wait has
        elapsed the machine moves to CLOSED and returns that state; otherwise
        returns the current state unchanged.  Calling ``tick`` outside
        TIME_WAIT is harmless and returns the current state.
        """
        if self._state is TCPState.TIME_WAIT and self.time_wait_remaining == 0.0:
            self._state = TCPState.CLOSED
            self._time_wait_entered_at = None
            self._last_transition_reason = None
        return self._state

    # ------------------------------------------------------------------
# Internals
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self._state = TCPState.CLOSED
        self._time_wait_entered_at = None
        self._last_transition_reason = Event.RST

    @staticmethod
    def _transition(current: TCPState, event: Event) -> Optional[TCPState]:
        """Pure transition table.  Returns the destination state or ``None``
        if the event is illegal from ``current``.

        Kept as a static method so it is trivially unit-testable without
        constructing a machine, and so callers can inspect the table if they
        wish to build tooling on top.
        """
        table: dict[tuple[TCPState, Event], TCPState] = {
            # Active open: we send SYN, peer responds with SYN-ACK.
            (TCPState.CLOSED, Event.SYN): TCPState.SYN_SENT,
            (TCPState.SYN_SENT, Event.SYN_ACK): TCPState.ESTABLISHED,
            # Passive open path: we receive a SYN, then send our own SYN-ACK.
            # We model receiving the peer's SYN as entering SYN_RECEIVED.
            (TCPState.CLOSED, Event.SYN_ACK): TCPState.SYN_RECEIVED,
            (TCPState.SYN_RECEIVED, Event.SYN): TCPState.ESTABLISHED,
            # Established data flow is a self-transition we still record.
            (TCPState.ESTABLISHED, Event.DATA): TCPState.ESTABLISHED,
            # Active close: we send FIN.
            (TCPState.ESTABLISHED, Event.FIN): TCPState.FIN_WAIT_1,
            (TCPState.FIN_WAIT_1, Event.FIN_ACK): TCPState.FIN_WAIT_2,
            # Peer's FIN arrives while we wait for their ACK of our FIN.
            # Without sequence numbers we assume an ordered close: they ACK'd
            # us and now close themselves.
            (TCPState.FIN_WAIT_2, Event.FIN): TCPState.TIME_WAIT,
            # Peer ACKs our FIN after we've already seen their FIN.  This is
            # the passive-close variant of entering TIME_WAIT.
            (TCPState.FIN_WAIT_1, Event.FIN): TCPState.TIME_WAIT,
            (TCPState.TIME_WAIT, Event.FIN_ACK): TCPState.TIME_WAIT,
        }
        return table.get((current, event))


def _default_clock() -> float:
    import time

    return time.monotonic()
