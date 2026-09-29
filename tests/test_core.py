"""Unit tests for the TCP state machine.

All tests are deterministic: where time matters we inject a fake clock so
no test ever depends on wall-clock behaviour.
"""

import unittest

from tcp_state_machine import (
    TCPStateMachine,
    TCPState,
    Event,
    InvalidTransition,
)


class FakeClock:
    """A manually advanced clock.  Starts at zero to keep arithmetic obvious."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TransitionTableTest(unittest.TestCase):
    """Exercise the static table directly so table bugs surface without the
    surrounding state machinery in the way."""

    def test_active_open_reaches_establishished(self):
        self.assertIsNone(TCPStateMachine._transition(TCPState.CLOSED, Event.FIN))
        self.assertEqual(
            TCPStateMachine._transition(TCPState.SYN_SENT, Event.SYN_ACK),
            TCPState.ESTABLISHED,
        )

    def test_active_close_path(self):
        self.assertEqual(
            TCPStateMachine._transition(TCPState.ESTABLISHED, Event.FIN),
            TCPState.FIN_WAIT_1,
        )
        self.assertEqual(
            TCPStateMachine._transition(TCPState.FIN_WAIT_1, Event.FIN_ACK),
            TCPState.FIN_WAIT_2,
        )
        self.assertEqual(
            TCPStateMachine._transition(TCPState.FIN_WAIT_2, Event.FIN),
            TCPState.TIME_WAIT,
        )

    def test_illegal_events_return_none(self):
        self.assertIsNone(
            TCPStateMachine._transition(TCPState.SYN_SENT, Event.FIN)
        )
        self.assertIsNone(
            TCPStateMachine._transition(TCPState.ESTABLISHED, Event.SYN_ACK)
        )
        self.assertIsNone(
            TCPStateMachine._transition(TCPState.FIN_WAIT_2, Event.SYN)
        )


class HappyPathTest(unittest.TestCase):
    def test_full_active_connection_lifecycle(self):
        sm = TCPStateMachine(time_wait_duration=10.0)
        self.assertEqual(sm.state, TCPState.CLOSED)
        self.assertEqual(sm.handle(Event.SYN), TCPState.SYN_SENT)
        self.assertEqual(sm.handle(Event.SYN_ACK), TCPState.ESTABLISHED)
        self.assertEqual(sm.handle(Event.DATA), TCPState.ESTABLISHED)
        self.assertEqual(sm.handle(Event.FIN), TCPState.FIN_WAIT_1)
        self.assertEqual(sm.handle(Event.FIN_ACK), TCPState.FIN_WAIT_2)
        self.assertEqual(sm.handle(Event.FIN), TCPState.TIME_WAIT)
        self.assertEqual(sm.handle(Event.FIN_ACK), TCPState.TIME_WAIT)

    def test_last_transition_reason_tracks_event(self):
        sm = TCPStateMachine()
        self.assertIsNone(sm.last_transition_reason)
        sm.handle(Event.SYN)
        self.assertEqual(sm.last_transition_reason, Event.SYN)
        sm.handle(Event.SYN_ACK)
        self.assertEqual(sm.last_transition_reason, Event.SYN_ACK)


class PassiveOpenTest(unittest.TestCase):
    def test_passive_open_reaches_establishished(self):
        sm = TCPStateMachine()
        # We receive a SYN from the peer, then our own SYN-ACK completes it.
        self.assertEqual(sm.handle(Event.SYN_ACK), TCPState.SYN_RECEIVED)
        self.assertEqual(sm.handle(Event.SYN), TCPState.ESTABLISHED)


class TimeWaitTest(unittest.TestCase):
    def test_tick_before_expiry_stays_in_time_wait(self):
        clock = FakeClock()
        sm = TCPStateMachine(time_wait_duration=10.0, clock=clock)
        for ev in (Event.SYN, Event.SYN_ACK, Event.FIN, Event.FIN_ACK, Event.FIN):
            sm.handle(ev)
        self.assertEqual(sm.state, TCPState.TIME_WAIT)
        clock.advance(9.0)
        self.assertEqual(sm.tick(), TCPState.TIME_WAIT)
        self.assertEqual(sm.state, TCPState.TIME_WAIT)

    def test_tick_at_expiry_moves_to_closed(self):
        clock = FakeClock()
        sm = TCPStateMachine(time_wait_duration=10.0, clock=clock)
        for ev in (Event.SYN, Event.SYN_ACK, Event.FIN, Event.FIN_ACK, Event.FIN):
            sm.handle(ev)
        clock.advance(10.0)
        self.assertEqual(sm.tick(), TCPState.CLOSED)
        self.assertEqual(sm.state, TCPState.CLOSED)

    def test_time_wait_remaining_clamped_to_zero(self):
        clock = FakeClock()
        sm = TCPStateMachine(time_wait_duration=5.0, clock=clock)
        for ev in (Event.SYN, Event.SYN_ACK, Event.FIN, Event.FIN_ACK, Event.FIN):
            sm.handle(ev)
        clock.advance(100.0)
        self.assertEqual(sm.time_wait_remaining, 0.0)

    def test_time_wait_remaining_is_none_outside_time_wait(self):
        sm = TCPStateMachine(time_wait_duration=5.0)
        self.assertIsNone(sm.time_wait_remaining)
        sm.handle(Event.SYN)
        self.assertIsNone(sm.time_wait_remaining)

    def test_tick_outside_time_wait_is_noop(self):
        sm = TCPStateMachine()
        sm.handle(Event.SYN)
        self.assertEqual(sm.tick(), TCPState.SYN_SENT)
        self.assertEqual(sm.state, TCPState.SYN_SENT)


class RstTest(unittest.TestCase):
    def test_rst_from_any_state_returns_closed(self):
        for state in TCPState:
            sm = TCPStateMachine()
            # Drive the machine into ``state`` via its legal path, if any.
            _drive_to(sm, state)
            if sm.state is not state:
                continue  # not all states are reachable via the happy paths
            sm.handle(Event.RST)
            self.assertEqual(sm.state, TCPState.CLOSED)
            self.assertEqual(sm.last_transition_reason, Event.RST)

    def test_rst_in_closed_is_noop_not_error(self):
        sm = TCPStateMachine()
        self.assertEqual(sm.handle(Event.RST), TCPState.CLOSED)
        # last_transition_reason still records it for observability.
        self.assertEqual(sm.last_transition_reason, Event.RST)


class InvalidTransitionTest(unittest.TestCase):
    def test_fin_in_closed_raises(self):
        sm = TCPStateMachine()
        with self.assertRaises(InvalidTransition):
            sm.handle(Event.FIN)

    def test_syn_ack_in_established_raises(self):
        sm = TCPStateMachine()
        sm.handle(Event.SYN)
        sm.handle(Event.SYN_ACK)
        with self.assertRaises(InvalidTransition):
            sm.handle(Event.SYN_ACK)

    def test_fin_in_syn_sent_raises(self):
        sm = TCPStateMachine()
        sm.handle(Event.SYN)
        with self.assertRaises(InvalidTransition):
            sm.handle(Event.FIN)

    def test_bad_event_type_raises_type_error(self):
        sm = TCPStateMachine()
        with self.assertRaises(TypeError):
            sm.handle("SYN")  # type: ignore[arg-type]


class ConstructionTest(unittest.TestCase):
    def test_negative_duration_raises(self):
        with self.assertRaises(ValueError):
            TCPStateMachine(time_wait_duration=-1.0)

    def test_zero_duration_immediately_closes_on_tick(self):
        clock = FakeClock()
        sm = TCPStateMachine(time_wait_duration=0.0, clock=clock)
        for ev in (Event.SYN, Event.SYN_ACK, Event.FIN, Event.FIN_ACK, Event.FIN):
            sm.handle(ev)
        # Even with zero duration we enter TIME_WAIT first so callers see the
        # transition; a subsequent tick closes immediately.
        self.assertEqual(sm.state, TCPState.TIME_WAIT)
        self.assertEqual(sm.tick(), TCPState.CLOSED)


class FinWaitOneDirectTimeWaitTest(unittest.TestCase):
    """If the peer's FIN arrives while we are still in FIN_WAIT_1 (their ACK
    not yet seen), we go straight to TIME_WAIT.  This is the ordered-close
    variant the table supports."""

    def test_fin_in_fin_wait_1_goes_to_time_wait(self):
        sm = TCPStateMachine(time_wait_duration=5.0)
        sm.handle(Event.SYN)
        sm.handle(Event.SYN_ACK)
        sm.handle(Event.FIN)
        self.assertEqual(sm.state, TCPState.FIN_WAIT_1)
        self.assertEqual(sm.handle(Event.FIN), TCPState.TIME_WAIT)


def _drive_to(sm: TCPStateMachine, target: TCPState) -> None:
    """Best-effort drive of ``sm`` toward ``target`` along the legal path.

    Used only by RstTest; states not reachable via this path are skipped by
    the caller.
    """
    paths: dict[TCPState, list[Event]] = {
        TCPState.CLOSED: [],
        TCPState.SYN_SENT: [Event.SYN],
        TCPState.SYN_RECEIVED: [Event.SYN_ACK],
        TCPState.ESTABLISHED: [Event.SYN, Event.SYN_ACK],
        TCPState.FIN_WAIT_1: [Event.SYN, Event.SYN_ACK, Event.FIN],
        TCPState.FIN_WAIT_2: [Event.SYN, Event.SYN_ACK, Event.FIN, Event.FIN_ACK],
        TCPState.TIME_WAIT: [
            Event.SYN,
            Event.SYN_ACK,
            Event.FIN,
            Event.FIN_ACK,
            Event.FIN,
        ],
    }
    for ev in paths.get(target, []):
        try:
            sm.handle(ev)
        except InvalidTransition:
            break


if __name__ == "__main__":
    unittest.main()
