# Tracking-Lost Status Design

## Goal

Make a tracking loss unmistakable in the GUI without preventing automatic
recovery. When a previously tracked hand disappears, the application must stop
sending new joint commands immediately, flash `TRACKING LOST` for two seconds,
and then either show the recovered runtime state or enter a persistent fault.

## Scope

This change is limited to runtime tracking-loss behavior and state presentation
in `realtime_orcahand.py`. It does not reorganize the rest of the dashboard,
change landmark detection, or alter joint estimation.

## Runtime Behavior

1. While `LIVE`, the first frame without a selected hand moves the state machine
   to `TRACKING_LOST`. Because hardware commands are sent only while the state is
   `LIVE`, no new joint command is sent after this transition. The controller
   remains connected and torque remains enabled, so the hand holds its last
   commanded pose and can recover automatically.
2. The state machine remembers the state that preceded the loss. A valid hand
   detected within two seconds restores that state, normally `LIVE`.
3. A tracking loss that lasts for two seconds enters `FAULT` and invokes the
   existing emergency-stop path. `FAULT` remains visible until the existing
   fault-reset/reconnect flow clears it.
4. A new loss starts a new two-second timeout and a new visual notification.

The two-second timeout replaces the current 0.3-second tracking-loss stop
window. It is both the maximum automatic-recovery window and the duration of the
visible tracking-loss notification.

## GUI Behavior

- On entry to `TRACKING_LOST`, the preview state indicator and System Snapshot
  state value immediately display `TRACKING LOST`.
- The preview state indicator alternates between an alert color and its panel
  background every 250 ms for two seconds. The System Snapshot value remains
  readable as `TRACKING LOST` throughout the notification.
- If tracking recovers before two seconds, runtime command output may resume
  immediately, but the visual notification completes its full two-second cycle
  so the event cannot disappear before an operator notices it. At the end of
  the cycle, both state displays show the actual restored state.
- If tracking does not recover, the timeout changes the state to `FAULT`, stops
  blinking, and immediately shows `FAULT` in the danger color.
- Starting a later tracking-loss event cancels any stale blink callbacks from
  the earlier event by using an event generation/token check.
- Closing the GUI leaves scheduled callbacks harmless: each callback checks
  that its event token is still current and that the target widget exists.

## State Refresh Boundary

Add one GUI method responsible for synchronizing runtime-state text and visual
style. State-transition paths call it immediately instead of relying only on
the next `_tick()` and `_update_button_states()` pass. Button enablement remains
in `_update_button_states()`; state presentation is no longer coupled to button
snapshot deduplication.

## Testing

Automated tests will cover these observable behaviors:

1. The first missing-hand frame changes `LIVE` to `TRACKING_LOST` and prevents
   hardware output without invoking emergency stop.
2. A hand recovered before two seconds restores the previous runtime state.
3. A loss shorter than two seconds does not fault; a loss reaching two seconds
   invokes the existing safety-stop path and enters `FAULT`.
4. The GUI state refresh starts a two-second tracking-loss notification and
   presents `TRACKING LOST` immediately.
5. The notification preserves the visible alert through early recovery, then
   reveals the actual state when its two-second duration finishes.
6. A fault cancels the blink and is presented immediately as a persistent
   `FAULT` state.

Existing state-machine, GUI-control, and single-file entrypoint tests must
continue to pass.
