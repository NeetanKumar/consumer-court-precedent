from ccpf.ui.limits import SessionRateLimiter


def test_blocks_after_max_requests_then_recovers():
    now = [0.0]
    rl = SessionRateLimiter(max_requests=2, window_s=60, clock=lambda: now[0])
    assert rl.allow() and rl.allow()
    assert not rl.allow()
    assert rl.retry_after_s() == 60
    now[0] = 61
    assert rl.allow()
    assert rl.retry_after_s() == 0
