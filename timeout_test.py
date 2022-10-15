import signal
from contextlib import contextmanager
import time
@contextmanager
def timeout(time):
    def raise_timeout(signum, frame):
        raise TimeoutError(f'block timeout after {time} seconds')
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.alarm(0)

def sleeper(duration):
    time.sleep(duration)
    print('finished')

if __name__ == '__main__':
    with timeout(5):
        sleeper(3)