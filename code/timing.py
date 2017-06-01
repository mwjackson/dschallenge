import time


def timeit(f):
    start = time.time()
    ret = f()
    elapsed = time.time() - start
    print(elapsed)

