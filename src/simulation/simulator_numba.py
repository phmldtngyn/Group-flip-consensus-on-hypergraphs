"""
Two-color group-flip simulator on arbitrary balls (bitmask words), Numba-accelerated.
Measures: stationary majority fraction M (batch means) and newborn extinction rate.
"""
import numpy as np
from numba import njit


@njit(inline='always')
def popcnt(x):
    c = 0
    x = np.uint64(x)
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@njit(cache=True)
def simulate_M(balls_words, n, T, burn, batch_len, seed):
    """
    balls_words: (n, W) uint64 bitmask words of each ball.
    Returns array of batch means of majority fraction after burn-in.
    """
    W = balls_words.shape[1]
    np.random.seed(seed)
    fullw = np.empty(W, dtype=np.uint64)
    for w in range(W):
        fullw[w] = np.uint64(0xFFFFFFFFFFFFFFFF)
    rem = n % 64
    if rem != 0:
        fullw[W - 1] = (np.uint64(1) << np.uint64(rem)) - np.uint64(1)

    bits0 = fullw.copy()          # all nodes color 0
    pop0 = np.int64(n)

    n_batches = (T - burn) // batch_len
    sums = np.zeros(n_batches)
    cnts = np.zeros(n_batches)

    for t in range(T):
        i = np.random.randint(0, n)
        wi = i // 64
        bi = i % 64
        c_is_0 = ((bits0[wi] >> np.uint64(bi)) & np.uint64(1)) == np.uint64(1)

        delta = np.int64(0)
        for w in range(W):
            m = balls_words[i, w]
            if m == 0:
                continue
            if c_is_0:
                f = bits0[w] & m
            else:
                f = (~bits0[w]) & m & fullw[w]
            if f:
                bits0[w] ^= f
                k = np.int64(popcnt(f))
                if c_is_0:
                    delta -= k
                else:
                    delta += k
        pop0 += delta

        if t >= burn:
            b = (t - burn) // batch_len
            maj = max(pop0, np.int64(n) - pop0) / n
            sums[b] += maj
            cnts[b] += 1.0

    out = np.empty(n_batches)
    for b in range(n_batches):
        out[b] = sums[b] / cnts[b] if cnts[b] > 0 else np.nan
    return out


@njit(cache=True)
def newborn_extinction_test(balls_words, n, T, seed):
    """
    Newborn identity (main text App. C, SM Prop. F.1).

    A *birth* is any fire from a monochrome state; the newborn minority is
    the minority colour class just after it. A *trial* is recorded when the
    next fire is by a node of that newborn minority; it is a success if this
    fire makes the state monochrome again. If the next fire is by a majority
    node, no trial is recorded. Returns (trials, successes); the success
    fraction estimates c_tilde.

    Revision audit (Nov 2026): the previous version declared a birth when
    the state AFTER a fire was monochrome (that is an erasure, not a birth),
    so it could never record a success; data/processed/newborn_test_v2.csv,
    produced with it, has 0 successes in all 72 rows and is archived.
    """
    W = balls_words.shape[1]
    np.random.seed(seed)
    fullw = np.empty(W, dtype=np.uint64)
    for w in range(W):
        fullw[w] = np.uint64(0xFFFFFFFFFFFFFFFF)
    rem = n % 64
    if rem != 0:
        fullw[W - 1] = (np.uint64(1) << np.uint64(rem)) - np.uint64(1)

    bits0 = fullw.copy()          # set bits = nodes of colour 0
    pop0 = np.int64(n)
    res = np.zeros(W, dtype=np.uint64)
    active = False                # a newborn exists and no fire has happened since
    trials = 0
    succ = 0

    for t in range(T):
        was_mono = (pop0 == 0) or (pop0 == n)
        i = np.random.randint(0, n)
        wi = i // 64
        bi = i % 64
        c_is_0 = ((bits0[wi] >> np.uint64(bi)) & np.uint64(1)) == np.uint64(1)
        in_res = ((res[wi] >> np.uint64(bi)) & np.uint64(1)) == np.uint64(1)

        delta = np.int64(0)
        for w in range(W):
            m = balls_words[i, w]
            if m == 0:
                continue
            if c_is_0:
                f = bits0[w] & m
            else:
                f = (~bits0[w]) & m & fullw[w]
            if f:
                bits0[w] ^= f
                k = np.int64(popcnt(f))
                if c_is_0:
                    delta -= k
                else:
                    delta += k
        pop0 += delta
        mono = (pop0 == 0) or (pop0 == n)

        if was_mono:
            # birth: newborn = minority colour class after this fire
            active = False
            if not mono and 2 * pop0 != n:
                minority_is_0 = 2 * pop0 < n
                for w in range(W):
                    if minority_is_0:
                        res[w] = bits0[w]
                    else:
                        res[w] = (~bits0[w]) & fullw[w]
                active = True
        elif active:
            if in_res:
                trials += 1
                if mono:
                    succ += 1
            active = False

    return trials, succ
