# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# distutils: language = c
"""Fast alpha-beta endgame solver for Ulti (Cython implementation).

Supports pluggable contract evaluators via C-level function pointers.
The search engine (minimax, legal actions, apply/undo) is fully
contract-agnostic; the contract only affects terminal evaluation,
early termination, score bounds, and move ordering.

Supported contracts:
    "parti"       — soloist wins majority of card points
    "betli"       — soloist must take zero tricks
    "durchmars"   — soloist must win all 10 tricks
    "parti_ulti"  — parti + soloist must win last trick with trump 7

Card encoding (matches the Python encoder):
    card_id = suit_idx * 8 + rank_idx  (0..31)
    suit: 0=HEARTS, 1=BELLS, 2=LEAVES, 3=ACORNS
    rank: 0=SEVEN, 1=EIGHT, 2=NINE, 3=JACK, 4=QUEEN, 5=KING, 6=TEN, 7=ACE

Usage:
    from trickster._solver_core import solve_root, solve_best

    values = solve_root(gs)                          # auto-detect contract
    values = solve_root(gs, contract="parti_ulti")   # explicit contract
    card, val = solve_best(gs, contract="betli")
"""

# ===========================================================================
#  Compile-time constants
# ===========================================================================

DEF C_NP = 3            # num players
DEF C_TRICKS = 10       # tricks per game
DEF C_LAST_BONUS = 10   # last trick bonus
DEF C_NO_TRUMP = -1     # sentinel for betli / no trump
DEF C_MAX_MOVES = 10    # max cards in a hand
DEF C_INF = 1000000.0   # +/- infinity for alpha-beta

# Contract IDs (used by _get_eval)
DEF EV_PARTI = 0
DEF EV_BETLI = 1
DEF EV_DURCHMARS = 2
DEF EV_PARTI_ULTI = 3

# Betli terminal values (binary: won all 0 tricks or not)
DEF C_BETLI_WIN = 10.0
DEF C_BETLI_LOSE = 0.0

# Durchmars terminal values (binary)
DEF C_DM_WIN = 10.0
DEF C_DM_LOSE = 0.0

# Ulti bonus/penalty (large enough to dominate card-point differences)
DEF C_ULTI_BONUS = 100.0

# ===========================================================================
#  Inline bit helpers (portable, no compiler intrinsics)
# ===========================================================================

cdef inline int _popcount(unsigned int x) noexcept nogil:
    x = x - ((x >> 1) & 0x55555555u)
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u)
    return <int>((((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24)

cdef inline int _ctz(unsigned int x) noexcept nogil:
    cdef int n = 0
    if x == 0:
        return 32
    if (x & 0x0000FFFFu) == 0:
        n += 16; x >>= 16
    if (x & 0x000000FFu) == 0:
        n += 8; x >>= 8
    if (x & 0x0000000Fu) == 0:
        n += 4; x >>= 4
    if (x & 0x00000003u) == 0:
        n += 2; x >>= 2
    if (x & 0x00000001u) == 0:
        n += 1
    return n

# ===========================================================================
#  Card helpers
# ===========================================================================

cdef inline int _suit(int c) noexcept nogil:
    return c >> 3

cdef inline int _rank(int c) noexcept nogil:
    return c & 7

cdef inline unsigned int _bit(int c) noexcept nogil:
    return 1u << c

cdef inline unsigned int _smask(int s) noexcept nogil:
    return 0xFFu << (s * 8)

cdef inline int _pts(int c) noexcept nogil:
    cdef int r = c & 7
    if r >= 6:
        return 10
    return 0

cdef inline int _str_n(int c) noexcept nogil:
    return c & 7

cdef inline int _str_b(int c) noexcept nogil:
    cdef int r = c & 7
    if r == 3: return 4   # JACK
    if r == 4: return 5   # QUEEN
    if r == 5: return 6   # KING
    if r == 6: return 3   # TEN
    return r

cdef inline int _strength(int c, int betli) noexcept nogil:
    if betli:
        return _str_b(c)
    return _str_n(c)

# ===========================================================================
#  C-level game state
# ===========================================================================

cdef struct CState:
    unsigned int hands[3]
    int trump               # 0..3 or C_NO_TRUMP
    int betli
    int soloist
    int leader
    int trick_no
    int tc_n                # current trick card count (0, 1, 2)
    int tc_p[2]             # current trick players
    int tc_c[2]             # current trick card IDs
    int scores[3]
    int tricks_won[3]
    int has_ulti
    # Last completed trick (for ulti check at terminal)
    int lt_p[3]             # last trick players
    int lt_c[3]             # last trick card IDs
    int lt_winner           # last trick winner (-1 if none)


cdef struct Undo:
    int player
    int card
    int done                # 1 = trick completed
    int stc_p[2]
    int stc_c[2]
    int stc_n
    int sleader
    int strick_no
    int sscores[3]
    int stw[3]
    # Saved last-trick info (only when done=1)
    int s_lt_p[3]
    int s_lt_c[3]
    int s_lt_winner


cdef struct Moves:
    int c[C_MAX_MOVES]
    int n

# ===========================================================================
#  Current player
# ===========================================================================

cdef inline int _cur(CState* s) noexcept nogil:
    cdef int p = s.leader
    cdef int i
    for i in range(s.tc_n):
        p = (p + 1) % C_NP
    return p

# ===========================================================================
#  Trick winner (unrolled for 3 cards)
# ===========================================================================

cdef inline int _tw3(
    int p0, int c0, int p1, int c1, int p2, int c2,
    int trump, int betli,
) noexcept nogil:
    cdef int led = _suit(c0)
    cdef int bp = p0, bc = c0, bt = (trump >= 0 and _suit(c0) == trump)
    cdef int ct, ss, bs

    ct = (trump >= 0 and _suit(c1) == trump)
    if ct and not bt:
        bp = p1; bc = c1; bt = 1
    elif ct and bt:
        ss = _strength(c1, betli); bs = _strength(bc, betli)
        if ss > bs:
            bp = p1; bc = c1
    elif not ct and not bt:
        if _suit(c1) == led:
            if _suit(bc) == led:
                ss = _strength(c1, betli); bs = _strength(bc, betli)
                if ss > bs:
                    bp = p1; bc = c1
            else:
                bp = p1; bc = c1

    ct = (trump >= 0 and _suit(c2) == trump)
    if ct and not bt:
        bp = p2; bc = c2; bt = 1
    elif ct and bt:
        ss = _strength(c2, betli); bs = _strength(bc, betli)
        if ss > bs:
            bp = p2; bc = c2
    elif not ct and not bt:
        if _suit(c2) == led:
            if _suit(bc) == led:
                ss = _strength(c2, betli); bs = _strength(bc, betli)
                if ss > bs:
                    bp = p2; bc = c2
            else:
                bp = p2; bc = c2

    return bp

# ===========================================================================
#  Apply / Undo
# ===========================================================================

cdef inline void _apply(CState* s, int card, Undo* u) noexcept nogil:
    cdef int p = _cur(s)
    cdef int w, pts, i

    u.player = p
    u.card = card
    s.hands[p] &= ~_bit(card)

    if s.tc_n < 2:
        s.tc_p[s.tc_n] = p
        s.tc_c[s.tc_n] = card
        s.tc_n += 1
        u.done = 0
        return

    # --- Trick completing (3rd card) ---
    u.done = 1
    # Save trick state
    u.stc_p[0] = s.tc_p[0]; u.stc_p[1] = s.tc_p[1]
    u.stc_c[0] = s.tc_c[0]; u.stc_c[1] = s.tc_c[1]
    u.stc_n = s.tc_n
    u.sleader = s.leader
    u.strick_no = s.trick_no
    for i in range(C_NP):
        u.sscores[i] = s.scores[i]
        u.stw[i] = s.tricks_won[i]
    # Save last-trick info
    u.s_lt_p[0] = s.lt_p[0]; u.s_lt_p[1] = s.lt_p[1]; u.s_lt_p[2] = s.lt_p[2]
    u.s_lt_c[0] = s.lt_c[0]; u.s_lt_c[1] = s.lt_c[1]; u.s_lt_c[2] = s.lt_c[2]
    u.s_lt_winner = s.lt_winner

    # Record this trick as the last trick (before clearing)
    s.lt_p[0] = s.tc_p[0]; s.lt_c[0] = s.tc_c[0]
    s.lt_p[1] = s.tc_p[1]; s.lt_c[1] = s.tc_c[1]
    s.lt_p[2] = p;          s.lt_c[2] = card

    # Resolve
    w = _tw3(s.tc_p[0], s.tc_c[0], s.tc_p[1], s.tc_c[1],
             p, card, s.trump, s.betli)
    pts = _pts(s.tc_c[0]) + _pts(s.tc_c[1]) + _pts(card)
    s.scores[w] += pts
    s.tricks_won[w] += 1
    s.trick_no += 1
    if s.trick_no == C_TRICKS:
        s.scores[w] += C_LAST_BONUS
    s.leader = w
    s.lt_winner = w
    s.tc_n = 0


cdef inline void _undo(CState* s, Undo* u) noexcept nogil:
    cdef int i
    s.hands[u.player] |= _bit(u.card)
    if u.done:
        s.tc_p[0] = u.stc_p[0]; s.tc_p[1] = u.stc_p[1]
        s.tc_c[0] = u.stc_c[0]; s.tc_c[1] = u.stc_c[1]
        s.tc_n = u.stc_n
        s.leader = u.sleader
        s.trick_no = u.strick_no
        for i in range(C_NP):
            s.scores[i] = u.sscores[i]
            s.tricks_won[i] = u.stw[i]
        s.lt_p[0] = u.s_lt_p[0]; s.lt_p[1] = u.s_lt_p[1]; s.lt_p[2] = u.s_lt_p[2]
        s.lt_c[0] = u.s_lt_c[0]; s.lt_c[1] = u.s_lt_c[1]; s.lt_c[2] = u.s_lt_c[2]
        s.lt_winner = u.s_lt_winner
    else:
        s.tc_n -= 1

# ===========================================================================
#  Legal actions (contract-agnostic — pure game mechanics)
# ===========================================================================

cdef inline int _max_str(unsigned int mask, int betli) noexcept nogil:
    cdef int best = -1, c, st
    while mask:
        c = _ctz(mask)
        st = _strength(c, betli)
        if st > best:
            best = st
        mask &= mask - 1
    return best


cdef void _legal(CState* s, Moves* m) noexcept nogil:
    cdef int p = _cur(s)
    cdef unsigned int hand = s.hands[p]
    cdef int n = 0
    cdef int led, c, i, j, mx, t7
    cdef unsigned int sc, ps, hi, tmp, trumps, tp, ht

    if s.tc_n == 0:
        tmp = hand
        while tmp:
            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
    else:
        led = _suit(s.tc_c[0])
        sc = hand & _smask(led)

        if sc:
            if s.betli:
                tmp = sc
                while tmp:
                    c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
            else:
                ps = 0
                for i in range(s.tc_n):
                    if _suit(s.tc_c[i]) == led:
                        ps |= _bit(s.tc_c[i])
                if ps:
                    mx = _max_str(ps, 0)
                    hi = 0
                    tmp = sc
                    while tmp:
                        c = _ctz(tmp)
                        if _str_n(c) > mx:
                            hi |= _bit(c)
                        tmp &= tmp - 1
                    if hi:
                        tmp = hi
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                    else:
                        tmp = sc
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                else:
                    tmp = sc
                    while tmp:
                        c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
        else:
            if s.betli or s.trump < 0:
                tmp = hand
                while tmp:
                    c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
            else:
                trumps = hand & _smask(s.trump)
                if not trumps:
                    tmp = hand
                    while tmp:
                        c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                else:
                    tp = 0
                    for i in range(s.tc_n):
                        if _suit(s.tc_c[i]) == s.trump:
                            tp |= _bit(s.tc_c[i])
                    if tp:
                        mx = _max_str(tp, 0)
                        ht = 0
                        tmp = trumps
                        while tmp:
                            c = _ctz(tmp)
                            if _str_n(c) > mx:
                                ht |= _bit(c)
                            tmp &= tmp - 1
                        if ht:
                            tmp = ht
                            while tmp:
                                c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                        else:
                            tmp = trumps
                            while tmp:
                                c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1
                    else:
                        tmp = trumps
                        while tmp:
                            c = _ctz(tmp); m.c[n] = c; n += 1; tmp &= tmp - 1

    # 7esre tartás
    if (s.has_ulti and p == s.soloist and s.trump >= 0
            and s.trick_no < C_TRICKS - 1 and n > 1):
        t7 = s.trump * 8
        for i in range(n):
            if m.c[i] == t7:
                n -= 1
                for j in range(i, n):
                    m.c[j] = m.c[j + 1]
                break

    m.n = n

# ===========================================================================
#  Remaining card points (for bounds calculation)
# ===========================================================================

cdef inline int _rem_pts(CState* s) noexcept nogil:
    cdef unsigned int all_c = s.hands[0] | s.hands[1] | s.hands[2]
    cdef int i
    for i in range(s.tc_n):
        all_c |= _bit(s.tc_c[i])
    cdef int pts = _popcount(all_c & 0xC0C0C0C0u) * 10
    if s.trick_no < C_TRICKS:
        pts += C_LAST_BONUS
    return pts

# ===========================================================================
#  Move ordering
# ===========================================================================

cdef void _order_default(Moves* m, CState* s, int maximising) noexcept nogil:
    """Standard ordering: trumps/high-strength for MAX, low for MIN."""
    cdef int i, j, n = m.n, ki, c, it, pt, st
    cdef int tmp
    cdef int keys[C_MAX_MOVES]

    if n <= 1:
        return

    for i in range(n):
        c = m.c[i]
        it = 1 if (s.trump >= 0 and _suit(c) == s.trump) else 0
        pt = _pts(c)
        st = _strength(c, s.betli)
        keys[i] = it * 10000 + pt * 100 + st

    for i in range(1, n):
        ki = keys[i]; tmp = m.c[i]; j = i - 1
        if maximising:
            while j >= 0 and keys[j] < ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        else:
            while j >= 0 and keys[j] > ki:
                keys[j + 1] = keys[j]; m.c[j + 1] = m.c[j]; j -= 1
        keys[j + 1] = ki; m.c[j + 1] = tmp


cdef void _order_betli(Moves* m, CState* s, int maximising) noexcept nogil:
    """Betli ordering: soloist (MAX) prefers LOW cards, defenders HIGH.

    This is the opposite of the default because the soloist wants to
    *avoid* winning tricks.
    """
    _order_default(m, s, 1 - maximising)

# ===========================================================================
#  Contract evaluator (Strategy pattern via C function pointers)
# ===========================================================================

ctypedef float (*TermFn)(CState*) noexcept nogil
ctypedef int   (*EarlyFn)(CState*) noexcept nogil
ctypedef void  (*BoundsFn)(CState*, float*, float*) noexcept nogil
ctypedef void  (*OrderFn)(Moves*, CState*, int) noexcept nogil

cdef struct ContractEval:
    TermFn   terminal     # score at terminal / early-terminal
    EarlyFn  early_term   # can we stop before trick 10?
    BoundsFn bounds       # [lo, hi] for futility pruning
    OrderFn  order        # move ordering heuristic

# ---------------------------------------------------------------------------
#  Parti evaluator (card points)
# ---------------------------------------------------------------------------

cdef float _term_parti(CState* s) noexcept nogil:
    return <float>s.scores[s.soloist]

cdef int _early_parti(CState* s) noexcept nogil:
    return 0

cdef void _bounds_parti(CState* s, float* lo, float* hi) noexcept nogil:
    cdef float sc = <float>s.scores[s.soloist]
    lo[0] = sc
    hi[0] = sc + <float>_rem_pts(s)

# ---------------------------------------------------------------------------
#  Betli evaluator (binary: 0 tricks taken = win)
# ---------------------------------------------------------------------------

cdef float _term_betli(CState* s) noexcept nogil:
    if s.tricks_won[s.soloist] == 0:
        return C_BETLI_WIN
    return C_BETLI_LOSE

cdef int _early_betli(CState* s) noexcept nogil:
    return s.tricks_won[s.soloist] > 0

cdef void _bounds_betli(CState* s, float* lo, float* hi) noexcept nogil:
    # Only called when soloist has 0 tricks (early_term handles the rest)
    lo[0] = C_BETLI_LOSE
    hi[0] = C_BETLI_WIN

# ---------------------------------------------------------------------------
#  Durchmars evaluator (binary: all 10 tricks = win)
# ---------------------------------------------------------------------------

cdef float _term_dm(CState* s) noexcept nogil:
    if s.tricks_won[s.soloist] == C_TRICKS:
        return C_DM_WIN
    return C_DM_LOSE

cdef int _early_dm(CState* s) noexcept nogil:
    # Defenders won a trick → durchmars impossible
    return s.trick_no > s.tricks_won[s.soloist]

cdef void _bounds_dm(CState* s, float* lo, float* hi) noexcept nogil:
    lo[0] = C_DM_LOSE
    hi[0] = C_DM_WIN

# ---------------------------------------------------------------------------
#  Parti + Ulti evaluator (card points + ulti bonus/penalty)
# ---------------------------------------------------------------------------

cdef inline int _ulti_check(CState* s) noexcept nogil:
    """1 if soloist won the last trick with trump 7, else 0."""
    if s.trump < 0 or s.lt_winner < 0:
        return 0
    cdef int t7 = s.trump * 8  # trump SEVEN (rank 0)
    cdef int i
    for i in range(C_NP):
        if s.lt_c[i] == t7 and s.lt_p[i] == s.soloist:
            return 1 if s.lt_winner == s.soloist else 0
    return 0

cdef float _term_parti_ulti(CState* s) noexcept nogil:
    cdef float pts = <float>s.scores[s.soloist]
    if _ulti_check(s):
        return pts + C_ULTI_BONUS
    return pts - C_ULTI_BONUS

cdef int _early_parti_ulti(CState* s) noexcept nogil:
    return 0  # ulti can only be evaluated at game end

cdef void _bounds_parti_ulti(CState* s, float* lo, float* hi) noexcept nogil:
    cdef float sc = <float>s.scores[s.soloist]
    lo[0] = sc - C_ULTI_BONUS
    hi[0] = sc + <float>_rem_pts(s) + C_ULTI_BONUS

# ---------------------------------------------------------------------------
#  Evaluator factory
# ---------------------------------------------------------------------------

cdef ContractEval _get_eval(int contract_id) noexcept nogil:
    cdef ContractEval ev
    if contract_id == EV_BETLI:
        ev.terminal = _term_betli
        ev.early_term = _early_betli
        ev.bounds = _bounds_betli
        ev.order = _order_betli
    elif contract_id == EV_DURCHMARS:
        ev.terminal = _term_dm
        ev.early_term = _early_dm
        ev.bounds = _bounds_dm
        ev.order = _order_default
    elif contract_id == EV_PARTI_ULTI:
        ev.terminal = _term_parti_ulti
        ev.early_term = _early_parti_ulti
        ev.bounds = _bounds_parti_ulti
        ev.order = _order_default
    else:  # EV_PARTI (default)
        ev.terminal = _term_parti
        ev.early_term = _early_parti
        ev.bounds = _bounds_parti
        ev.order = _order_default
    return ev

# ===========================================================================
#  Alpha-beta minimax (contract-agnostic)
# ===========================================================================

cdef long long _g_nodes
cdef long long _g_cuts


cdef float _ab(CState* s, float alpha, float beta,
               ContractEval* ev) noexcept nogil:
    """Recursive alpha-beta from soloist's perspective.

    The contract evaluator ``ev`` controls terminal scoring, early
    termination, bounds pruning, and move ordering — the search
    engine itself is contract-agnostic.
    """
    global _g_nodes, _g_cuts
    _g_nodes += 1

    # Terminal or early-terminal
    if s.trick_no >= C_TRICKS or ev.early_term(s):
        return ev.terminal(s)

    # Bounds (futility) pruning
    cdef float lo, hi
    ev.bounds(s, &lo, &hi)
    if hi <= alpha:
        return hi
    if lo >= beta:
        return lo

    cdef int player = _cur(s)
    cdef int maxi = (player == s.soloist)

    cdef Moves mv
    _legal(s, &mv)
    ev.order(&mv, s, maxi)

    if mv.n == 0:
        return ev.terminal(s)

    cdef float val, v
    cdef int i
    cdef Undo u

    if maxi:
        val = -C_INF
        for i in range(mv.n):
            _apply(s, mv.c[i], &u)
            v = _ab(s, alpha, beta, ev)
            _undo(s, &u)
            if v > val:
                val = v
            if val > alpha:
                alpha = val
            if alpha >= beta:
                _g_cuts += 1
                break
        return val
    else:
        val = C_INF
        for i in range(mv.n):
            _apply(s, mv.c[i], &u)
            v = _ab(s, alpha, beta, ev)
            _undo(s, &u)
            if v < val:
                val = v
            if val < beta:
                beta = val
            if alpha >= beta:
                _g_cuts += 1
                break
        return val

# ===========================================================================
#  Python ↔ C conversion
# ===========================================================================

_SUIT_MAP = {}
_SUIT_RMAP = {}
_CARD_CACHE = {}

def _init_maps():
    from trickster.games.ulti.cards import Suit, Rank, Card, ALL_SUITS, ALL_RANKS
    for i, s in enumerate(ALL_SUITS):
        _SUIT_MAP[s] = i
        _SUIT_RMAP[i] = s
    for s in ALL_SUITS:
        for r in ALL_RANKS:
            cid = _SUIT_MAP[s] * 8 + int(r)
            _CARD_CACHE[cid] = Card(s, r)

_init_maps()

# Contract string → C enum
_CONTRACT_MAP = {
    "parti": EV_PARTI,
    "betli": EV_BETLI,
    "durchmars": EV_DURCHMARS,
    "parti_ulti": EV_PARTI_ULTI,
    "ulti": EV_PARTI_ULTI,         # alias
}


cdef inline int _c2id(object card):
    return _SUIT_MAP[card.suit] * 8 + int(card.rank)

cdef inline object _id2c(int card_id):
    return _CARD_CACHE[card_id]


cdef CState _to_cs(object gs):
    """Convert Python GameState → CState."""
    cdef CState s
    cdef int i

    for i in range(C_NP):
        s.hands[i] = 0
        s.scores[i] = gs.scores[i]
        s.tricks_won[i] = len(gs.captured[i]) // C_NP

    for i in range(C_NP):
        for card in gs.hands[i]:
            s.hands[i] |= _bit(_c2id(card))

    s.trump = _SUIT_MAP[gs.trump] if gs.trump is not None else C_NO_TRUMP
    s.betli = 1 if gs.betli else 0
    s.soloist = gs.soloist
    s.leader = gs.leader
    s.trick_no = gs.trick_no
    s.has_ulti = 1 if gs.has_ulti else 0

    s.tc_n = len(gs.trick_cards)
    for i in range(s.tc_n):
        p, c = gs.trick_cards[i]
        s.tc_p[i] = p
        s.tc_c[i] = _c2id(c)

    # Last-trick info
    if gs.last_trick is not None:
        s.lt_winner = gs.last_trick.winner
        for i in range(C_NP):
            s.lt_p[i] = gs.last_trick.players[i]
            s.lt_c[i] = _c2id(gs.last_trick.cards[i])
    else:
        s.lt_winner = -1
        for i in range(C_NP):
            s.lt_p[i] = -1
            s.lt_c[i] = -1

    return s


def _detect_contract(gs):
    """Auto-detect contract type from GameState flags."""
    if gs.betli:
        return "betli"
    if getattr(gs, 'has_ulti', False):
        return "parti_ulti"
    return "parti"

# ===========================================================================
#  Public API
# ===========================================================================

# List of available contracts for discovery
CONTRACTS = list(_CONTRACT_MAP.keys())


def solve_root(gs, int max_exact_tricks=C_TRICKS, contract=None):
    """Compute exact value for every legal move.

    Parameters
    ----------
    gs : GameState
    max_exact_tricks : int
        Kept for API compatibility; ignored.
    contract : str or None
        Contract type: "parti", "betli", "durchmars", "parti_ulti".
        If None, auto-detected from gs.betli / gs.has_ulti.

    Returns
    -------
    dict[Card, float]
        Value for each legal move from the soloist's perspective.
    """
    global _g_nodes, _g_cuts
    _g_nodes = 0; _g_cuts = 0

    if contract is None:
        contract = _detect_contract(gs)
    cdef int ev_id = _CONTRACT_MAP.get(contract, EV_PARTI)
    cdef ContractEval ev = _get_eval(ev_id)

    cdef CState s = _to_cs(gs)
    cdef int player = _cur(&s)
    cdef int maxi = (player == s.soloist)

    cdef Moves mv
    _legal(&s, &mv)
    ev.order(&mv, &s, maxi)

    cdef dict values = {}
    cdef float v
    cdef Undo u
    cdef int i

    for i in range(mv.n):
        _apply(&s, mv.c[i], &u)
        v = _ab(&s, -C_INF, C_INF, &ev)
        _undo(&s, &u)
        values[_id2c(mv.c[i])] = v

    return values


def solve_best(gs, int max_exact_tricks=C_TRICKS, contract=None):
    """Find the best move and its value.

    Returns
    -------
    (Card | None, float)
    """
    global _g_nodes, _g_cuts
    _g_nodes = 0; _g_cuts = 0

    if contract is None:
        contract = _detect_contract(gs)
    cdef int ev_id = _CONTRACT_MAP.get(contract, EV_PARTI)
    cdef ContractEval ev = _get_eval(ev_id)

    cdef CState s = _to_cs(gs)
    cdef int player = _cur(&s)
    cdef int maxi = (player == s.soloist)

    cdef Moves mv
    _legal(&s, &mv)
    ev.order(&mv, &s, maxi)

    if mv.n == 0:
        return None, 0.0
    if mv.n == 1:
        return _id2c(mv.c[0]), 0.0

    cdef float best_val, v
    cdef int best_i = 0
    cdef Undo u
    cdef int i

    if maxi:
        best_val = -C_INF
        for i in range(mv.n):
            _apply(&s, mv.c[i], &u)
            v = _ab(&s, best_val, C_INF, &ev)
            _undo(&s, &u)
            if v > best_val:
                best_val = v; best_i = i
    else:
        best_val = C_INF
        for i in range(mv.n):
            _apply(&s, mv.c[i], &u)
            v = _ab(&s, -C_INF, best_val, &ev)
            _undo(&s, &u)
            if v < best_val:
                best_val = v; best_i = i

    return _id2c(mv.c[best_i]), best_val


def get_stats():
    """Diagnostics from the most recent solve call."""
    return {
        "nodes_explored": _g_nodes,
        "cutoffs": _g_cuts,
        "pruning_ratio": (_g_cuts / _g_nodes) if _g_nodes > 0 else 0.0,
    }
