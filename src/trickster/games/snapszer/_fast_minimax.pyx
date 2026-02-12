# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Fast Alpha-Beta Minimax for Snapszer — Cython implementation.

Represents the entire game state as a compact C struct:
- Cards are integers 0..19  (color*5 + rank_idx)
- Hands/captured as fixed-size int arrays
- No Python object allocation in the search loop

Provides two entry points:
- ``c_alphabeta(node, game, player)`` — drop-in for ``minimax.alphabeta``
- ``c_pimc_minimax(node, game, player, n_samples, rng)`` — drop-in for
  ``minimax.pimc_minimax``
"""

from libc.string cimport memcpy

# ── Card encoding ──────────────────────────────────────────────
# card_id = color * 5 + rank_idx
# color: 0=HEARTS, 1=BELLS, 2=LEAVES, 3=ACORNS
# rank_idx: 0→2pts(J), 1→3pts(Q), 2→4pts(K), 3→10pts(Ten), 4→11pts(Ace)
cdef int RANK_POINTS[5]
RANK_POINTS[0] = 2
RANK_POINTS[1] = 3
RANK_POINTS[2] = 4
RANK_POINTS[3] = 10
RANK_POINTS[4] = 11

cdef int NUM_CARDS = 20
cdef int MAX_HAND = 5
cdef int MAX_TRICKS = 10
cdef int EMPTY = -1

cdef inline int card_color(int card_id) noexcept nogil:
    return card_id // 5

cdef inline int card_rank_idx(int card_id) noexcept nogil:
    return card_id % 5

cdef inline int card_points(int card_id) noexcept nogil:
    return RANK_POINTS[card_id % 5]


# ── Compact game state ────────────────────────────────────────
cdef struct CState:
    int hand0[5]        # player 0's hand
    int hand1[5]        # player 1's hand
    int hand0_n         # count of cards in hand0
    int hand1_n         # count of cards in hand1
    int scores[2]
    int leader
    int trick_no
    int trump_color     # 0-3
    int talon_closed    # bool
    int talon_closed_by # -1 if not closed
    int talon_close_any_zero_tricks  # bool
    int captured_n[2]   # count of captured cards per player (for schwarz check)
    int pending_lead    # EMPTY or card_id
    # Draw pile (needed for PIMC / phase1 late)
    int draw_pile[10]
    int draw_pile_n
    int trump_card      # EMPTY or card_id
    # Marriage tracking
    int pending_marriage_suit  # -1 = none, 0-3 = suit
    int pending_marriage_pts   # 0 if none


# ── State manipulation helpers ────────────────────────────────

cdef inline void remove_card(int* hand, int* n, int card_id) noexcept nogil:
    """Remove first occurrence of card_id from hand."""
    cdef int i, j
    for i in range(n[0]):
        if hand[i] == card_id:
            # Shift down
            for j in range(i, n[0] - 1):
                hand[j] = hand[j + 1]
            n[0] -= 1
            return


cdef inline int has_card(int* hand, int n, int card_id) noexcept nogil:
    cdef int i
    for i in range(n):
        if hand[i] == card_id:
            return 1
    return 0


cdef inline int* get_hand(CState* s, int player) noexcept nogil:
    if player == 0:
        return s.hand0
    return s.hand1


cdef inline int get_hand_n(CState* s, int player) noexcept nogil:
    if player == 0:
        return s.hand0_n
    return s.hand1_n


cdef inline void set_hand_n(CState* s, int player, int n) noexcept nogil:
    if player == 0:
        s.hand0_n = n
    else:
        s.hand1_n = n


cdef inline int* get_hand_n_ptr(CState* s, int player) noexcept nogil:
    if player == 0:
        return &s.hand0_n
    return &s.hand1_n


# ── Terminal / scoring ────────────────────────────────────────

cdef inline int c_is_terminal(CState* s) noexcept nogil:
    if s.scores[0] >= 66 or s.scores[1] >= 66:
        return 1
    if s.trick_no >= MAX_TRICKS:
        return 1
    if s.hand0_n == 0 and s.hand1_n == 0:
        return 1
    return 0


cdef inline double c_outcome(CState* s, int player) noexcept nogil:
    """Compute normalized outcome in [-1, +1] for `player`."""
    cdef int w, l, pts
    cdef int s0 = s.scores[0], s1 = s.scores[1]

    if s0 >= 66 and s1 < 66:
        w = 0
    elif s1 >= 66 and s0 < 66:
        w = 1
    elif s0 >= 66 and s1 >= 66:
        w = 0 if s0 >= s1 else 1
    elif s.talon_closed and s.talon_closed_by >= 0:
        w = 1 - s.talon_closed_by
    else:
        # Last trick winner — approximate: current leader
        w = s.leader
    l = 1 - w

    if s.talon_closed and s.talon_closed_by >= 0 and w != s.talon_closed_by:
        pts = 3 if s.talon_close_any_zero_tricks else 2
    elif s.captured_n[l] == 0:
        pts = 3
    elif s.scores[l] < 33:
        pts = 2
    else:
        pts = 1

    if w == player:
        return <double>pts / 3.0
    return -<double>pts / 3.0


# ── Legal actions (Phase 2 / closed talon) ────────────────────

cdef int c_legal_actions(CState* s, int* out_actions) noexcept nogil:
    """Fill out_actions with legal actions, return count.

    Actions: card_ids (0-19), or 100+suit for marriage, or 99 for close_talon.
    """
    cdef int player, n, i, j, k
    cdef int* hand
    cdef int card, suit
    cdef int count = 0
    cdef int lead_color, lead_rank, must_follow
    cdef int same[5]
    cdef int same_n = 0
    cdef int higher[5]
    cdef int higher_n = 0
    cdef int trumps[5]
    cdef int trumps_n = 0
    cdef int talon_sz, king, queen

    if s.pending_lead != EMPTY:
        # Follower's turn
        player = 1 - s.leader
        hand = get_hand(s, player)
        n = get_hand_n(s, player)
        lead_color = card_color(s.pending_lead)
        lead_rank = card_rank_idx(s.pending_lead)
        must_follow = s.talon_closed or (s.draw_pile_n == 0 and s.trump_card == EMPTY)

        if not must_follow:
            for i in range(n):
                out_actions[count] = hand[i]
                count += 1
            return count

        # Must follow: same suit first
        same_n = 0
        higher_n = 0
        trumps_n = 0

        for i in range(n):
            card = hand[i]
            if card_color(card) == lead_color:
                same[same_n] = card
                same_n += 1
                if card_rank_idx(card) > lead_rank:
                    higher[higher_n] = card
                    higher_n += 1
            elif card_color(card) == s.trump_color:
                trumps[trumps_n] = card
                trumps_n += 1

        if same_n == 0:
            if trumps_n > 0:
                for i in range(trumps_n):
                    out_actions[count] = trumps[i]
                    count += 1
            else:
                for i in range(n):
                    out_actions[count] = hand[i]
                    count += 1
        elif higher_n > 0:
            for i in range(higher_n):
                out_actions[count] = higher[i]
                count += 1
        else:
            for i in range(same_n):
                out_actions[count] = same[i]
                count += 1
        return count

    # Leader's turn
    player = s.leader
    hand = get_hand(s, player)
    n = get_hand_n(s, player)

    # If pending marriage, must lead K or Q of that suit
    if s.pending_marriage_suit >= 0:
        suit = s.pending_marriage_suit
        for i in range(n):
            card = hand[i]
            if card_color(card) == suit and (card_rank_idx(card) == 1 or card_rank_idx(card) == 2):
                # Q (rank_idx=1) or K (rank_idx=2)
                out_actions[count] = card
                count += 1
        return count

    # All cards in hand
    for i in range(n):
        out_actions[count] = hand[i]
        count += 1

    # Close talon?
    if (not s.talon_closed and s.trump_card != EMPTY
            and s.hand0_n == MAX_HAND and s.hand1_n == MAX_HAND):
        talon_sz = s.draw_pile_n + 1  # +1 for trump_card
        if talon_sz >= 4:
            out_actions[count] = 99  # close_talon sentinel
            count += 1

    # Marriages
    for suit in range(4):
        king = suit * 5 + 2   # rank_idx 2 = King
        queen = suit * 5 + 1  # rank_idx 1 = Queen
        if has_card(hand, n, king) and has_card(hand, n, queen):
            if s.pending_marriage_suit < 0:
                out_actions[count] = 100 + suit  # marriage sentinel
                count += 1

    return count


# ── Apply action (make) ──────────────────────────────────────

cdef inline void draw_one(CState* s, int player) noexcept nogil:
    cdef int* hand = get_hand(s, player)
    cdef int* n = get_hand_n_ptr(s, player)
    if s.talon_closed:
        return
    if n[0] >= MAX_HAND:
        return
    if s.draw_pile_n > 0:
        s.draw_pile_n -= 1
        hand[n[0]] = s.draw_pile[s.draw_pile_n]
        n[0] += 1
    elif s.trump_card != EMPTY:
        hand[n[0]] = s.trump_card
        n[0] += 1
        s.trump_card = EMPTY


cdef inline void draw_to_five(CState* s, int player) noexcept nogil:
    if s.talon_closed:
        return
    while get_hand_n(s, player) < MAX_HAND:
        if s.draw_pile_n > 0 or s.trump_card != EMPTY:
            draw_one(s, player)
        else:
            break


cdef void c_apply(CState* s, int action) noexcept nogil:
    """Apply an action in-place. Does NOT save undo info."""
    cdef int leader, follower, winner
    cdef int lead_card, resp_card
    cdef int pts, suit
    cdef int lead_trump, resp_trump
    cdef int trump_jack
    cdef int* w_hand
    cdef int w_n

    # Close talon
    if action == 99:
        s.talon_closed = 1
        s.talon_closed_by = s.leader
        s.talon_close_any_zero_tricks = (s.captured_n[0] == 0) or (s.captured_n[1] == 0)
        return

    # Marriage declaration
    if action >= 100:
        suit = action - 100
        pts = 40 if suit == s.trump_color else 20
        s.scores[s.leader] += pts
        s.pending_marriage_suit = suit
        s.pending_marriage_pts = pts
        return

    # Card play
    if s.pending_lead == EMPTY:
        # Leader plays: set pending lead
        s.pending_lead = action
        remove_card(get_hand(s, s.leader), get_hand_n_ptr(s, s.leader), action)
        return

    # Follower plays: resolve trick
    leader = s.leader
    follower = 1 - leader
    lead_card = s.pending_lead
    resp_card = action

    remove_card(get_hand(s, follower), get_hand_n_ptr(s, follower), resp_card)

    # Resolve trick
    lead_trump = (card_color(lead_card) == s.trump_color)
    resp_trump = (card_color(resp_card) == s.trump_color)

    if resp_trump and not lead_trump:
        winner = follower
    elif lead_trump and not resp_trump:
        winner = leader
    elif card_color(resp_card) != card_color(lead_card):
        winner = leader
    elif card_rank_idx(resp_card) > card_rank_idx(lead_card):
        winner = follower
    else:
        winner = leader

    s.scores[winner] += card_points(lead_card) + card_points(resp_card)
    s.captured_n[winner] += 2
    s.leader = winner
    s.trick_no += 1
    s.pending_lead = EMPTY
    s.pending_marriage_suit = -1
    s.pending_marriage_pts = 0

    # Draw cards
    draw_to_five(s, winner)
    draw_to_five(s, 1 - winner)

    # Auto exchange trump jack for new leader
    if (s.trump_card != EMPTY and not s.talon_closed
            and s.draw_pile_n >= 1):
        trump_jack = s.trump_color * 5 + 0  # rank_idx 0 = Jack
        w_hand = get_hand(s, winner)
        w_n = get_hand_n(s, winner)
        if has_card(w_hand, w_n, trump_jack):
            remove_card(w_hand, get_hand_n_ptr(s, winner), trump_jack)
            w_hand = get_hand(s, winner)
            w_n = get_hand_n(s, winner)
            w_hand[w_n] = s.trump_card
            set_hand_n(s, winner, w_n + 1)
            s.trump_card = trump_jack


# ── Core Alpha-Beta ──────────────────────────────────────────

cdef struct ABResult:
    double value
    int best_action


cdef ABResult _c_alphabeta(CState* state, int max_player,
                           double alpha, double beta,
                           int depth, int max_depth) noexcept nogil:
    cdef ABResult result
    # Max theoretical actions: 5 cards + 1 close_talon + 4 marriages = 10.
    # Use 16 with headroom; c_legal_actions never exceeds this.
    cdef int actions[16]
    cdef int n_actions
    cdef int i
    cdef CState child
    cdef ABResult child_result
    cdef int player
    cdef int is_max
    cdef double value

    if c_is_terminal(state):
        result.value = c_outcome(state, max_player)
        result.best_action = EMPTY
        return result

    if depth >= max_depth:
        result.value = 0.0
        result.best_action = EMPTY
        return result

    n_actions = c_legal_actions(state, actions)
    if n_actions == 0:
        result.value = 0.0
        result.best_action = EMPTY
        return result

    # Forced move
    if n_actions == 1:
        memcpy(&child, state, sizeof(CState))
        c_apply(&child, actions[0])
        child_result = _c_alphabeta(&child, max_player, alpha, beta,
                                     depth + 1, max_depth)
        result.value = child_result.value
        result.best_action = actions[0]
        return result

    # Determine current player
    if state.pending_lead == EMPTY:
        player = state.leader
    else:
        player = 1 - state.leader
    is_max = (player == max_player)

    result.best_action = actions[0]

    if is_max:
        value = -2.0
        for i in range(n_actions):
            memcpy(&child, state, sizeof(CState))
            c_apply(&child, actions[i])
            child_result = _c_alphabeta(&child, max_player, alpha, beta,
                                         depth + 1, max_depth)
            if child_result.value > value:
                value = child_result.value
                result.best_action = actions[i]
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        result.value = value
    else:
        value = 2.0
        for i in range(n_actions):
            memcpy(&child, state, sizeof(CState))
            c_apply(&child, actions[i])
            child_result = _c_alphabeta(&child, max_player, alpha, beta,
                                         depth + 1, max_depth)
            if child_result.value < value:
                value = child_result.value
                result.best_action = actions[i]
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        result.value = value

    return result


# ── Python bridge ─────────────────────────────────────────────

# Card mapping tables: populated once at import time
_py_card_to_id = {}
_py_id_to_card = {}
_py_color_to_int = {}
_py_int_to_color = {}


def _init_mappings():
    from trickster.games.snapszer.cards import Card, Color, ALL_COLORS, RANK_VALUES
    for ci, col in enumerate(ALL_COLORS):
        _py_color_to_int[col] = ci
        _py_int_to_color[ci] = col
        for ri, rv in enumerate(RANK_VALUES):
            card = Card(col, rv)
            cid = ci * 5 + ri
            _py_card_to_id[card] = cid
            _py_id_to_card[cid] = card

_init_mappings()


cdef CState _node_to_cstate(object node):
    """Convert a SnapszerNode to a CState.

    Bounds-clamps all array sizes so a buggy Python state can never
    cause an out-of-bounds write into the fixed-size C arrays.
    """
    cdef CState s
    gs = node.gs
    cdef int i

    # Hands — clamp to MAX_HAND (5) for safety
    s.hand0_n = min(len(gs.hands[0]), MAX_HAND)
    for i in range(s.hand0_n):
        s.hand0[i] = _py_card_to_id[gs.hands[0][i]]
    s.hand1_n = min(len(gs.hands[1]), MAX_HAND)
    for i in range(s.hand1_n):
        s.hand1[i] = _py_card_to_id[gs.hands[1][i]]

    s.scores[0] = gs.scores[0]
    s.scores[1] = gs.scores[1]
    s.leader = gs.leader
    s.trick_no = gs.trick_no
    s.trump_color = _py_color_to_int[gs.trump_color]
    s.talon_closed = 1 if gs.talon_closed else 0
    s.talon_closed_by = gs.talon_closed_by if gs.talon_closed_by is not None else -1
    s.talon_close_any_zero_tricks = 1 if gs.talon_close_any_zero_tricks else 0
    s.captured_n[0] = len(gs.captured[0])
    s.captured_n[1] = len(gs.captured[1])

    if node.pending_lead is not None:
        s.pending_lead = _py_card_to_id[node.pending_lead]
    else:
        s.pending_lead = EMPTY

    # Draw pile — clamp to array size (10) for safety
    dp = list(gs.draw_pile)
    s.draw_pile_n = min(len(dp), 10)
    for i in range(s.draw_pile_n):
        s.draw_pile[i] = _py_card_to_id[dp[i]]

    if gs.trump_card is not None:
        s.trump_card = _py_card_to_id[gs.trump_card]
    else:
        s.trump_card = EMPTY

    if gs.pending_marriage is not None:
        _, suit, pts = gs.pending_marriage
        s.pending_marriage_suit = _py_color_to_int[suit]
        s.pending_marriage_pts = pts
    else:
        s.pending_marriage_suit = -1
        s.pending_marriage_pts = 0

    return s


cdef object _action_to_py(int action):
    """Convert a C action int back to a Python Action."""
    if action == 99:
        return "close_talon"
    if action >= 100:
        suit = action - 100
        col = _py_int_to_color[suit]
        return f"marry_{col.value}"
    return _py_id_to_card[action]


def c_alphabeta(node, game, int player):
    """Drop-in replacement for minimax.alphabeta.

    Returns (value, best_action) — same interface as the Python version.
    """
    cdef CState s = _node_to_cstate(node)
    cdef ABResult r
    with nogil:
        r = _c_alphabeta(&s, player, -2.0, 2.0, 0, 30)
    if r.best_action == EMPTY:
        return r.value, None
    return r.value, _action_to_py(r.best_action)


def c_pimc_minimax(node, game, int player, int n_samples=20, rng=None):
    """Drop-in replacement for minimax.pimc_minimax.

    Returns (best_action, avg_value).
    """
    import random as _random
    if rng is None:
        rng = _random.Random()

    actions = game.legal_actions(node)
    if len(actions) <= 1:
        return (actions[0] if actions else None), 0.0

    # We need determinize from the game object (Python-level),
    # but run alphabeta in C for each sample.
    cdef dict action_wins = {a: 0 for a in actions}
    cdef dict action_value = {a: 0.0 for a in actions}

    cdef CState s
    cdef ABResult r
    cdef int i

    for i in range(n_samples):
        det = game.determinize(node, player, rng)
        s = _node_to_cstate(det)
        with nogil:
            r = _c_alphabeta(&s, player, -2.0, 2.0, 0, 30)
        if r.best_action != EMPTY:
            best_py = _action_to_py(r.best_action)
            if best_py in action_wins:
                action_wins[best_py] += 1
                action_value[best_py] += r.value

    best = max(actions, key=lambda a: (action_wins[a], action_value[a]))
    n_best = action_wins[best] or 1
    avg_val = action_value[best] / n_best
    return best, avg_val
