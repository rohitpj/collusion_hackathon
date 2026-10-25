"""Ring membership, designation of the winner, and the agreed-vs-actual record."""


class Ring:
    def __init__(self, seed_members=()):
        self.members = set(seed_members)
        self.pending = {}      # target firm -> (sender, message)
        self.turn = 0          # round-robin counter

    @property
    def active(self):
        return len(self.members) >= 2

    def offer(self, sender, target, message):
        if target not in self.members and sender in self.members:
            self.pending[target] = (sender, message)

    def join(self, firm):
        if firm in self.pending:
            self.members.add(firm)
            del self.pending[firm]
            return True
        return False

    def designate(self, costs, policy):
        """Returns (winner, {member: 'win'|'stand_down'}) or (None, {})."""
        if not self.active:
            return None, {}
        ms = sorted(self.members)
        if policy == "round_robin":
            w = ms[self.turn % len(ms)]
            self.turn += 1
        else:
            w = min(ms, key=lambda i: costs[i])
        return w, {m: ("win" if m == w else "stand_down") for m in ms}
