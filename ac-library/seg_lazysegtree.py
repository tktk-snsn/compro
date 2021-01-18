"""
segtree(ACL移植)の場合
    seg = SegTree(N, op, u_data)
    seg.initialize(arr)

lazysegtree(ACL移植)
    lazyseg = LazySegTree(N, op, u_data, composition, id_, op_merge)
    lazyseg.initialize(arr)

更新クエリで区間長が必要になる場合は、モノイドに区間長をもたせるより以下のようにすると定数倍速い
    * SegTree.__init__　及び SegTree.initialize のコメントアウト部分を解除する
    * mapping関数に区間長を第3引数に追加する
      [ex]
      def mapping(F, S, size):
          return F * size + S
"""


class SegTree(object):
    def __init__(self, N, op, u_data):
        self._n = N
        self.log = (N-1).bit_length()
        self.size = 1 << self.log

        self.op = op
        self.e = u_data

        self.data = [u_data] * (2 * self.size)
        # self.len = [1] * (2 * self.size)

    def _update(self, i):
        self.data[i] = self.op(self.data[i << 1], self.data[i << 1 | 1])

    def initialize(self, arr=None):
        """ segtreeをarrで初期化する。len(arr) == Nにすること """
        if arr:
            for i, a in enumerate(arr, self.size):
                self.data[i] = a
        for i in reversed(range(1, self.size)):
            self._update(i)
            # self.len[i] = self.len[i << 1] + self.len[i << 1 | 1]

    def update(self, p, x):
        """ data[p] = x とする (0-indexed)"""
        p += self.size
        self.data[p] = x
        for i in range(1, self.log + 1):
            self._update(p >> i)

    def get(self, p):
        """ data[p]を返す """
        return self.data[p + self.size]

    def prod(self, l, r):
        """
        op(data[l], data[l+1], ..., data[r-1])を返す (0-indexed)
        """
        sml = self.e
        smr = self.e
        l += self.size
        r += self.size

        while l < r:
            if l & 1:
                sml = self.op(sml, self.data[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self.op(self.data[r], smr)
            l >>= 1
            r >>= 1
        return self.op(sml, smr)

    def all_prod(self):
        """ op(data[0], data[1], ... data[N-1])を返す """
        return self.data[1]

    def max_right(self, l, func):
        """
        func(l, l+1, ..., r-1) = True,
        func(l, l+1, ..., r-1, r) = Falseとなる r を返す
        """
        if l == self._n:
            return self._n
        l += self.size
        sm = self.e
        while True:
            while l % 2 == 0:
                l >>= 1
            if not func(self.op(sm, self.data[l])):
                while l < self.size:
                    l <<= 1
                    if func(self.op(sm, self.data[l])):
                        sm = self.op(sm, self.data[l])
                        l += 1
                return l - self.size
            sm = self.op(sm, self.data[l])
            l += 1
            if (l & -l) == l:
                break
        return self._n

    def min_left(self, r, func):
        """
        func(     l, l+1, ..., r-1) = True,
        func(l-1, l, l+1, ..., r-1) = Falseとなる l を返す
        """
        if r == 0:
            return 0
        r += self.size
        sm = self.e
        while True:
            r -= 1
            while r > 1 and r & 1:
                r >>= 1
            if not func(self.op(self.data[r], sm)):
                while r < self.size:
                    r = r << 1 | 1
                    if func(self.op(self.data[r], sm)):
                        sm = self.op(self.data[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.op(self.data[r], sm)
            if (r & -r) == r:
                break
        return 0


class LazySegTree(SegTree):
    def __init__(self, N, op, u_data, composition, id_, op_merge):
        super().__init__(N, op, u_data)
        self.composition = composition
        self.mapping = op_merge
        self.id = id_

        self.lazy = [id_] * self.size

    def _all_apply(self, i, F):
        # self.data[i] = self.mapping(F, self.data[i], self.len[i])
        self.data[i] = self.mapping(F, self.data[i])
        if i < self.size:
            self.lazy[i] = self.composition(F, self.lazy[i])

    def _push(self, i):
        self._all_apply(i << 1, self.lazy[i])
        self._all_apply(i << 1 | 1, self.lazy[i])
        self.lazy[i] = self.id

    def update(self, p, x):
        """ data[p] = x とする (0-indexed)"""
        p += self.size
        for i in reversed(range(1, self.log + 1)):
            self._push(p >> i)
        self.data[p] = x
        for i in range(1, self.log + 1):
            self._update(p >> i)

    def apply(self, p, F):
        """ data[p]にFを作用させる(data[p] = op_merge(F, data[p])とする, 0-indexed) """
        p += self.size
        for i in reversed(range(1, self.log + 1)):
            self._push(p >> i)
        # self.data[p] = self.mapping(F, self.data[p], self.len[p])
        self.data[p] = self.mapping(F, self.data[p])
        for i in range(1, self.log + 1):
            self._update(p >> i)

    def range_apply(self, l, r, F):
        """ i = l, l+1, ..., r-1 について、Fを作用させる(op_merge(F, data[i]), 0-indexed) """
        if l == r:
            return

        l += self.size
        r += self.size
        for i in reversed(range(1, self.log + 1)):  # too->down
            if ((l >> i) << i) != l:
                self._push(l >> i)
            if ((r >> i) << i) != r:
                self._push((r - 1) >> i)

        l2, r2 = l, r
        while l < r:
            if l & 1:
                self._all_apply(l, F)
                l += 1
            if r & 1:
                r -= 1
                self._all_apply(r, F)
            l >>= 1
            r >>= 1
        l, r = l2, r2

        for i in range(1, self.log + 1):
            if ((l >> i) << i) != l:
                self._update(l >> i)
            if ((r >> i) << i) != r:
                self._update((r - 1) >> i)

    def get(self, p):
        """ data[p]を返す """
        p += self.size
        for i in reversed(range(1, self.log + 1)):
            self._push(p >> i)
        return self.data[p]

    def prod(self, l, r):
        """
        op(data[l], data[l+1], ..., data[r-1])を返す (0-indexed)
        l == rの時は単位元u_dataを返す
        """
        if l == r:
            return self.e

        l += self.size
        r += self.size
        for i in reversed(range(1, self.log + 1)):
            if ((l >> i) << i) != l:
                self._push(l >> i)
            if ((r >> i) << i) != r:
                self._push(r >> i)

        sml = self.e
        smr = self.e
        while l < r:
            if l & 1:
                sml = self.op(sml, self.data[l])
                l += 1
            if r & 1:
                r -= 1
                smr = self.op(self.data[r], smr)
            l >>= 1
            r >>= 1
        return self.op(sml, smr)

    def max_right(self, l, func):
        """
        func(l, l+1, ..., r-1) = True,
        func(l, l+1, ..., r-1, r) = Falseとなる r を返す
        """
        if l == self._n:
            return self._n
        l += self.size
        for i in reversed(range(1, self.log + 1)):
            self._push(l >> i)

        sm = self.e
        while True:
            while l % 2 == 0:
                l >>= 1
            if not func(self.op(sm, self.data[[l]])):
                while l < self.size:
                    self._push(l)
                    l <<= 1
                    if func(self.op(sm, self.data[l])):
                        sm = self.op(sm, self.data[l])
                        l += 1
                return l - self.size
            sm = self.op(sm, self.data[l])
            l += 1
            if (l & -l) == l:
                break
        return self._n

    def min_left(self, r, func):
        """
        func(     l, l+1, ..., r-1) = True,
        func(l-1, l, l+1, ..., r-1) = Falseとなる l を返す
        """
        if r == 0:
            return 0
        r += self.size
        for i in reversed(range(1, self.log + 1)):
            self._push((r - 1) >> i)
        sm = self.e
        while True:
            r -= 1
            while r > 1 and r & 1:
                r >>= 1
            if not func(self.op(self.data[r], sm)):
                while r < self.size:
                    self._push(r)
                    r = r << 1 | 1
                    if func(self.op(self.data[r], sm)):
                        sm = self.op(self.data[r], sm)
                        r -= 1
                return r + 1 - self.size
            sm = self.op(self.data[r], sm)
            if (r & -r) == r:
                break
        return 0


if __name__ == "__main__":
    "https://atcoder.jp/contests/abc185/tasks/abc185_f"
    from operator import xor
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    seg = SegTree(N, xor, 0)
    seg.initialize(A)
    for _ in range(Q):
        t, x, y = map(int, input().split())
        if t == 1:
            a = seg.get(x - 1) ^ y
            seg.update(x - 1, a)
        else:
            print(seg.prod(x - 1, y))
