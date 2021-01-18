class fenwick_tree(object):
    def __init__(self, n):
        self.n = n
        self.log = n.bit_length()
        self.data = [0] * n

    def __sum(self, r):
        s = 0
        while r > 0:
            s += self.data[r - 1]
            r -= r & -r
        return s

    def add(self, p, x):
        """ a[p] += xを行う"""
        p += 1
        while p <= self.n:
            self.data[p - 1] += x
            p += p & -p

    def sum(self, l, r):
        """a[l] + a[l+1] + .. + a[r-1]を返す"""
        return self.__sum(r) - self.__sum(l)

    def lower_bound(self, x):
        """a[0] + a[1] + .. a[i] >= x となる最小のiを返す"""
        if x <= 0:
            return -1
        i = 0
        k = 1 << self.log
        while k:
            if i + k <= self.n and self.data[i + k - 1] < x:
                x -= self.data[i + k - 1]
                i += k
            k >>= 1
        return i
