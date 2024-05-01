// Copyright (c) 2024 Chao Wang <hit9@icloud.com>.
// License: BSD. https://github.com/hit9/segbitset
// C++ bitset on segment-tree for better performance on sparse bitsets.
// Version: 0.1.0
//
// Tree structure schematic diagram::
//
//   [                  1                   ]     => root  -+
//   [        1         ][       0          ]               |--> OR summary of descendants
//   [   1    ][   0    ][   0    ][   0    ]              -+
//   [ 1 ][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ][ 0 ]     => bit data itself
//
// Core acceleration points:
//  1. any(),none() just reads root bit's value, O(1)
//  2. quickly skip subtrees that are all 0, for and,or,xor operations.
//  3. find positions storing true bits faster for sparse bits.
//
// Tradeoffs:
//  1. set(pos),test(pos),flip(pos) now are slower than std::bitset, O(logN).
//  2. occupying x4 times space than an equivalent std::bitset.
//  3. shift operations and to_string/to_ulong aren't implementated yet.

#ifndef __HIT9_SEGBITSET
#define __HIT9_SEGBITSET

#include <bitset>
#include <cstddef>     // for size_t
#include <functional>  // for function
#include <stdexcept>

namespace segbitset {

using size_t = std::size_t;

template <size_t N>
class segbitset {
  using __segbitset = segbitset<N>;
  using __bitset = std::bitset<1 + (N << 2)>;

 public:
  class reference {  // reference to a bit
   private:
    __segbitset& s;
    const size_t x = 0;

   public:
    constexpr explicit reference(__segbitset& s, size_t x) : s(s), x(x) {}

    constexpr reference& operator=(bool value) noexcept;        // for b[i] = value;
    constexpr reference& operator=(const reference&) noexcept;  // for b[i] = b[j];
    constexpr bool operator~() const noexcept;                  // flips the bit
    constexpr operator bool() const noexcept;                   // for x = b[i];
    constexpr reference& flip() noexcept;                       // for b[i].flip();
  };

  constexpr explicit segbitset() noexcept {}
  // creates a segbitset from a std::bitset
  constexpr segbitset(const std::bitset<N>& a) noexcept;
  // copy constructor
  constexpr segbitset(const __segbitset& o) noexcept;

  // returns the number of bits that the bitset holds
  constexpr size_t size() const noexcept { return N; }
  // returns the number of bits set to true
  constexpr size_t count() const noexcept;

  // returns the value of the bit at the position pos (counting from 0).
  // throws std::out_of_range if pos is invalid.
  constexpr bool test(size_t pos) const;
  // checks if all of the bits are set to true
  constexpr bool all() const noexcept;
  // checks if any of the bits are set to true
  constexpr bool any() const noexcept;
  // checks if none of the bits are set to true
  constexpr bool none() const noexcept;
  // sets all bits to true.
  constexpr __segbitset& set() noexcept;
  // sets the bit at position pos to the given value.
  // throws std::out_of_range if pos is invalid.
  constexpr __segbitset& set(size_t pos, bool value = true);
  // sets all bits to false.
  constexpr __segbitset& reset() noexcept;
  // sets the bit at position pos to false.
  // throws std::out_of_range if pos is invalid.
  constexpr __segbitset& reset(size_t pos);
  // flips all bits (like operator~, but in-place).
  constexpr __segbitset& flip() noexcept;
  // flips the bit at the position pos.
  // throws std::out_of_range if pos is invalid.
  constexpr __segbitset& flip(size_t pos);
  // find the first position where stores a true bit, returns true if found.
  // The position found will be assigned to given parameter pos.
  constexpr bool first(size_t& pos) const noexcept;
  // find the next position where stores a true bit, returns true if found.
  // The position found will be assigned to given parameter pos.
  constexpr bool next(size_t& pos) const noexcept;
  // callback is a readonly function that receives a position as parameter.
  using callback = std::function<const void(size_t pos)>;
  // iterates all true bits from left to right and execute given callback function,
  // with the position of true bits as a argument.
  // foreach1 should be faster than first & next, since it dosen't require walking from root again.
  constexpr void foreach1(callback& cb) const noexcept;
  // constructs and returns an equivalent std::bitset from this segbitset.
  constexpr std::bitset<N> to_bitset() const noexcept;
  // fill given std::bitset as an equivalent of this segbitset.
  // the given bitset should be all zero in advance.
  constexpr void to_bitset(std::bitset<N>& a) noexcept;

  constexpr bool operator==(const __segbitset& rhs) const noexcept;  // for b == rhs;
  constexpr bool operator!=(const __segbitset& rhs) const noexcept { return !(*this == rhs); }
  // b[pos] returns the value of bit at position pos.
  constexpr bool operator[](size_t pos) const { return test(pos); }
  // b[pos] returns a reference to the bit at position pos.
  constexpr reference operator[](size_t pos);
  constexpr __segbitset& operator&=(const __segbitset& other) noexcept;  // for b &= other
  constexpr __segbitset& operator|=(const __segbitset& other) noexcept;  // for b |= other
  constexpr __segbitset& operator^=(const __segbitset& other) noexcept;  // for b ^= other
  constexpr __segbitset operator~() const noexcept;                      // ~b, returns a copy of flipped b

 private:
  __bitset tree;

  inline constexpr size_t __ls(size_t x) const { return x << 1; }
  inline constexpr size_t __rs(size_t x) const { return (x << 1) | 1; }
  constexpr void __pushup(size_t x) noexcept;
  constexpr void __pushup_to_root(size_t x) noexcept;
  constexpr void __build(const std::bitset<N>& a, size_t l, size_t r, size_t x) noexcept;
  constexpr size_t __count(size_t l, size_t r, size_t x) const noexcept;
  constexpr size_t __find(size_t pos, size_t l, size_t r, size_t x) const noexcept;  // helper
  constexpr bool __all(size_t l, size_t r, size_t x) const noexcept;
  constexpr void __flip(size_t l, size_t r, size_t x) noexcept;
  constexpr void __reset(size_t l, size_t r, size_t x) noexcept;
  constexpr bool __equal(const __segbitset& rhs, size_t l, size_t r, size_t x) const noexcept;
  constexpr void __copy(const __segbitset& o, size_t l, size_t r, size_t x) noexcept;
  constexpr void __and_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept;
  constexpr void __or_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept;
  constexpr void __xor_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept;
  constexpr void __to_bitset(__bitset& a, size_t l, size_t r, size_t x) noexcept;
  constexpr void __next(size_t pos, size_t l, size_t r, size_t x, size_t& ans) const noexcept;
  constexpr void __foreach1(callback& cb, size_t l, size_t r, size_t x) const noexcept;

  friend class reference;
};

////// Implementation ///////

template <size_t N>
constexpr void segbitset<N>::__pushup(size_t x) noexcept {
  tree[x] = tree[__ls(x)] | tree[__rs(x)];
}

template <size_t N>
constexpr void segbitset<N>::__pushup_to_root(size_t x) noexcept {
  while (x) {
    tree[x] = tree[__ls(x)] | tree[__rs(x)];
    x >>= 1;
  }
}

template <size_t N>
constexpr void segbitset<N>::__build(const std::bitset<N>& a, size_t l, size_t r, size_t x) noexcept {
  if (l == r) {
    tree[x] = a[l - 1];
    return;
  }
  auto m = (l + r) >> 1;
  __build(a, l, m, __ls(x));
  __build(a, m + 1, r, __rs(x));
  __pushup(x);
}

template <size_t N>
constexpr segbitset<N>::segbitset(const std::bitset<N>& a) noexcept {
  __build(a, 1, N, 1);
}

template <size_t N>
constexpr void segbitset<N>::__copy(const segbitset<N>& o, size_t l, size_t r, size_t x) noexcept {
  if (!tree[x] && !o.tree[x]) return;  // children of both are all 0, tree won't change.
  if (l == r) {
    tree[x] = o.tree[x];
    return;
  }
  auto m = (l + r) >> 1;
  __copy(o, l, m, __ls(x));
  __copy(o, r, m + 1, __rs(x));
  __pushup(x);
}

template <size_t N>
constexpr segbitset<N>::segbitset(const __segbitset& o) noexcept {
  // TODO: performance depends..
  // which is faster?
  // 1. copy o.tree directly, x4 time slower than a std::bitset<N> copy.
  // 2. __copy, but seems working not that fast.
  // 3. tree reset, and then |= other, seems a bit faster than 2.
  __copy(o, 1, N, 1);
}

template <size_t N>
constexpr size_t segbitset<N>::__count(size_t l, size_t r, size_t x) const noexcept {
  if (!tree[x]) return 0;
  if (l == r) return 1;
  auto m = (l + r) >> 1;
  return __count(l, m, __ls(x)) + __count(m + 1, r, __rs(x));
}

template <size_t N>
constexpr size_t segbitset<N>::count() const noexcept {
  return __count(1, N, 1);
}

template <size_t N>
constexpr size_t segbitset<N>::__find(size_t pos, size_t l, size_t r, size_t x) const noexcept {
  if (l == r) return x;
  auto m = (l + r) >> 1;
  if (pos <= m) return __find(pos, l, m, __ls(x));
  return __find(pos, m + 1, r, __rs(x));
}

template <size_t N>
constexpr bool segbitset<N>::test(size_t pos) const {
  if (pos >= N) throw std::out_of_range("segbitset::test pos >= N");
  return tree[__find(pos + 1, 1, N, 1)];
}

template <size_t N>
constexpr bool segbitset<N>::__all(size_t l, size_t r, size_t x) const noexcept {
  if (!tree[x]) return false;
  if (l == r) return tree[x];
  auto m = (l + r) >> 1;
  return __all(l, m, __ls(x)) && __all(m + 1, r, __rs(x));  // && has short circuit effect
}

template <size_t N>
constexpr bool segbitset<N>::all() const noexcept {
  return __all(1, N, 1);
}

template <size_t N>
constexpr bool segbitset<N>::any() const noexcept {
  return tree[1];  // root == 1 indicates the whole tree contains true bits
}

template <size_t N>
constexpr bool segbitset<N>::none() const noexcept {
  return !tree[1];  // root == 0 indicates the whole tree contains no true bits
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::set() noexcept {
  tree.set();  // TODO: any faster solution?
  return *this;
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::set(size_t pos, bool value) {
  if (pos >= N) throw std::out_of_range("segbitset::set pos >= N");
  auto x = __find(pos + 1, 1, N, 1);
  tree[x] = 1;
  __pushup_to_root(x >> 1);
  return *this;
}

template <size_t N>
constexpr void segbitset<N>::__reset(size_t l, size_t r, size_t x) noexcept {
  if (!tree[x]) return;  // won't change
  if (l == r) {
    tree[x] = 0;
    return;
  }
  auto m = (l + r) >> 1;
  __reset(l, m, __ls(x));
  __reset(m + 1, r, __rs(x));
  __pushup(x);
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::reset() noexcept {
  // Not using tree.reset() (aka std::bitset's)
  // for hoping better performance on sparse dataset.
  // TODO: performance
  __reset(1, N, 1);
  return *this;
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::reset(size_t pos) {
  if (pos >= N) throw std::out_of_range("segbitset::reset pos >= N");
  auto x = __find(pos + 1, 1, N, 1);
  tree[x] = 0;
  __pushup_to_root(x >> 1);
  return *this;
}

template <size_t N>
constexpr void segbitset<N>::__flip(size_t l, size_t r, size_t x) noexcept {
  if (l == r) {
    tree[x] = !tree[x];
    return;
  }
  auto m = (l + r) >> 1;
  __flip(l, m, __ls(x));
  __flip(m + 1, r, __rs(x));
  __pushup(x);
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::flip() noexcept {
  __flip(1, N, 1);
  return *this;
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::flip(size_t pos) {
  if (pos >= N) throw std::out_of_range("segbitset::flip pos >= N");
  auto x = __find(pos + 1, 1, N, 1);
  tree[x] = !tree[x];
  __pushup_to_root(x >> 1);
}

template <size_t N>
constexpr bool segbitset<N>::__equal(const segbitset<N>& rhs, size_t l, size_t r, size_t x) const noexcept {
  if (tree[x] != rhs.tree[x]) return false;
  if (!tree[x] && !rhs.tree[x]) return true;  // children of both are all 0.
  if (l == r) return tree[x] == rhs.tree[x];
  auto m = (l + r) >> 1;
  return __equal(rhs, l, m, __ls(x)) && __equal(rhs, r, m + 1, __rs(x));
}

template <size_t N>
constexpr bool segbitset<N>::operator==(const segbitset<N>& rhs) const noexcept {
  return __equal(rhs, 1, N, 1);
}

template <size_t N>
constexpr void segbitset<N>::__and_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept {
  if (l == r) {
    tree[x] = tree[x] & other.tree[x];
    return;
  }
  // 0 & 0 -> 0   *
  // 0 & 1 -> 0   *
  // 1 & 0 -> 0
  // 1 & 1 -> 1
  // if children of this tree are all zeros, results won't change.
  if (!tree[x]) return;
  auto m = (l + r) >> 1;
  __and_assign(other, l, m, __ls(x));
  __and_assign(other, m + 1, r, __rs(x));
  __pushup(x);
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::operator&=(const segbitset<N>& other) noexcept {
  __and_assign(other, 1, N, 1);
  return *this;
}

template <size_t N>
constexpr void segbitset<N>::__or_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept {
  if (l == r) {
    tree[x] = tree[x] | other.tree[x];
    return;
  }
  // 0 | 0 -> 0   *
  // 0 | 1 -> 1
  // 1 | 0 -> 1   *
  // 1 | 1 -> 1
  // if children of other are all of 0, results of this tree won't change.
  if (!other.tree[x]) return;
  auto m = (l + r) >> 1;
  __or_assign(other, l, m, __ls(x));
  __or_assign(other, m + 1, r, __rs(x));
  __pushup(x);
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::operator|=(const segbitset<N>& other) noexcept {
  __or_assign(other, 1, N, 1);
  return *this;
}

template <size_t N>
constexpr void segbitset<N>::__xor_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept {
  if (l == r) {
    tree[x] = tree[x] ^ other.tree[x];
    return;
  }
  // 0 ^ 0 -> 0    *
  // 0 ^ 1 -> 1
  // 1 ^ 0 -> 1    *
  // 1 ^ 1 -> 0
  // if children of other are all 0, tree won't change.
  if (!other.tree[x]) return;
  auto m = (l + r) >> 1;
  __xor_assign(other, l, m, __ls(x));
  __xor_assign(other, m + 1, r, __rs(x));
  __pushup(x);
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::operator^=(const __segbitset& other) noexcept {
  __xor_assign(other, 1, N, 1);
  return *this;
}

template <size_t N>
constexpr segbitset<N> segbitset<N>::operator~() const noexcept {
  decltype(*this) clone(*this);  // copy
  clone.flip();                  // inplace
  return clone;
}

template <size_t N>
constexpr void segbitset<N>::__to_bitset(__bitset& a, size_t l, size_t r, size_t x) noexcept {
  if (l == r) {
    a[l - 1] = tree[x];
    return;
  }

  auto m = (l + r) >> 1;
  __to_bitset(l, m, __ls(x));
  __to_bitset(m + 1, r, __rs(x));
}

template <size_t N>
constexpr std::bitset<N> segbitset<N>::to_bitset() const noexcept {
  std::bitset<N> a;
  __to_bitset(a, 1, N, 1);
  return a;
}

template <size_t N>
constexpr void segbitset<N>::to_bitset(std::bitset<N>& a) noexcept {
  __to_bitset(a, 1, N, 1);
}

template <size_t N>
constexpr void segbitset<N>::__next(size_t pos, size_t l, size_t r, size_t x, size_t& ans) const noexcept {
  if (!tree[x]) return;
  if (r < pos) return;  // skip previously scanned intervals.
  if (l == r) {
    ans = l;
    return;
  }
  auto m = (l + r) >> 1;
  if (!ans) __next(pos, l, m, __ls(x), ans);
  if (!ans) __next(pos, m + 1, r, __rs(x), ans);
}

template <size_t N>
constexpr bool segbitset<N>::first(size_t& pos) const noexcept {
  size_t ans = 0;
  __next(1, 1, N, 1, ans);
  if (!ans) return false;
  pos = ans - 1;
  return true;
}

template <size_t N>
constexpr bool segbitset<N>::next(size_t& pos) const noexcept {
  ++pos;  // excludes previous result
  size_t ans = 0;
  __next(pos + 1, 1, N, 1, ans);
  if (!ans) return false;
  pos = ans - 1;
  return true;
}

template <size_t N>
constexpr void segbitset<N>::__foreach1(callback& cb, size_t l, size_t r, size_t x) const noexcept {
  if (!tree[x]) return;
  if (l == r) {
    cb(l - 1);
    return;
  }
  auto m = (l + r) >> 1;
  __foreach1(cb, l, m, __ls(x));
  __foreach1(cb, m + 1, r, __rs(x));
}

template <size_t N>
constexpr void segbitset<N>::foreach1(callback& cb) const noexcept {
  __foreach1(cb, 1, N, 1);
}

////////////////////////////////////////
/// reference
////////////////////////////////////////

template <size_t N>
constexpr typename segbitset<N>::reference segbitset<N>::operator[](size_t pos) {
  if (pos >= N) throw std::out_of_range("segbitset::operator[] pos >= N");
  auto x = __find(pos + 1, 1, N, 1);
  return segbitset<N>::reference(*this, x);
}

template <size_t N>
constexpr typename segbitset<N>::reference& segbitset<N>::reference::operator=(bool value) noexcept {
  s.tree[x] = value;
  s.__pushup_to_root(x >> 1);
  return *this;
}

template <size_t N>
constexpr typename segbitset<N>::reference& segbitset<N>::reference::operator=(
    const segbitset<N>::reference& reference) noexcept {
  s.tree[x] = reference.s.tree[x];
  s.__pushup_to_root(x >> 1);
  return *this;
}

template <size_t N>
constexpr bool segbitset<N>::reference::operator~() const noexcept {
  auto b = s.tree[x] = !s.tree[x];
  s.__pushup_to_root(x >> 1);
  return b;
}

template <size_t N>
constexpr segbitset<N>::reference::operator bool() const noexcept {
  return s.tree[x];
}

template <size_t N>
constexpr typename segbitset<N>::reference& segbitset<N>::reference::flip() noexcept {
  s.tree[x] = !s.tree[x];
  return *this;
}

////////////////////////////////////////
/// Non member operators
////////////////////////////////////////

template <size_t N>
constexpr segbitset<N> operator&(const segbitset<N>& lhs, const segbitset<N>& rhs) noexcept {
  auto s = lhs;
  s &= rhs;
  return s;
}

template <size_t N>
constexpr segbitset<N> operator|(const segbitset<N>& lhs, const segbitset<N>& rhs) noexcept {
  auto s = lhs;
  s |= rhs;
  return s;
}

template <size_t N>
constexpr segbitset<N> operator^(const segbitset<N>& lhs, const segbitset<N>& rhs) noexcept {
  auto s = lhs;
  s ^= rhs;
  return s;
}

}  // namespace segbitset

#endif
