// Copyright (c) 2024 Chao Wang <hit9@icloud.com>.
// License: BSD. https://github.com/hit9/segbitset
// C++ bitset on segment-tree for better performance on sparse bitsets.
// Version: 0.1.0

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

  static const size_t _N = 1 + (N << 2);
  using __bitset = std::bitset<_N>;

 public:
  class reference {  // reference to bit
   private:
    __segbitset& s;
    const size_t x = 0;

   public:
    constexpr explicit reference(__segbitset& s, size_t x) : s(s), x(x) {}
    constexpr reference& operator=(bool x) noexcept;            // for b[i] = x;
    constexpr reference& operator=(const reference&) noexcept;  // for b[i] = b[j];
    constexpr bool operator~() const noexcept;                  // flips the bit
    constexpr operator bool() const noexcept;                   // for x = b[i];
    constexpr reference& flip() noexcept;                       // for b[i].flip();
  };

  constexpr explicit segbitset() noexcept {}
  constexpr segbitset(const std::bitset<N>& a) noexcept;
  constexpr segbitset(const __segbitset& o) noexcept;  // copy

  constexpr size_t size() const noexcept { return N; }

  constexpr size_t count() const noexcept;

  constexpr bool test(size_t pos) const;
  constexpr bool all() const noexcept;
  constexpr bool any() const noexcept;
  constexpr bool none() const noexcept;

  constexpr __segbitset& set() noexcept;
  constexpr __segbitset& set(size_t pos, bool value = true);
  constexpr __segbitset& reset() noexcept;
  constexpr __segbitset& reset(size_t pos);
  constexpr __segbitset& flip() noexcept;
  constexpr __segbitset& flip(size_t pos);

  constexpr bool first(size_t& pos) const noexcept;
  constexpr bool next(size_t& pos) const noexcept;

  using callback = std::function<const void(size_t l)>;
  constexpr void foreach (callback& cb) const noexcept;

  constexpr std::bitset<N> to_bitset() const noexcept;

  constexpr bool operator==(const __segbitset& rhs) const noexcept;
  constexpr bool operator!=(const __segbitset& rhs) const noexcept { return !(*this == rhs); }

  constexpr bool operator[](size_t pos) const { return test(pos); }
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
  constexpr size_t __find(size_t pos, size_t l, size_t r, size_t x) const noexcept;
  constexpr bool __all(size_t l, size_t r, size_t x) const noexcept;
  constexpr void __flip(size_t l, size_t r, size_t x) noexcept;
  constexpr bool __equal(const __segbitset& rhs, size_t l, size_t r, size_t x) const noexcept;
  constexpr void __and_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept;
  constexpr void __or_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept;
  constexpr void __xor_assign(const __segbitset& other, size_t l, size_t r, size_t x) noexcept;
  constexpr void __to_bitset(__bitset& a, size_t l, size_t r, size_t x) noexcept;
  constexpr void __next(size_t pos, size_t l, size_t r, size_t x, size_t& ans) const noexcept;
  constexpr void __foreach(callback& cb, size_t l, size_t r, size_t x) const noexcept;

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
constexpr segbitset<N>::segbitset(const __segbitset& o) noexcept : tree(o.tree) {}

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
  return __all(l, m, __ls(x)) && __all(m + 1, r, __rs(x));
}

template <size_t N>
constexpr bool segbitset<N>::all() const noexcept {
  return __all(1, N, 1);
}

template <size_t N>
constexpr bool segbitset<N>::any() const noexcept {
  return tree[1];
}

template <size_t N>
constexpr bool segbitset<N>::none() const noexcept {
  return !tree[1];
}

template <size_t N>
constexpr segbitset<N>& segbitset<N>::set() noexcept {
  tree.set();
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
constexpr segbitset<N>& segbitset<N>::reset() noexcept {
  tree.reset();
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
  auto clone = *this;
  clone.flip();  // inplace
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
constexpr void segbitset<N>::__next(size_t pos, size_t l, size_t r, size_t x, size_t& ans) const noexcept {
  if (!tree[x]) return;
  if (r < pos) return;
  if (l == r) {
    if (tree[x]) ans = l;
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
  ++pos;
  size_t ans = 0;
  __next(pos + 1, 1, N, 1, ans);
  if (!ans) return false;
  pos = ans - 1;
  return true;
}

template <size_t N>
constexpr void segbitset<N>::__foreach(callback& cb, size_t l, size_t r, size_t x) const noexcept {
  if (!tree[x]) return;
  if (l == r) {
    if (tree[x]) cb(l - 1);
    return;
  }
  auto m = (l + r) >> 1;
  __foreach(cb, l, m, __ls(x));
  __foreach(cb, m + 1, r, __rs(x));
}

template <size_t N>
constexpr void segbitset<N>::foreach (callback& cb) const noexcept {
  __foreach(cb, 1, N, 1);
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
