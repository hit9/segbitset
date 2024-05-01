#include <iostream>

#include "segbitset.h"

static const std::size_t N = 1024 * 100;

auto test() {
  std::bitset<N> a;
  a.set(1);
  a.set(20);
  a.set(31);
  a.set(1341);
  segbitset::segbitset<N> s(a);
  return s;
}

consteval auto test1() {
  std::bitset<N> a;
  a.set(1);
  a.set(20);
  a.set(31);
  a.set(1341);
  segbitset::segbitset<N> s(a);
  std::size_t p;
  s.first(p);
  s.next(p);
  s.next(p);
  s.next(p);
  return p;
}

int main(void) {
  std::cout << "xbool: " << test1() << std::endl;

  auto x = test();
  std::cout << x[1] << std::endl;
  std::cout << x[0] << std::endl;
  std::size_t p;
  std::cout << "bool: " << x.first(p) << " pos:" << p << std::endl;
  std::cout << "bool: " << x.next(p) << " pos:" << p << std::endl;
  std::cout << "bool: " << x.next(p) << " pos:" << p << std::endl;
  std::cout << "bool: " << x.next(p) << " pos:" << p << std::endl;

  decltype(x)::callback cb = [](int pos) { std::cout << "pos of 1:" << pos << std::endl; };
  x.foreach (cb);
  return 0;
}
