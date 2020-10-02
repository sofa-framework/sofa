#ifndef SOFA_HELPER_HASH_H
#define SOFA_HELPER_HASH_H

#include <functional>

/// to combine hashes (based on boost implementation)
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}


/// generic hash for pairs
namespace std
{
  template<typename S, typename T> struct hash<pair<S, T> >
  {
    inline size_t operator()(const pair<S, T> & v) const
    {
      size_t seed = 0;
      ::hash_combine(seed, v.first);
      ::hash_combine(seed, v.second);
      return seed;
    }
  };
}



#endif
