/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
