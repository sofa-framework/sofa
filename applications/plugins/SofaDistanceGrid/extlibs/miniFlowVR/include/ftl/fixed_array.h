/******* COPYRIGHT ************************************************
*                                                                 *
*                             FlowVR                              *
*                       Template Library                          *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 20054 by                                          *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU LGPL, please refer to the     *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: include/ftl/fixed_array.h                                 *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
/* The following code declares class array,
 * an STL container (as wrapper) for arrays of constant size.
 *
 * See
 *      http://www.josuttis.com/cppcode
 * for details and the latest version.
 *
 * (C) Copyright Nicolai M. Josuttis 2001.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 *
 * 29 Jun 2005 - remove boost includes and reverse iterators. (Jeremie Allard)
 * 23 Aug 2002 - fix for Non-MSVC compilers combined with MSVC libraries.
 * 05 Aug 2001 - minor update (Nico Josuttis)
 * 20 Jan 2001 - STLport fix (Beman Dawes)
 * 29 Sep 2000 - Initial Revision (Nico Josuttis)
 */

// See http://www.boost.org/libs/array for Documentation.

#ifndef FTL_FIXED_ARRAY_H
#define FTL_FIXED_ARRAY_H

#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <algorithm>

namespace ftl
{

template<class T, std::size_t N>
class fixed_array {
public:
  T elems[N];    // fixed-size array of elements of type T
  
public:
  // type definitions
  typedef T              value_type;
  typedef T*             iterator;
  typedef const T*       const_iterator;
  typedef T&             reference;
  typedef const T&       const_reference;
  typedef std::size_t    size_type;
  typedef std::ptrdiff_t difference_type;
  
  // iterator support
  iterator begin() { return elems; }
  const_iterator begin() const { return elems; }
  iterator end() { return elems+N; }
  const_iterator end() const { return elems+N; }
  
  // operator[]
  reference operator[](size_type i) { return elems[i]; }
  const_reference operator[](size_type i) const { return elems[i]; }
  
  // at() with range check
  reference at(size_type i) { rangecheck(i); return elems[i]; }
  const_reference at(size_type i) const { rangecheck(i); return elems[i]; }
  
  // front() and back()
  reference front() { return elems[0]; }
  const_reference front() const { return elems[0]; }
  reference back() { return elems[N-1]; }
  const_reference back() const { return elems[N-1]; }
  
  // size is constant
  static size_type size() { return N; }
  static bool empty() { return false; }
  static size_type max_size() { return N; }
  enum { static_size = N };
  
  // swap (note: linear complexity)
  void swap (fixed_array<T,N>& y)
  {
    std::swap_ranges(begin(),end(),y.begin());
  }
  
  // direct access to data
  const T* data() const { return elems; }
  
  // assignment with type conversion
  template <typename T2>
  fixed_array<T,N>& operator= (const fixed_array<T2,N>& rhs)
  {
    std::copy(rhs.begin(),rhs.end(), begin());
    return *this;
  }
  
  // assign one value to all elements
  void assign (const T& value)
  {
    std::fill_n(begin(),size(),value);
  }
  
private:

  // check range (may be private because it is static)
  static void rangecheck (size_type i) {
    if (i >= size()) { throw std::range_error("fixed_array"); }
  }
  
};

// comparisons
template<class T, std::size_t N>
bool operator== (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
  return std::equal(x.begin(), x.end(), y.begin());
}
template<class T, std::size_t N>
bool operator< (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
  return std::lexicographical_compare(x.begin(),x.end(),y.begin(),y.end());
}
template<class T, std::size_t N>
bool operator!= (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
  return !(x==y);
}
template<class T, std::size_t N>
bool operator> (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
  return y<x;
}
template<class T, std::size_t N>
bool operator<= (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
  return !(y<x);
}
template<class T, std::size_t N>
bool operator>= (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
  return !(x<y);
}

// global swap()
template<class T, std::size_t N>
inline void swap (fixed_array<T,N>& x, fixed_array<T,N>& y)
{
  x.swap(y);
}

} // namespace ftl

#endif
