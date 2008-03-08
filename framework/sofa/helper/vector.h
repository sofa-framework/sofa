/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_HELPER_VECTOR_H
#define SOFA_HELPER_VECTOR_H

#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdlib.h>

namespace sofa
{

namespace helper
{

//======================================================================
/**	Same as std::vector, + range checking on operator[ ]

 Range checking can be turned of using compile option -DNDEBUG
\author Francois Faure, 1999
*/
//======================================================================
template<
class T,
      class Alloc = std::allocator<T>
      >
class vector: public std::vector<T,Alloc>
{
public:

    /// size_type
    typedef typename std::vector<T,Alloc>::size_type size_type;
    /// reference to a value (read-write)
    typedef typename std::vector<T,Alloc>::reference reference;
    /// const reference to a value (read only)
    typedef typename std::vector<T,Alloc>::const_reference const_reference;

    /// Basic onstructor
    vector() : std::vector<T,Alloc>() {}
    /// Constructor
    vector(size_type n, const T& value): std::vector<T,Alloc>(n,value) {}
    /// Constructor
    vector(int n, const T& value): std::vector<T,Alloc>(n,value) {}
    /// Constructor
    vector(long n, const T& value): std::vector<T,Alloc>(n,value) {}
    /// Constructor
    explicit vector(size_type n): std::vector<T,Alloc>(n) {}
    /// Constructor
    vector(const std::vector<T, Alloc>& x): std::vector<T,Alloc>(x) {}
    /// Constructor
    vector<T, Alloc>& operator=(const std::vector<T, Alloc>& x)
    {
        std::vector<T,Alloc>::operator = (x);
        return (*this);
    }

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    vector(InputIterator first, InputIterator last): std::vector<T,Alloc>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    vector(typename vector<T,Alloc>::const_iterator first, typename vector<T,Alloc>::const_iterator last): std::vector<T,Alloc>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */


/// Read/write random access
    reference operator[](size_type n)
    {
#ifndef NDEBUG
        assert( n<this->size() );
#endif
        return *(this->begin() + n);
    }

/// Read-only random access
    const_reference operator[](size_type n) const
    {
#ifndef NDEBUG
        assert( n<this->size() );
#endif
        return *(this->begin() + n);
    }

    std::ostream& write(std::ostream& os) const
    {
        if( this->size()>0 )
        {
            for( unsigned int i=0; i<this->size()-1; ++i ) os<<(*this)[i]<<" ";
            os<<(*this)[this->size()-1];
        }
        return os;
    }

    std::istream& read(std::istream& in)
    {
        T t;
        this->clear();
        while(in>>t)
        {
            this->push_back(t);
        }
        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

/// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const vector<T,Alloc>& vec )
    {
        return vec.write(os);
    }

/// Input stream
    inline friend std::istream& operator>> ( std::istream& in, vector<T,Alloc>& vec )
    {
        return vec.read(in);
    }

    /// Sets every element to 'value'
    void fill( const T& value )
    {
        std::fill( this->begin(), this->end(), value );
    }

};

/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B
template<>
inline std::istream& vector<int, std::allocator<int> >::read( std::istream& in )
{
    int t;
    this->clear();
    std::string s;
    while(in>>s)
    {
        std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            t = atoi(s.c_str());
            this->push_back(t);
        }
        else
        {
            std::string s1(s,0,hyphen);
            std::string s2(s,hyphen+1);
            int t1,t2;
            t1 = atoi(s1.c_str());
            t2 = atoi(s2.c_str());
            std::cout << s << " = "<<t1 << " -> " << t2 << std::endl;
            if (t1<=t2)
                for (t=t1; t<=t2; ++t)
                    this->push_back(t);
            else
                for (t=t1; t>=t2; --t)
                    this->push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B
template<>
inline std::istream& vector<unsigned int, std::allocator<unsigned int> >::read( std::istream& in )
{
    unsigned int t;
    this->clear();
    std::string s;
    while(in>>s)
    {
        std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            t = atoi(s.c_str());
            this->push_back(t);
        }
        else
        {
            std::string s1(s,0,hyphen);
            std::string s2(s,hyphen+1);
            unsigned int t1,t2;
            t1 = atoi(s1.c_str());
            t2 = atoi(s2.c_str());
            std::cout << s << " = "<<t1 << " -> " << t2 << std::endl;
            if (t1<=t2)
                for (t=t1; t<=t2; ++t)
                    this->push_back(t);
            else
                for (t=t1; t>=t2; --t)
                    this->push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}


// ======================  operations on standard vectors

// -----------------------------------------------------------
//
/*! @name vector class-related methods

*/
//
// -----------------------------------------------------------
//@{
/** Remove the first occurence of a given value.

The remaining values are shifted.
*/
template<class T1, class T2>
void remove( T1& v, const T2& elem )
{
    typename T1::iterator e = std::find( v.begin(), v.end(), elem );
    if( e != v.end() )
    {
        typename T1::iterator next = e;
        next++;
        for( ; next != v.end(); ++e, ++next )
            *e = *next;
    }
    v.pop_back();
}

/** Remove the first occurence of a given value.

The last value is moved to where the value was found, and the other values are not shifted.
*/
template<class T1, class T2>
void removeValue( T1& v, const T2& elem )
{
    typename T1::iterator e = std::find( v.begin(), v.end(), elem );
    if( e != v.end() )
    {
        *e = v.back();
        v.pop_back();
    }
}

/// Remove value at given index, replace it by the value at the last index, other values are not changed
template<class T, class TT>
void removeIndex( std::vector<T,TT>& v, size_t index )
{
#ifndef NDEBUG
    assert( 0<= static_cast<int>(index) && index <v.size() );
#endif
    v[index] = v.back();
    v.pop_back();
}



//@}

} // namespace helper

} // namespace sofa

/*
/// Output stream
template<class T, class Alloc>
  std::ostream& operator<< ( std::ostream& os, const std::vector<T,Alloc>& vec )
{
  if( vec.size()>0 ){
    for( unsigned int i=0; i<vec.size()-1; ++i ) os<<vec[i]<<" ";
    os<<vec[vec.size()-1];
  }
  return os;
}

/// Input stream
template<class T, class Alloc>
    std::istream& operator>> ( std::istream& in, std::vector<T,Alloc>& vec )
{
  T t;
  vec.clear();
  while(in>>t){
    vec.push_back(t);
  }
  if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
  return in;
}

/// Input a pair
template<class T, class U>
    std::istream& operator>> ( std::istream& in, std::pair<T,U>& pair )
{
  in>>pair.first>>pair.second;
  return in;
}

/// Output a pair
template<class T, class U>
    std::ostream& operator<< ( std::ostream& out, const std::pair<T,U>& pair )
{
  out<<pair.first<<" "<<pair.second;
  return out;
}
*/

#endif


