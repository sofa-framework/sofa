#ifndef SOFA_COMPONENTS_COMMON_VECTOR_H
#define SOFA_COMPONENTS_COMMON_VECTOR_H

#include <vector>
#include <algorithm>
#include <cassert>

namespace Sofa
{
namespace Components
{
namespace Common
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

/// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const std::vector<T,Alloc>& vec )
    {
        if( vec.size()>0 )
        {
            for( unsigned int i=0; i<vec.size()-1; ++i ) os<<vec[i]<<" ";
            os<<vec[vec.size()-1];
        }
        return os;
    }

/// Input stream
    inline friend std::istream& operator>> ( std::istream& in, std::vector<T,Alloc>& vec )
    {
        T t;
        vec.clear();
        while(in>>t)
        {
            vec.push_back(t);
        }
        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

};

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
template<class T, class TT>
void remove( std::vector<T,TT>& v, const T& elem )
{
    typename vector<T>::iterator e = std::find( v.begin(), v.end(), elem );
    if( e != v.end() )
    {
        typename vector<T>::iterator next = e;
        next++;
        for( ; next != v.end(); ++e, ++next )
            *e = *next;
    }
    v.pop_back();
}

/** Remove the first occurence of a given value.

The last value is moved to where the value was found, and the other values are not shifted.
*/
template<class T, class TT>
void removeValue( std::vector<T,TT>& v, const T& elem )
{
    typename vector<T>::iterator e = std::find( v.begin(), v.end(), elem );
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




} // ==============================   end namespace Common
} // ==============================   end namespace Components
} // ==============================   end namespace Sofa



#endif


