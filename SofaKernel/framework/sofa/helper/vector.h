/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_VECTOR_H
#define SOFA_HELPER_VECTOR_H

#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include <cstdio>

#include <sofa/helper/helper.h>
#include <sofa/helper/MemoryManager.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/helper/logging/Messaging.h>

#if !defined(NDEBUG) && !defined(SOFA_NO_VECTOR_ACCESS_FAILURE)
#if !defined(SOFA_VECTOR_ACCESS_FAILURE)
#define SOFA_VECTOR_ACCESS_FAILURE
#endif
#endif

namespace sofa
{

namespace helper
{

void SOFA_HELPER_API vector_access_failure(const void* vec, unsigned size, unsigned i, const std::type_info& type);

/// Convert the string 's' into an unsigned int. The error are reported in msg & numErrors
/// is incremented.
int SOFA_HELPER_API getInteger(const std::string& s, std::stringstream& msg, unsigned int& numErrors) ;

/// Convert the string 's' into an unsigned int. The error are reported in msg & numErrors
/// is incremented.
unsigned int SOFA_HELPER_API getUnsignedInteger(const std::string& s, std::stringstream& msg, unsigned int& numErrors) ;


template <class T, class MemoryManager = CPUMemoryManager<T> >
class vector;

/// Regular vector
/// Using CPUMemoryManager, it has the same behavior as std::helper with extra conveniences:
///  - string serialization (making it usable in Data)
///  - operator[] is checking if the index is within the bounds in debug
template <class T>
class SOFA_HELPER_API vector<T, CPUMemoryManager<T> > : public std::vector<T, std::allocator<T> >
{
public:
    typedef CPUMemoryManager<T> memory_manager;
    typedef std::allocator<T> Alloc;
    /// size_type
    typedef typename std::vector<T,Alloc>::size_type size_type;
    /// reference to a value (read-write)
    typedef typename std::vector<T,Alloc>::reference reference;
    /// const reference to a value (read only)
    typedef typename std::vector<T,Alloc>::const_reference const_reference;

    template<class T2> struct rebind
    {
        typedef vector< T2,CPUMemoryManager<T2> > other;
    };

    /// Basic constructor
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
    /// Brace initalizer constructor
    vector(const std::initializer_list<T>& t) : std::vector<T,Alloc>(t) {}
    /// Move constructor
    vector(std::vector<T,Alloc>&& v): std::vector<T,Alloc>(std::move(v)) {}
    /// Copy operator
    vector<T, Alloc>& operator=(const std::vector<T, Alloc>& x)
    {
        std::vector<T,Alloc>::operator=(x);
        return *this;
    }
    /// Move assignment operator
    vector<T, Alloc>& operator=(std::vector<T,Alloc>&& v)
    {
        std::vector<T,Alloc>::operator=(std::move(v));
        return *this;
    }


#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    vector(InputIterator first, InputIterator last): std::vector<T,Alloc>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    vector(typename vector<T>::const_iterator first, typename vector<T>::const_iterator last): std::vector<T>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */


#ifdef SOFA_VECTOR_ACCESS_FAILURE

    /// Read/write random access
    reference operator[](size_type n)
    {
        if (n>=this->size())
            vector_access_failure(this, this->size(), n, typeid(T));
        //assert( n<this->size() );
        return *(this->begin() + n);
    }

    /// Read-only random access
    const_reference operator[](size_type n) const
    {
        if (n>=this->size())
            vector_access_failure(this, this->size(), n, typeid(T));
        //assert( n<this->size() );
        return *(this->begin() + n);
    }

#endif // SOFA_VECTOR_ACCESS_FAILURE


    size_t readFromPythonRepr(std::istream& in, std::ostream& errstr=sofa::helper::logging::MessageDispatcher::null()) ;
    size_t readFromSofaRepr(std::istream& in, std::ostream& errstr=sofa::helper::logging::MessageDispatcher::null()) ;
    size_t read(std::istream& in, std::ostream& errstr=sofa::helper::logging::MessageDispatcher::null()) ;

    void writeToPythonRepr(std::ostream& in) const ;
    void writeToSofaRepr(std::ostream& in) const ;
    void write(std::ostream& in) const ;

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const vector<T>& vec )
    {
        vec.write(os);
        return os;
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, vector<T>& vec )
    {
        vec.read(in) ;
        return in ;
    }

    /// Sets every element to 'value'
    void fill( const T& value )
    {
        std::fill( this->begin(), this->end(), value );
    }

    /// this function is usefull for vector_device because it resize the vector without device operation (if device is not valid).
    /// Therefore the function is used in asynchronous code to safly resize a vector which is either cuda of helper::vector
    void fastResize(size_type n) {
        this->resize(n);
    }

    using std::vector<T>::size ;
    using std::vector<T>::clear ;
};

template<class T> inline
size_t vector<T>::read(std::istream& in, std::ostream& errstr)
{
    std::streampos pos = in.tellg();
    char c;

    if (!(in >> c) || in.eof())
        return 0; // empty stream

    in.seekg(pos); // coming-back to the previous character
    if (c == '[') {
        return this->readFromPythonRepr(in, errstr);
    }

    return this->readFromSofaRepr(in, errstr);
}


/// Input stream
/// Generic version
template<class T> inline
size_t vector<T>::readFromSofaRepr( std::istream& in, std::ostream& )
{
    T t=T();
    clear();
    while(in>>t)
    {
        this->push_back(t);
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return size();
}

/// Input stream
/// Generic version
template<class T> inline
size_t vector<T>::readFromPythonRepr( std::istream& in, std::ostream& errmsg )
{
    clear();

    char c;
    in >> c;
    if( in.eof() ) // empty stream
        return 0;
    if ( c != '[' )
    {
        errmsg << "Bad begin character. Found '" << c << "' while expecting '['";
        in.setstate(std::ios::failbit);
        return size();
    }
    std::streampos pos = in.tellg();
    in >> c;
    if( c == ']' ) // empty vector
    {
        return size();
    }
    else
    {
        T t=T();
        in.seekg( pos ); // coming-back to previous character
        c = ' ';
        while( (in >> t) && (in >> c))
        {
            this->push_back ( t );
            if (c!=',')
                break;
        }
        if ( c != ']' ) {
            errmsg << "Bad end character. Found '" << c << "' while expecting ']'";
            in.setstate(std::ios::failbit);
        }
        if (in.eof())
            in.clear(std::ios::eofbit);
        if (in.fail())
            errmsg << "Error reading list with comma separated values. ";
        return size();
    }
    return size();
}

/// READING SPECIALIZATION
template<> size_t vector<std::string>::readFromSofaRepr( std::istream& in, std::ostream& o );
template<> size_t vector<unsigned int>::readFromSofaRepr( std::istream& in, std::ostream& errup) ;
template<> size_t vector<int>::readFromSofaRepr( std::istream& in, std::ostream& ) ;
template<> size_t vector<unsigned char>::readFromSofaRepr(std::istream& in, std::ostream& ) ;

template<> size_t vector<std::string>::readFromPythonRepr( std::istream& in, std::ostream& o );


/// WRITING SPECIALIZATION
template<class T>
inline void vector<T>::writeToSofaRepr( std::ostream& os ) const
{
    if( this->size()>0 )
    {
        for( size_type i=0; i<this->size()-1; ++i )
            os<<(*this)[i]<<" ";
        os<<(*this)[this->size()-1];
    }
}

template<> void vector<std::string>::writeToSofaRepr( std::ostream& in ) const ;
template<> void vector<unsigned char>::writeToSofaRepr(std::ostream& os) const ;

template<class T>
inline void vector<T>::writeToPythonRepr( std::ostream& os ) const
{
    if ( !this->empty() )
    {
        typename vector<T>::const_iterator i = this->begin();
        os << "[" << *i;
        ++i;
        for ( ; i!=this->end(); ++i )
            os << ", " << *i;
        os << "]";

    }
    else os << "[]"; // empty vector
    return;
}
template<> void vector<std::string>::writeToPythonRepr( std::ostream& in ) const ;
template<> void vector<unsigned char>::writeToPythonRepr(std::ostream& os) const ;

template<class T>
inline void vector<T>::write(std::ostream& os) const
{
    this->writeToSofaRepr(os) ;
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
        if (e != v.end()-1)
            *e = v.back();
        v.pop_back();
    }
}

/// Remove value at given index, replace it by the value at the last index, other values are not changed
template<class T, class TT>
void removeIndex( std::vector<T,TT>& v, size_t index )
{
#if defined(SOFA_VECTOR_ACCESS_FAILURE)
    //assert( 0<= static_cast<int>(index) && index <v.size() );
    if (index>=v.size())
        vector_access_failure(&v, v.size(), index, typeid(T));
#endif
    if (index != v.size()-1)
        v[index] = v.back();
    v.pop_back();
}


} // namespace helper
} // namespace sofa

#endif //SOFA_HELPER_VECTOR_H
