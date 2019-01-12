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


    std::ostream& write(std::ostream& os) const
    {
        if( this->size()>0 )
        {
            for( size_type i=0; i<this->size()-1; ++i )
                os<<(*this)[i]<<" ";
            os<<(*this)[this->size()-1];
        }
        return os;
    }

    std::istream& read(std::istream& in)
    {
        T t=T();
        this->clear();
        while(in>>t)
        {
            this->push_back(t);
        }
        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const vector<T>& vec )
    {
        return vec.write(os);
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, vector<T>& vec )
    {
        return vec.read(in);
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

};

/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B, optionnally specifying a step using "A-B-step" notation.
template<> inline
std::istream& vector<int>::read( std::istream& in )
{
    int t;
    this->clear();
    std::string s;
    std::stringstream msg;
    unsigned int numErrors=0;

    /// Cut the input stream in words using the standard's '<space>' token eparator.
    while(in>>s)
    {
        /// Check if there is the sign '-' in the string s.
        std::string::size_type hyphen = s.find_first_of('-',1);

        /// If there is no '-' in s
        if (hyphen == std::string::npos)
        {
            /// Convert the word into an integer number.
            /// Use strtol because it reports error in case of parsing problem.
            t = getInteger(s, msg, numErrors) ;
            this->push_back(t);
        }

        /// If there is at least one '-'
        else
        {
            int t1,t2,tinc;
            std::string s1(s,0,hyphen);
            t1 = getInteger(s1, msg, numErrors) ;
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = getInteger(s2, msg, numErrors) ;
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                t2 =  getInteger(s2, msg, numErrors) ;
                tinc =  getInteger(s3, msg, numErrors) ;
                if (tinc == 0)
                {
                    tinc = (t1<t2) ? 1 : -1;
                    msg << "- Increment 0 is replaced by "<< tinc << msgendl;
                }
                if ((t2-t1)*tinc < 0)
                {
                    // increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }

            /// Go in backward order.
            if (tinc < 0)
                for (t=t1; t>=t2; t+=tinc)
                    this->push_back(t);
            /// Go in Forward order
            else
                for (t=t1; t<=t2; t+=tinc)
                    this->push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    if(numErrors!=0)
    {
        msg_warning("vector<int>") << "Unable to parse vector values:" << msgendl
                                   << msg.str() ;
    }
    return in;
}


/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B
template<> inline
std::istream& vector<unsigned int>::read( std::istream& in )
{
    std::stringstream errmsg ;
    unsigned int errcnt = 0 ;
    unsigned int t = 0 ;

    this->clear();
    std::string s;
    while(in>>s)
    {
        std::string::size_type hyphen = s.find_first_of('-',1);
        if (hyphen == std::string::npos)
        {
            t = getUnsignedInteger(s, errmsg, errcnt) ;
            this->push_back(t);
        }
        else
        {
            unsigned int t1,t2;
            int tinc;
            std::string s1(s,0,hyphen);
            t1 = getUnsignedInteger(s1, errmsg, errcnt) ;
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = getUnsignedInteger(s2, errmsg, errcnt);
                tinc = (t1<=t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2-hyphen-1);
                std::string s3(s,hyphen2+1);
                t2 = getUnsignedInteger(s2, errmsg, errcnt);
                tinc = getInteger(s3, errmsg, errcnt);
                if (tinc == 0)
                {
                    tinc = (t1<=t2) ? 1 : -1;
                    errmsg << "- problem while parsing '"<<s<<"': increment is 0. Use " << tinc << " instead." ;
                }
                if (((int)(t2-t1))*tinc < 0)
                {
                    /// increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }
            if (tinc < 0){
                for (t=t1; t>t2; t=t+tinc)
                    this->push_back(t);
                this->push_back(t2);
            } else {
                for (t=t1; t<=t2; t=t+tinc)
                    this->push_back(t);
            }
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    if(errcnt!=0)
    {
        msg_warning("vector<unsigned int>") << "Unable to parse values" << msgendl
                                            << errmsg.str() ;
    }

    return in;
}

/// Output stream
/// Specialization for writing vectors of unsigned char
template<> inline
std::ostream& vector<unsigned char>::write(std::ostream& os) const
{
    if( this->size()>0 )
    {
        for( size_type i=0; i<this->size()-1; ++i )
            os<<(int)(*this)[i]<<" ";
        os<<(int)(*this)[this->size()-1];
    }
    return os;
}

/// Input stream
/// Specialization for reading vectors of unsigned char
template<> inline
std::istream& vector<unsigned char>::read(std::istream& in)
{
    int t;
    this->clear();
    while(in>>t)
    {
        this->push_back((unsigned char)t);
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
