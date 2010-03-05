/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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
#include <stdlib.h>
#include <typeinfo>

#include <sofa/helper/helper.h>
#include <sofa/helper/MemoryManager.h>


namespace sofa
{

namespace defaulttype
{
template<class T>
class DataTypeInfo;
}

namespace helper
{

template <class T, class MemoryManager = CPUMemoryManager<T> >
class vector
{
public:
    typedef T      value_type;
    typedef size_t size_type;
    typedef T&     reference;
    typedef const T& const_reference;
    typedef T*     iterator;
    typedef const T* const_iterator;

protected:
    size_type     vectorSize;     ///< Current size of the vector
    size_type     allocSize;      ///< Allocated size
    mutable void* devicePointer;  ///< Pointer to the data on the GPU side
    T*            hostPointer;    ///< Pointer to the data on the CPU side
    mutable bool  deviceIsValid;  ///< True if the data on the GPU is currently valid
    mutable bool  hostIsValid;    ///< True if the data on the CPU is currently valid

public:

    vector()
        : vectorSize ( 0 ), allocSize ( 0 ), devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {}
    vector ( size_type n )
        : vectorSize ( 0 ), allocSize ( 0 ), devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {
        resize ( n );
    }
    vector ( const vector<T,MemoryManager >& v )
        : vectorSize ( 0 ), allocSize ( 0 ), devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {
        *this = v;
    }

    void clear()
    {
        vectorSize = 0;
        deviceIsValid = true;
        hostIsValid = true;
    }

    void operator= ( const vector<T,MemoryManager >& v )
    {
        if (&v == this)
        {
            //COMM : std::cerr << "ERROR: self-assignment of CudaVector< " << core::objectmodel::Base::decodeTypeName(typeid(T)) << ">"<<std::endl;
            return;
        }
        size_type newSize = v.size();
        clear();
        fastResize ( newSize );
        deviceIsValid = v.deviceIsValid;
        hostIsValid = v.hostIsValid;
        if ( vectorSize > 0 && deviceIsValid )
        {
            MemoryManager::memcpyDeviceToDevice ( devicePointer, v.devicePointer, vectorSize*sizeof ( T ) );
        }
        if ( vectorSize!=0 && hostIsValid )
            std::copy ( v.hostPointer, v.hostPointer+vectorSize, hostPointer );
    }

    ~vector()
    {
        if ( hostPointer!=NULL ) MemoryManager::freeHost ( hostPointer );
        if ( devicePointer!=NULL ) MemoryManager::free ( devicePointer );
    }

    size_type size() const
    {
        return vectorSize;
    }

    size_type capacity() const
    {
        return allocSize;
    }

    bool empty() const
    {
        return vectorSize==0;
    }

    void reserve (size_type s)
    {
        if ( s <= allocSize ) return;
        allocSize = ( s>2*allocSize ) ?s:2*allocSize;
        // always allocate multiples of BSIZE values
        allocSize = ( allocSize+MemoryManager::BSIZE-1 ) & (size_type)(-(long)MemoryManager::BSIZE);

        void* prevDevicePointer = devicePointer;
        //COMM : if (mycudaVerboseLevel>=LOG_INFO) std::cout << "CudaVector<"<<sofa::core::objectmodel::Base::className((T*)NULL)<<"> : reserve("<<s<<")"<<std::endl;

        MemoryManager::deviceAlloc( &devicePointer, allocSize*sizeof ( T ) );

        if ( vectorSize > 0 && deviceIsValid )
            MemoryManager::memcpyDeviceToDevice ( devicePointer, prevDevicePointer, vectorSize*sizeof ( T ) );

        if ( prevDevicePointer != NULL )
            MemoryManager::deviceFree ( prevDevicePointer );

        T* prevHostPointer = hostPointer;
        void* newHostPointer = NULL;
        MemoryManager::mallocHost ( &newHostPointer, allocSize*sizeof ( T ) );

        hostPointer = (T*)newHostPointer;
        if ( vectorSize!=0 && hostIsValid )
            std::copy ( prevHostPointer, prevHostPointer+vectorSize, hostPointer );
        if ( prevHostPointer != NULL )
            mycudaFreeHost ( prevHostPointer );
    }
    /// resize the vector without calling constructors or destructors, and without synchronizing the device and host copy
    void fastResize ( size_type s)
    {
        if ( s == vectorSize ) return;
        reserve ( s,MemoryManager::BWARP_SIZE);
        vectorSize = s;
        if ( !vectorSize )
        {
            // special case when the vector is now empty -> host and device are valid
            deviceIsValid = true;
            hostIsValid = true;
        }
    }
    /// resize the vector discarding any old values, without calling constructors or destructors, and without synchronizing the device and host copy
    void recreate( size_type s)
    {
        clear();
        fastResize(s,MemoryManager::BWARP_SIZE);
    }

    void memsetDevice(int v = 0)
    {
        MemoryManager::memsetDevice(devicePointer, v, vectorSize*sizeof(T));
        hostIsValid = false;
        deviceIsValid = true;
    }

    void memsetHost(int v = 0)
    {
        MemoryManager::memsetHost(hostPointer,v,vectorSize*sizeof(T));
        hostIsValid = true;
        deviceIsValid = false;
    }

    void resize ( size_type s)
    {
        if ( s == vectorSize ) return;
        reserve ( s,MemoryManager::BSIZE);
        if ( s > vectorSize )
        {
            if (sofa::defaulttype::DataTypeInfo<T>::ZeroConstructor )   // can use memset instead of constructors
            {
                if (hostIsValid)
                    memset(hostPointer+vectorSize,0,(s-vectorSize)*sizeof(T));
                if (deviceIsValid)
                    mycudaMemset((T*)devicePointer+vectorSize, 0, (s-vectorSize)*sizeof(T));
            }
            else
            {
                MemoryManager::copyToHost();
                memset(hostPointer+vectorSize,0,(s-vectorSize)*sizeof(T));
                // Call the constructor for the new elements
                for ( size_type i = vectorSize; i < s; i++ )
                {
                    ::new ( hostPointer+i ) T;
                }
                if ( deviceIsValid )
                {
                    if ( vectorSize == 0 )
                    {
                        // wait until the transfer is really necessary, as other modifications might follow
                        deviceIsValid = false;
                    }
                    else
                    {
                        mycudaMemcpyHostToDevice ( ( ( T* ) devicePointer ) +vectorSize, hostPointer+vectorSize, ( s-vectorSize ) *sizeof ( T ) );
                    }
                }
            }
        }
        else if (s < vectorSize && !(defaulttype::DataTypeInfo<T>::SimpleCopy))     // need to call destructors
        {
            MemoryManager::copyToHost();
            // Call the destructor for the deleted elements
            for ( size_type i = s; i < vectorSize; i++ )
            {
                hostPointer[i].~T();
            }
        }
        vectorSize = s;

        if ( !vectorSize )   // special case when the vector is now empty -> host and device are valid
        {
            deviceIsValid = true;
            hostIsValid = true;
        }
    }

    void swap ( vector<T,MemoryManager>& v )
    {
#define VSWAP(type, var) { type t = var; var = v.var; v.var = t; }
        VSWAP ( size_type, vectorSize );
        VSWAP ( size_type, allocSize );
        VSWAP ( void*    , devicePointer );
        VSWAP ( T*       , hostPointer );
        VSWAP ( bool     , deviceIsValid );
        VSWAP ( bool     , hostIsValid );
#undef VSWAP
    }

    const void* deviceReadAt ( int i ) const
    {
        MemoryManager::copyToDevice();
        return ( ( const T* ) devicePointer ) +i;
    }

    const void* deviceRead ( ) const { return deviceReadAt(0); }

    void* deviceWrite ( int i=0 )
    {
        MemoryManager::copyToDevice();
        hostIsValid = false;
        return ( ( T* ) devicePointer ) +i;
    }

    const T* hostRead ( int i=0 ) const
    {
        MemoryManager::copyToHost();
        return hostPointer+i;
    }

    T* hostWrite ( int i=0 )
    {
        MemoryManager::copyToHost();
        deviceIsValid = false;
        return hostPointer+i;
    }

    bool isHostValid() const
    {
        return hostIsValid;
    }

    bool isDeviceValid() const
    {
        return deviceIsValid;
    }

    void push_back ( const T& t )
    {
        size_type i = size();
        MemoryManager::copyToHost();
        deviceIsValid = false;
        fastResize ( i+1 );
        ::new ( hostPointer+i ) T ( t );
    }
    void pop_back()
    {
        if ( !empty() )
            resize ( size()-1 );
    }

    const T& operator[] ( size_type i ) const
    {
        checkIndex ( i );
        return hostRead() [i];
    }

    T& operator[] ( size_type i )
    {
        checkIndex ( i );
        return hostWrite() [i];
    }

    const T& getCached ( size_type i ) const
    {
        checkIndex ( i );
        return hostPointer[i];
    }

    const T& getSingle ( size_type i ) const
    {
        MemoryManager::copyToHostSingle(i);
        return hostPointer[i];
    }

    const_iterator begin() const { return hostRead(); }
    const_iterator end() const { return hostRead()+size(); }

    iterator begin() { return hostWrite(); }
    iterator end() { return hostWrite()+size(); }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const vector<T,MemoryManager>& vec )
    {
        if ( vec.size() >0 )
        {
            for ( unsigned int i=0; i<vec.size()-1; ++i ) os<<vec[i]<<" ";
            os<<vec[vec.size()-1];
        }
        return os;
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, vector<T,MemoryManager>& vec )
    {
        T t;
        vec.clear();
        while ( in>>t )
        {
            vec.push_back ( t );
        }
        if ( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

#ifdef NDEBUG
    void checkIndex ( size_type,size_type ) const
    {
    }
#else
    void checkIndex ( size_type x,size_type y) const
    {
        assert (x<this->sizeX);
        assert (y<this->sizeY);
    }
#endif
};

//classic vector (using CPUMemoryManager, same behavior as std::helper)
template <class T>
class vector<T, CPUMemoryManager<T> > : public std::vector<T, std::allocator<T> >
{
public:
    typedef std::allocator<T> Alloc;
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
    vector(typename vector<T>::const_iterator first, typename vector<T>::const_iterator last): std::vector<T>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */


#ifndef SOFA_NO_VECTOR_ACCESS_FAILURE

    /// Read/write random access
    reference operator[](size_type n)
    {
#ifndef NDEBUG
        if (n>=this->size())
            vector_access_failure(this, this->size(), n, typeid(T));
        //assert( n<this->size() );
#endif
        return *(this->begin() + n);
    }

    /// Read-only random access
    const_reference operator[](size_type n) const
    {
#ifndef NDEBUG
        if (n>=this->size())
            vector_access_failure(this, this->size(), n, typeid(T));
        //assert( n<this->size() );
#endif
        return *(this->begin() + n);
    }

#endif // SOFA_NO_VECTOR_ACCESS_FAILURE


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
};


/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B, optionnally specifying a step using "A-B-step" notation.
template<>
inline std::istream& vector<int >::read( std::istream& in )
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
            int t1,t2,tinc;
            std::string s1(s,0,hyphen);
            t1 = atoi(s1.c_str());
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = atoi(s2.c_str());
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2);
                std::string s3(s,hyphen2+1);
                t2 = atoi(s2.c_str());
                tinc = atoi(s3.c_str());
                if (tinc == 0)
                {
                    std::cerr << "ERROR parsing \""<<s<<"\": increment is 0\n";
                    tinc = (t1<t2) ? 1 : -1;
                }
                if ((t2-t1)*tinc < 0)
                {
                    // increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }
            if (tinc < 0)
                for (t=t1; t>=t2; t+=tinc)
                    this->push_back(t);
            else
                for (t=t1; t<=t2; t+=tinc)
                    this->push_back(t);
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

/// Output stream
/// Specialization for writing vectors of unsigned char
template<>
inline std::ostream& vector<unsigned char >::write(std::ostream& os) const
{
    if( this->size()>0 )
    {
        for( unsigned int i=0; i<this->size()-1; ++i ) os<<(int)(*this)[i]<<" ";
        os<<(int)(*this)[this->size()-1];
    }
    return os;
}

/// Inpu stream
/// Specialization for writing vectors of unsigned char
template<>
inline std::istream&  vector<unsigned char >::read(std::istream& in)
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

/// Input stream
/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B
template<>
inline std::istream& vector<unsigned int >::read( std::istream& in )
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
            unsigned int t1,t2;
            int tinc;
            std::string s1(s,0,hyphen);
            t1 = (unsigned int)atoi(s1.c_str());
            std::string::size_type hyphen2 = s.find_first_of('-',hyphen+2);
            if (hyphen2 == std::string::npos)
            {
                std::string s2(s,hyphen+1);
                t2 = (unsigned int)atoi(s2.c_str());
                tinc = (t1<t2) ? 1 : -1;
            }
            else
            {
                std::string s2(s,hyphen+1,hyphen2);
                std::string s3(s,hyphen2+1);
                t2 = (unsigned int)atoi(s2.c_str());
                tinc = atoi(s3.c_str());
                if (tinc == 0)
                {
                    std::cerr << "ERROR parsing \""<<s<<"\": increment is 0\n";
                    tinc = (t1<t2) ? 1 : -1;
                }
                if (((int)(t2-t1))*tinc < 0)
                {
                    // increment not of the same sign as t2-t1 : swap t1<->t2
                    t = t1;
                    t1 = t2;
                    t2 = t;
                }
            }
            if (tinc < 0)
                for (t=t1; t>=t2; t=(unsigned int)((int)t+tinc))
                    this->push_back(t);
            else
                for (t=t1; t<=t2; t=(unsigned int)((int)t+tinc))
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
#if !defined(NDEBUG) && !defined(SOFA_NO_VECTOR_ACCESS_FAILURE)
    //assert( 0<= static_cast<int>(index) && index <v.size() );
    if (index>=v.size())
        vector_access_failure(&v, v.size(), index, typeid(T));
#endif
    v[index] = v.back();
    v.pop_back();
}

} // namespace helper

} // namespace sofa

#endif //SOFA_HELPER_VECTOR_H


