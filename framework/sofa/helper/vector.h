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
*                               SOFA :: Modules                               *
*                                                                             *
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
#include <stdlib.h>
#include <typeinfo>
#include <stdio.h>

#include <sofa/helper/helper.h>
#include <sofa/helper/MemoryManager.h>
#include <sofa/defaulttype/DataTypeInfo.h>

/// uncomment if you want to allocate the minimum size on your device. however, this requires more reallocation if the size increase often
#define MINIMUM_SIZE_DEVICE

/// maximum number of bytes we allow to double the size when we reserve
/// if MINIMUM_SIZE_DEVICE is disable, it also use this value else it use the minimum size
#define MAXIMUM_DOUBLE_SIZE 32768

//#define DEBUG_OUT_VECTOR

#ifdef DEBUG_OUT_VECTOR
#define DEBUG_OUT_V(a) a
#define SPACEP std::cout << id << " : " << "(" << hostIsValid << "," << (deviceIsValid&1) << ") " ;for(int espaceaff=0;espaceaff<spaceDebug;espaceaff++) std::cout << "  ";spaceDebug++; std::cout << ">"
#define SPACEM std::cout << id << " : " << "(" << hostIsValid << "," << (deviceIsValid&1) << ") " ;spaceDebug--;for(int espaceaff=0;espaceaff<spaceDebug;espaceaff++) std::cout << "  "; std::cout << "<"
#define SPACEN std::cout << id << " : " << "(" << hostIsValid << "," << (deviceIsValid&1) << ") " ;for(int espaceaff=0;espaceaff<spaceDebug;espaceaff++) std::cout << "  "; std::cout << "."
#else
#define DEBUG_OUT_V(a)
#endif

namespace sofa
{

namespace helper
{

void SOFA_HELPER_API vector_access_failure(const void* vec, unsigned size, unsigned i, const std::type_info& type);

DEBUG_OUT_V(extern SOFA_HELPER_API int cptid);

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
    typedef typename MemoryManager::device_pointer device_pointer;

protected:
    size_type     vectorSize;     ///< Current size of the vector
    size_type     allocSize;      ///< Allocated size
    mutable size_type      vectorSizeDevice[MemoryManager::MAX_DEVICES];      ///< Allocated size
#ifdef MINIMUM_SIZE_DEVICE
    mutable size_type      deviceAllocSize;      ///< Allocated size
#endif
    mutable device_pointer devicePointer[MemoryManager::MAX_DEVICES];  ///< Pointer to the data on the GPU side
    mutable size_type      clearDevice;  ///< need to clear device until the next alloc?
    T*            hostPointer;    ///< Pointer to the data on the CPU side
    GLuint        bufferObject;   ///< Optionnal associated OpenGL buffer ID
    mutable int   deviceIsValid;  ///< True if the data on the GPU is currently valid
    mutable bool  hostIsValid;    ///< True if the data on the CPU is currently valid
    mutable bool  bufferIsRegistered;  ///< True if the OpenGL buffer is registered with CUDA
    enum { ALL_DEVICE_VALID = 0xFFFFFFFF };

    DEBUG_OUT_V(int id);
    DEBUG_OUT_V(mutable int spaceDebug);

public:

    vector()
        : vectorSize ( 0 ), allocSize ( 0 ), hostPointer ( NULL ), bufferObject(0), deviceIsValid ( ALL_DEVICE_VALID ), hostIsValid ( true ), bufferIsRegistered(false)
    {
        DEBUG_OUT_V(id = cptid);
        DEBUG_OUT_V(cptid++);
        DEBUG_OUT_V(spaceDebug = 0);
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            devicePointer[d] = MemoryManager::null();
            vectorSizeDevice[d] = 0;
        }
#ifdef MINIMUM_SIZE_DEVICE
        deviceAllocSize = 0;
#endif
        clearDevice = 0;
    }
    vector ( size_type n )
        : vectorSize ( 0 ), allocSize ( 0 ), hostPointer ( NULL ), bufferObject(0), deviceIsValid ( ALL_DEVICE_VALID ), hostIsValid ( true ), bufferIsRegistered(false)
    {
        DEBUG_OUT_V(id = cptid);
        DEBUG_OUT_V(cptid++);
        DEBUG_OUT_V(spaceDebug = 0);
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            devicePointer[d] = MemoryManager::null();
            vectorSizeDevice[d] = 0;
        }
        clearDevice = 0;
#ifdef MINIMUM_SIZE_DEVICE
        deviceAllocSize = 0;
#endif
        resize ( n );
    }
    vector ( const vector<T,MemoryManager >& v )
        : vectorSize ( 0 ), allocSize ( 0 ), hostPointer ( NULL ), bufferObject(0), deviceIsValid ( ALL_DEVICE_VALID ), hostIsValid ( true ), bufferIsRegistered(false)
    {
        DEBUG_OUT_V(id = cptid);
        DEBUG_OUT_V(cptid++);
        DEBUG_OUT_V(spaceDebug = 0);
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            devicePointer[d] = MemoryManager::null();
            vectorSizeDevice[d] = 0;
        }
        clearDevice = 0;
#ifdef MINIMUM_SIZE_DEVICE
        deviceAllocSize = 0;
#endif
        *this = v;
    }

    bool isHostValid() const
    {
        return hostIsValid;
    }
    bool isDeviceValid(unsigned gpu) const
    {
        return deviceIsValid & (1<<gpu);
    }

    void clear()
    {
        DEBUG_OUT_V(SPACEP << "clear vector" << std::endl);
        vectorSize = 0;
        deviceIsValid = ALL_DEVICE_VALID;
        hostIsValid = true;
        DEBUG_OUT_V(SPACEM << "clear vector " << std::endl);
    }

    void operator= ( const vector<T,MemoryManager >& v )
    {
        if (&v == this)
        {
            //COMM : std::cerr << "ERROR: self-assignment of CudaVector< " << core::objectmodel::Base::decodeTypeName(typeid(T)) << ">"<<std::endl;
            return;
        }
        DEBUG_OUT_V(SPACEP << "operator=, id is " << v.id << "(" << v.hostIsValid << "," << (v.deviceIsValid&1) << ") " << std::endl);
        DEBUG_OUT_V(std::cout << v.id << " : " << "(" << v.hostIsValid << "," << (v.deviceIsValid&1) << ") " << ". operator= param " << id << std::endl);

        size_type newSize = v.size();
        clear();

        fastResize ( newSize );

        if ( vectorSize > 0)
        {
            if (v.hostIsValid ) std::copy ( v.hostPointer, v.hostPointer+vectorSize, hostPointer );

            clearDevice = v.clearDevice;
            if (v.deviceIsValid)
            {

                if (MemoryManager::SUPPORT_GL_BUFFER)
                {
                    if (bufferObject) mapBuffer();//COMM : necessaire????
                    if (v.bufferObject) v.mapBuffer();
                }
                deviceIsValid = 0; /// we specify that we don't want to copy previous value of the current vector
                for (int d=0; d<MemoryManager::numDevices(); d++)
                {
                    if (v.isDeviceValid(d))
                    {
                        //v.allocate(d); /// make sure that the device data are correct
                        allocate(d); /// device are not valid so it only allocate

                        if (vectorSize <= v.vectorSizeDevice[d])
                        {
                            DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid alloc) " << 0 << "->" << vectorSize << std::endl);
                            MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], v.devicePointer[d], vectorSize*sizeof ( T ) );
                        }
                        else
                        {
                            DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid vector) " << 0 << "->" << v.vectorSizeDevice[d] << std::endl);
                            MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], v.devicePointer[d], v.vectorSizeDevice[d]*sizeof ( T ) );

                            if (clearDevice > v.vectorSizeDevice[d])
                            {
                                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetDevice (clear new data) " << v.vectorSizeDevice[d] << "->" << vectorSize-v.vectorSizeDevice[d] << std::endl);
                                MemoryManager::memsetDevice(d,MemoryManager::deviceOffset(devicePointer[d],v.vectorSizeDevice[d]), 0, (clearDevice-v.vectorSizeDevice[d])*sizeof(T));
                            }
                        }
                    }
                }
            }
        }

        hostIsValid = v.hostIsValid;
        deviceIsValid = v.deviceIsValid; /// finally we get the correct device valid

        DEBUG_OUT_V(SPACEM << "operator= " << std::endl);
    }

    ~vector()
    {
        if ( hostPointer!=NULL ) MemoryManager::hostFree ( hostPointer );

        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject )
        {
            unregisterBuffer();
            MemoryManager::bufferFree(bufferObject);
            devicePointer[MemoryManager::getBufferDevice()] = MemoryManager::null(); // already free
        }
        else
        {
            for (int d=0; d<MemoryManager::numDevices(); d++)
            {
                if ( !MemoryManager::isNull(devicePointer[d]) ) MemoryManager::deviceFree(d, (devicePointer[d]) );
            }
        }
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

    void reserve (size_type s,size_type WARP_SIZE=MemoryManager::BSIZE)
    {
#ifdef MINIMUM_SIZE_DEVICE
        if ( s > deviceAllocSize) deviceAllocSize = ((s+WARP_SIZE-1 ) / WARP_SIZE) * WARP_SIZE;
#endif
        if ( s <= allocSize ) return;
        DEBUG_OUT_V(SPACEP << "reserve " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std::endl);
        allocSize = ( s>2*allocSize || s>=MAXIMUM_DOUBLE_SIZE) ?s:2*allocSize;
        // always allocate multiples of BSIZE values
        allocSize = ( allocSize+WARP_SIZE-1 ) & (size_type)(-(long)WARP_SIZE);


        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject)
        {
            DEBUG_OUT_V(SPACEN << "BUFFEROBJ " << std::endl);
            //COMM : if (mycudaVerboseLevel>=LOG_INFO) std::cout << "CudaVector<"<<sofa::core::objectmodel::Base::className((T*)NULL)<<"> : GL reserve("<<s<<")"<<std::endl;
            hostRead(); // make sure the host copy is valid
            unregisterBuffer();
            //COMM fct opengl
            glBindBuffer( GL_ARRAY_BUFFER, bufferObject);
            glBufferData( GL_ARRAY_BUFFER, allocSize*sizeof ( T ), 0, GL_DYNAMIC_DRAW);
            glBindBuffer( GL_ARRAY_BUFFER, 0);
            if ( vectorSize > 0 ) deviceIsValid = 0;
        }

// 	else {
// 		for (int d=0;d<MemoryManager::numDevices();d++) {
//                         device_pointer prevDevicePointer = devicePointer[d];
// 			//COMM : if (mycudaVerboseLevel>=LOG_INFO) std::cout << "CudaVector<"<<sofa::core::objectmodel::Base::className((T*)NULL)<<"> : reserve("<<s<<")"<<std::endl;
// 			MemoryManager::deviceAlloc(d, &devicePointer[d], allocSize*sizeof ( T ) );
// 			if ( vectorSize > 0 && isDeviceValid(d)) MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], prevDevicePointer, vectorSize*sizeof ( T ) );
// 			if ( !MemoryManager::isNull(prevDevicePointer)) MemoryManager::deviceFree (d, prevDevicePointer );
// 		}
// 	}

        T* prevHostPointer = hostPointer;
        void* newHostPointer = NULL;
        DEBUG_OUT_V(SPACEN<< "MemoryManager::hostAlloc " << allocSize << std::endl);
        MemoryManager::hostAlloc( &newHostPointer, allocSize*sizeof ( T ) );
        hostPointer = (T*)newHostPointer;
        if ( vectorSize!=0 && hostIsValid ) std::copy ( prevHostPointer, prevHostPointer+vectorSize, hostPointer );
        if ( prevHostPointer != NULL ) MemoryManager::hostFree( prevHostPointer );
        DEBUG_OUT_V(SPACEM << "reserve " << " (alloc=" << allocSize << ")" << std::endl);
    }

    /// resize the vector without calling constructors or destructors, and without synchronizing the device and host copy
    void fastResize ( size_type s,size_type WARP_SIZE=MemoryManager::BSIZE)
    {
        if ( s == vectorSize ) return;
        DEBUG_OUT_V(SPACEP << "fastresize " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std::endl);
        reserve ( s,WARP_SIZE);
        if (s<vectorSize) clearDevice=0;
        vectorSize = s;
        if ( !vectorSize )
        {
            // special case when the vector is now empty -> host and device are valid
            deviceIsValid = ALL_DEVICE_VALID;
            hostIsValid = true;
        }
        DEBUG_OUT_V(SPACEM << "fastresize " << std::endl);
    }
    /// resize the vector discarding any old values, without calling constructors or destructors, and without synchronizing the device and host copy
    void recreate( size_type s,size_type WARP_SIZE=MemoryManager::BSIZE)
    {
        clear();
        fastResize(s,WARP_SIZE);
    }

    void invalidateDevice()
    {
        hostIsValid = true;
        deviceIsValid = 0;
    }

    void memsetDevice(int v = 0)
    {
        DEBUG_OUT_V(SPACEP << "memsetDevice " << std::endl);
        deviceIsValid = 0;
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            if (vectorSizeDevice[d]>0)   /// if the vector has already been used
            {
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetDevice " << vectorSizeDevice[d] << std::endl);
                allocate(d); /// make sure the size is correct device is not valid so it only resize if necessary
                MemoryManager::memsetDevice(d, devicePointer[d], v, vectorSize*sizeof(T));
                deviceIsValid |= 1<<d;
            }
        }

        /// if we found at least one device valid we invalidate the host else the host is memset, device will be set at next copytodevice
        if (deviceIsValid) hostIsValid = false;
        else memsetHost(v);

        DEBUG_OUT_V(SPACEM << "memsetDevice " << std::endl);
    }

    void memsetHost(int v = 0)
    {
        MemoryManager::memsetHost(hostPointer,v,vectorSize*sizeof(T));
        hostIsValid = true;
        deviceIsValid = 0;
    }

    void resize ( size_type s,size_type WARP_SIZE=MemoryManager::BSIZE)
    {
        if ( s == vectorSize ) return;
        DEBUG_OUT_V(SPACEP << "resize " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std::endl);
        reserve ( s,WARP_SIZE);
        if ( s > vectorSize )
        {
            if (sofa::defaulttype::DataTypeInfo<T>::ZeroConstructor )   // can use memset instead of constructors
            {
                if (hostIsValid)
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost (new data) " << (s-vectorSize) << std::endl);
                    MemoryManager::memsetHost(hostPointer+vectorSize,0,(s-vectorSize)*sizeof(T));
                }
                clearDevice=s;
                for (int d=0; d<MemoryManager::numDevices(); d++)
                {
                    if (isDeviceValid(d))
                    {
                        if (s<vectorSizeDevice[d]) MemoryManager::memsetDevice(d, devicePointer[d], 0, s*sizeof(T));
                        else deviceIsValid &= ~(1<<d);
                    }
                }
            }
            else     /// this is no thread safe
            {
                DEBUG_OUT_V(SPACEN << "ZEROCONST " << std::endl);
                copyToHost();
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost (new data) " << (s-vectorSize) << std::endl);
                MemoryManager::memsetHost(hostPointer+vectorSize,0,(s-vectorSize)*sizeof(T));
                // Call the constructor for the new elements
                for ( size_type i = vectorSize; i < s; i++ ) ::new ( hostPointer+i ) T;

                if ( vectorSize == 0 )   // wait until the transfer is really necessary, as other modifications might follow
                {
                    deviceIsValid = 0;
                }
                else
                {
                    if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();

                    for (int d=0; d<MemoryManager::numDevices(); d++)
                    {
                        if (!MemoryManager::isNull(devicePointer[d]) &&  isDeviceValid(d) )
                        {
                            allocate(d);
                            DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyHostToDevice " << vectorSize << "->" << ( s-vectorSize ) << std::endl);
                            MemoryManager::memcpyHostToDevice(d, MemoryManager::deviceOffset(devicePointer[d], vectorSize), hostPointer+vectorSize, ( s-vectorSize ) *sizeof ( T ) );
                        }
                    }
                }
            }
        }
        else if (s < vectorSize && !(defaulttype::DataTypeInfo<T>::SimpleCopy))     // need to call destructors
        {
            DEBUG_OUT_V(SPACEN << "SIMPLECOPY " << std::endl);
            copyToHost();
            // Call the destructor for the deleted elements
            for ( size_type i = s; i < vectorSize; i++ )
            {
                hostPointer[i].~T();
            }
        }
        vectorSize = s;


        if ( !vectorSize )   // special case when the vector is now empty -> host and device are valid
        {
            deviceIsValid = ALL_DEVICE_VALID;
            hostIsValid = true;
        }
        //deviceIsValid = 0;

        DEBUG_OUT_V(SPACEM << "resize " << std::endl);
    }

    void swap ( vector<T,MemoryManager>& v )
    {
        DEBUG_OUT_V(SPACEP << "swap " << std::endl);
#define VSWAP(type, var) { type t = var; var = v.var; v.var = t; }
        VSWAP ( size_type, vectorSize );
        VSWAP ( size_type, allocSize );
        VSWAP ( int, clearDevice );
#ifdef MINIMUM_SIZE_DEVICE
        VSWAP ( int, deviceAllocSize );
#endif
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            VSWAP ( void*    , devicePointer[d] );
            VSWAP ( int    ,  vectorSizeDevice[d] );
        }
        VSWAP ( T*       , hostPointer );
        VSWAP ( GLuint   , bufferObject );
        VSWAP ( int      , deviceIsValid );
        VSWAP ( bool     , hostIsValid );
        VSWAP ( bool     , bufferIsRegistered );
#undef VSWAP
        DEBUG_OUT_V(SPACEM << "swap " << std::endl);
    }

    const device_pointer deviceReadAt ( int i ,int gpu = MemoryManager::getBufferDevice()) const
    {
        DEBUG_OUT_V(if (!(deviceIsValid & (1<<gpu))) {SPACEN << "deviceRead" << std::endl;});
        copyToDevice(gpu);
        return MemoryManager::deviceOffset(devicePointer[gpu],i);
    }

    const device_pointer deviceRead ( int gpu = MemoryManager::getBufferDevice()) const { return deviceReadAt(0,gpu); }

    device_pointer deviceWriteAt ( int i ,int gpu = MemoryManager::getBufferDevice())
    {
        DEBUG_OUT_V(if (hostIsValid) {SPACEN << "deviceWrite" << std::endl;});
        copyToDevice(gpu);
        hostIsValid = false;
        deviceIsValid |= 1<<gpu;
        return MemoryManager::deviceOffset(devicePointer[gpu],i);
    }

    device_pointer deviceWrite (int gpu = MemoryManager::getBufferDevice()) { return deviceWriteAt(0,gpu); }

    const T* hostRead ( int i=0 ) const
    {
        DEBUG_OUT_V(if (!hostIsValid) {SPACEN << "hostRead" << std::endl;});
        copyToHost();
        return hostPointer+i;
    }

    T* hostWrite ( int i=0 )
    {
        DEBUG_OUT_V(if (deviceIsValid) {SPACEN << "hostWrite" << std::endl;});
        copyToHost();
        deviceIsValid = 0;
        return hostPointer+i;
    }

    /// Get the OpenGL Buffer Object ID for reading
    GLuint bufferRead(bool create = false)
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (!bufferObject)
            {
                if (create) createBuffer();
                else return 0;
            }
            if (!isDeviceValid(MemoryManager::getBufferDevice()))
                copyToDevice(MemoryManager::getBufferDevice());
            unmapBuffer();
            return bufferObject;
        }
        return 0;
    }

    /// Get the OpenGL Buffer Object ID for writing
    GLuint bufferWrite(bool create = false)
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (!bufferObject)
            {
                if (create) createBuffer();
                else return 0;
            }
            if (!isDeviceValid(MemoryManager::getBufferDevice()))
                copyToDevice(MemoryManager::getBufferDevice());
            unmapBuffer();
            hostIsValid = false;
            deviceIsValid |= 1<<MemoryManager::getBufferDevice();
            return bufferObject;
        }
    }

    void push_back ( const T& t )
    {
        size_type i = size();
        copyToHost();
        deviceIsValid = 0;
        fastResize ( i+1 );
        ::new ( hostPointer+i ) T ( t );
    }

    void pop_back()
    {
        if (!empty()) resize ( size()-1 );
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
        copyToHostSingle(i);
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



protected:
    void allocate(int d) const
    {
#ifdef MINIMUM_SIZE_DEVICE
        size_t alloc = deviceAllocSize;
#else
        size_t alloc = allocSize;
#endif

        if (vectorSizeDevice[d] < alloc)
        {
            DEBUG_OUT_V(SPACEP << "allocate device=" << d << " " << vectorSizeDevice[d] << "->" << alloc << std::endl);
            device_pointer prevDevicePointer = devicePointer[d];
            //COMM : if (mycudaVerboseLevel>=LOG_INFO) std::cout << "CudaVector<"<<sofa::core::objectmodel::Base::className((T*)NULL)<<"> : reserve("<<s<<")"<<std::endl;
            MemoryManager::deviceAlloc(d, &devicePointer[d], alloc*sizeof ( T ) );

            if ( vectorSize > 0  && isDeviceValid(d))
            {
                if (vectorSize <= vectorSizeDevice[d])
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid vector) " << 0 << "->" << vectorSize << std::endl);
                    MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], prevDevicePointer, vectorSize*sizeof ( T ) );
                }
                else
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid alloc) " << 0 << "->" << vectorSizeDevice[d] << std::endl);
                    MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], prevDevicePointer, vectorSizeDevice[d]*sizeof ( T ) );

                    if (clearDevice > vectorSizeDevice[d])
                    {
                        DEBUG_OUT_V(SPACEN << "MemoryManager::memsetDevice (clear new data) " << vectorSizeDevice[d] << "->" << clearDevice-vectorSizeDevice[d] << std::endl);
                        MemoryManager::memsetDevice(d,MemoryManager::deviceOffset(devicePointer[d],vectorSizeDevice[d]), 0, (clearDevice-vectorSizeDevice[d])*sizeof(T));
                    }
                }
            }

            if ( !MemoryManager::isNull(prevDevicePointer)) MemoryManager::deviceFree (d, prevDevicePointer );

            vectorSizeDevice[d] = alloc;
            DEBUG_OUT_V(SPACEM << "allocate " << std::endl);
        }
    }

    void copyToHost() const
    {
        if ( hostIsValid ) return;
        DEBUG_OUT_V(SPACEP << "copyToHost " << std::endl);
//#ifndef NDEBUG
        // COMM : if (mycudaVerboseLevel>=LOG_TRACE) {
        // COMM :     std::cout << "CUDA: GPU->CPU copy of "<<sofa::core::objectmodel::Base::decodeTypeName ( typeid ( *this ) ) <<": "<<vectorSize*sizeof ( T ) <<" B"<<std::endl;
        //sofa::helper::BackTrace::dump();
        // COMM : }
//#endif

        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();

        /// if host is not valid data are valid and allocated on a device
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            if (!MemoryManager::isNull(devicePointer[d]) && isDeviceValid(d) && vectorSize>0)
            {
                if (vectorSize <= vectorSizeDevice[d])
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToHost " << vectorSize << std::endl);
                    MemoryManager::memcpyDeviceToHost (d, hostPointer, devicePointer[d], vectorSize*sizeof ( T ) );
                }
                else
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToHost " << vectorSize << std::endl);
                    MemoryManager::memcpyDeviceToHost (d, hostPointer, devicePointer[d], vectorSizeDevice[d]*sizeof ( T ) );

                    if (clearDevice > vectorSizeDevice[d]) MemoryManager::memsetHost(hostPointer+vectorSizeDevice[d],0,(clearDevice-vectorSizeDevice[d])*sizeof(T));
                }
                hostIsValid = true;
                break;
            }
        }

        DEBUG_OUT_V(SPACEM << "copyToHost " << std::endl);
    }

    void copyToDevice(int d = 0) const
    {
        allocate(d);
        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
        if (isDeviceValid(d)) return;
        DEBUG_OUT_V(SPACEP << "copyToDevice " << std::endl);

//#ifndef NDEBUG
        //COMM : if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: CPU->GPU copy of "<<sofa::core::objectmodel::Base::decodeTypeName ( typeid ( *this ) ) <<": "<<vectorSize*sizeof ( T ) <<" B"<<std::endl;
//#endif
        if ( !hostIsValid ) copyToHost();
        DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyHostToDevice " << vectorSize << std::endl);
        MemoryManager::memcpyHostToDevice (d, devicePointer[d], hostPointer, vectorSize*sizeof ( T ) );
        deviceIsValid |= 1<<d;
        DEBUG_OUT_V(SPACEM << "copyToDevice " << std::endl);
    }

    void copyToHostSingle(size_type i) const
    {
        if ( hostIsValid ) return;
        DEBUG_OUT_V(SPACEP << "copyToHostSingle " << std::endl);
//#ifndef NDEBUG
        //COMM : if (mycudaVerboseLevel>=LOG_TRACE) {
        // COMM : std::cout << "CUDA: GPU->CPU single copy of "<<sofa::core::objectmodel::Base::decodeTypeName ( typeid ( *this ) ) <<": "<<sizeof ( T ) <<" B"<<std::endl;
        //sofa::helper::BackTrace::dump();
        //COMM : }
//#endif

        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            if (!MemoryManager::isNull(devicePointer[d]) && isDeviceValid(d) && vectorSize>0)
            {
                DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToHost " << 1 << std::endl);
                if (i < clearDevice) hostPointer[i] = 0;
                else if (i < vectorSizeDevice[d]) MemoryManager::memcpyDeviceToHost(d, ((T*)hostPointer)+i, MemoryManager::deviceOffset(devicePointer[d],i), sizeof ( T ) );
                break;
            }
        }
        DEBUG_OUT_V(SPACEM << "copyToHostSingle " << std::endl);
    }

#ifdef NDEBUG
    void checkIndex ( size_type ) const {}
#else
    void checkIndex ( size_type i ) const
    {
        assert ( i<this->size() );
    }
#endif

    void registerBuffer() const
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (this->allocSize > 0 && !bufferIsRegistered)
            {
                bufferIsRegistered = MemoryManager::bufferRegister(bufferObject);
            }
        }
    }

    void mapBuffer() const
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            registerBuffer();
            if (bufferIsRegistered)
            {
                if (this->allocSize > 0 && MemoryManager::isNull(this->devicePointer[MemoryManager::getBufferDevice()]))
                {
                    MemoryManager::bufferMapToDevice((device_pointer*)&(this->devicePointer[MemoryManager::getBufferDevice()]), bufferObject);
                }
            }
            else
            {
                std::cout << "CUDA: Unable to map buffer to opengl" << std::endl;
            }
        }
    }

    void unmapBuffer() const
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (this->allocSize > 0 && !MemoryManager::isNull(this->devicePointer[MemoryManager::getBufferDevice()]))
            {
                MemoryManager::bufferUnmapToDevice((device_pointer*)&(this->devicePointer[MemoryManager::getBufferDevice()]), bufferObject);
                *((device_pointer*) &(this->devicePointer[MemoryManager::getBufferDevice()])) = MemoryManager::null();
            }
        }
    }

    void unregisterBuffer() const
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            unmapBuffer();
            if (this->allocSize > 0 && bufferIsRegistered)
            {
                //MemoryManager::bufferUnregister(bufferObject);
                MemoryManager::bufferFree(bufferObject);
                bufferIsRegistered = false;
            }
        }
    }

    void createBuffer()
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (bufferObject) return;

            if (this->allocSize > 0)
            {
                MemoryManager::bufferAlloc(&bufferObject,this->allocSize*sizeof(T));

                void* prevDevicePointer = this->devicePointer[MemoryManager::getBufferDevice()];
                this->devicePointer[MemoryManager::getBufferDevice()] = NULL;
                if (this->vectorSize>0 && this->isDeviceValid(MemoryManager::getBufferDevice()))
                {
                    deviceRead(MemoryManager::getBufferDevice());//check datas are on device MemoryManager::getBufferDevice()
                    mapBuffer();
                    if (prevDevicePointer) MemoryManager::memcpyDeviceToDevice ( MemoryManager::getBufferDevice(), devicePointer[MemoryManager::getBufferDevice()], prevDevicePointer, vectorSize*sizeof ( T ) );
                }
                if ( prevDevicePointer != NULL ) MemoryManager::deviceFree(MemoryManager::getBufferDevice(), prevDevicePointer);
            }
        }
    }

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
        this->operator=(x); return *this;
        /* an other way??
        this->resize(x.size());
        for(unsigned int i=0;i<x.size();i++){
        	this->operator[](i)=x[i];
        }
        return *this;
        */

        //std::vector<T,Alloc>::operator = (x);
        //return vector(x);
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

#ifdef DEBUG_OUT_VECTOR
#undef DEBUG_OUT_V
#undef SPACEP
#undef SPACEM
#undef SPACEN
#else
#undef DEBUG_OUT_V
#endif



} // namespace helper

} // namespace sofa

#endif //SOFA_HELPER_VECTOR_H


