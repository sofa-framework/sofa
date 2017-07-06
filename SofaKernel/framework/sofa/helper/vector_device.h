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
#ifndef SOFA_HELPER_VECTOR_DEVICE_H
#define SOFA_HELPER_VECTOR_DEVICE_H

#ifndef PS3
#include "system/gl.h"
#include <sofa/helper/vector.h>

// maximum number of bytes we allow to increase the size when of a vector in a single step when we reserve on the host or device
#define SOFA_VECTOR_HOST_STEP_SIZE 32768
#define SOFA_VECTOR_DEVICE_STEP_SIZE 32768

#if SOFA_VECTOR_HOST_STEP_SIZE != SOFA_VECTOR_DEVICE_STEP_SIZE
#define SOFA_VECTOR_DEVICE_CUSTOM_SIZE
#endif


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

DEBUG_OUT_V(extern SOFA_HELPER_API int cptid;)

template <class T, class MemoryManager >
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

    typedef MemoryManager memory_manager;
    template<class T2> struct rebind
    {
        typedef vector<T2, typename memory_manager::template rebind<T2>::other > other;
    };

protected:
    size_type     vectorSize;     ///< Current size of the vector
    size_type     allocSize;      ///< Allocated size on host
    mutable size_type      deviceVectorSize[MemoryManager::MAX_DEVICES];      ///< Initialized size on each device
    mutable size_type      deviceAllocSize[MemoryManager::MAX_DEVICES];      ///< Allocated size on each device
    mutable device_pointer devicePointer[MemoryManager::MAX_DEVICES];  ///< Pointer to the data on the GPU side
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
    mutable size_type      deviceReserveSize;      ///< Desired allocated size
#endif
    mutable size_type      clearSize;  ///< when initializing missing device data, up to where entries should be set to zero ?
    T*            hostPointer;    ///< Pointer to the data on the CPU side
    mutable int   deviceIsValid;  ///< True if the data on the GPU is currently valid (up to the given deviceVectorSize of each device, i.e. additionnal space may need to be allocated and/or initialized)
    mutable bool  hostIsValid;    ///< True if the data on the CPU is currently valid
    mutable bool  bufferIsRegistered;  ///< True if the OpenGL buffer is registered with CUDA
#ifndef SOFA_NO_OPENGL
    GLuint        bufferObject;   ///< Optionnal associated OpenGL buffer ID
#endif
    enum { ALL_DEVICE_VALID = 0xFFFFFFFF };

    DEBUG_OUT_V(int id;)
    DEBUG_OUT_V(mutable int spaceDebug;)

public:

    vector()
        : vectorSize ( 0 ), allocSize ( 0 ), hostPointer ( NULL ), deviceIsValid ( ALL_DEVICE_VALID ), hostIsValid ( true ), bufferIsRegistered(false)
#ifndef SOFA_NO_OPENGL
    , bufferObject(0)
#endif // SOFA_NO_OPENGL
    {
        DEBUG_OUT_V(id = cptid);
        DEBUG_OUT_V(cptid++);
        DEBUG_OUT_V(spaceDebug = 0);
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            devicePointer[d] = MemoryManager::null();
            deviceAllocSize[d] = 0;
            deviceVectorSize[d] = 0;
        }
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        deviceReserveSize = 0;
#endif
        clearSize = 0;
    }
    vector ( size_type n )
        : vectorSize ( 0 ), allocSize ( 0 ), hostPointer ( NULL ), deviceIsValid ( ALL_DEVICE_VALID ), hostIsValid ( true ), bufferIsRegistered(false)
#ifndef SOFA_NO_OPENGL
        , bufferObject(0)
#endif // SOFA_NO_OPENGL
    {
        DEBUG_OUT_V(id = cptid);
        DEBUG_OUT_V(cptid++);
        DEBUG_OUT_V(spaceDebug = 0);
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            devicePointer[d] = MemoryManager::null();
            deviceAllocSize[d] = 0;
            deviceVectorSize[d] = 0;
        }
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        deviceReserveSize = 0;
#endif
        clearSize = 0;
        resize ( n );
    }
    vector ( const vector<T,MemoryManager >& v )
        : vectorSize ( 0 ), allocSize ( 0 ), hostPointer ( NULL ), deviceIsValid ( ALL_DEVICE_VALID ), hostIsValid ( true ), bufferIsRegistered(false)
#ifndef SOFA_NO_OPENGL
        , bufferObject(0)
#endif // SOFA_NO_OPENGL
    {
        DEBUG_OUT_V(id = cptid);
        DEBUG_OUT_V(cptid++);
        DEBUG_OUT_V(spaceDebug = 0);
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            devicePointer[d] = MemoryManager::null();
            deviceAllocSize[d] = 0;
            deviceVectorSize[d] = 0;
        }
        clearSize = 0;
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        deviceReserveSize = 0;
#endif
        *this = v;
    }

    bool isHostValid() const
    {
        return hostIsValid;
    }
    bool isDeviceValid(unsigned gpu) const
    {
        return (deviceIsValid & (1<<gpu))!=0;
    }

    void clear()
    {
        DEBUG_OUT_V(SPACEP << "clear vector" << std::endl);
        vectorSize = 0;
        hostIsValid = true;
        deviceIsValid = ALL_DEVICE_VALID;
        for (int d=0; d<MemoryManager::numDevices(); d++)
            deviceVectorSize[d] = 0;
        clearSize = 0;
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

            clearSize = v.clearSize;
            if (v.deviceIsValid)
            {
#ifndef SOFA_NO_OPENGL
                if (MemoryManager::SUPPORT_GL_BUFFER)
                {
                    if (bufferObject) mapBuffer();//COMM : necessaire????
                    if (v.bufferObject) v.mapBuffer();
                }
#endif // SOFA_NO_OPENGL
                deviceIsValid = 0; /// we specify that we don't want to copy previous value of the current vector
                for (int d=0; d<MemoryManager::numDevices(); d++)
                {
                    if (v.isDeviceValid(d) && v.deviceVectorSize[d] > 0)
                    {
                        //v.allocate(d); /// make sure that the device data are correct
                        allocate(d); /// device are not valid so it only allocate

                        if (vectorSize <= v.deviceVectorSize[d])
                        {
                            DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy full vector) " << 0 << "->" << vectorSize << std::endl);
                            MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], v.devicePointer[d], vectorSize*sizeof ( T ) );
                            deviceVectorSize[d] = vectorSize;
                        }
                        else
                        {
                            DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid vector) " << 0 << "->" << v.deviceAllocSize[d] << std::endl);
                            MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], v.devicePointer[d], v.deviceVectorSize[d]*sizeof ( T ) );
                            deviceVectorSize[d] = v.deviceVectorSize[d];
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
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject )
        {
            unregisterBuffer();
            MemoryManager::bufferFree(bufferObject);
            devicePointer[MemoryManager::getBufferDevice()] = MemoryManager::null(); // already free
        }
        else
#endif // SOFA_NO_OPENGL
        {
            for (int d=0; d<MemoryManager::numDevices(); d++)
            {
                if ( !MemoryManager::isNull(devicePointer[d]) )
                    MemoryManager::deviceFree(d, (devicePointer[d]) );
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
        s = ((s+WARP_SIZE-1 ) / WARP_SIZE) * WARP_SIZE;
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        if ( s > deviceReserveSize)
        {
            // we double the reserved size except if the requested size is bigger or if we would allocate more memory than the configured step size
            if (s <= 2*deviceReserveSize && 2*deviceReserveSize <= s+SOFA_VECTOR_HOST_STEP_SIZE)
                deviceReserveSize *= 2;
            else
                deviceReserveSize = s;
            // always allocate multiples of WARP_SIZE values
            deviceReserveSize = ((deviceReserveSize+WARP_SIZE-1 ) / WARP_SIZE) * WARP_SIZE;
        }
#endif
        if ( s <= allocSize ) return;
        DEBUG_OUT_V(SPACEP << "reserve " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std
                ::endl);
        // we double the reserved size except if the requested size is bigger or if we would allocate more memory than the configured step size
        if (s <= 2*allocSize && 2*allocSize <= s+SOFA_VECTOR_HOST_STEP_SIZE)
            allocSize *= 2;
        else
            allocSize = s;
        // always allocate multiples of WARP_SIZE values
        allocSize = ((allocSize+WARP_SIZE-1 ) / WARP_SIZE) * WARP_SIZE;

#ifndef SOFA_NO_OPENGL
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
#endif // SOFA_NO_OPENGL

//         else {
//                 for (int d=0;d<MemoryManager::numDevices();d++) {
//                         device_pointer prevDevicePointer = devicePointer[d];
//                         //COMM : if (mycudaVerboseLevel>=LOG_INFO) std::cout << "CudaVector<"<<sofa::core::objectmodel::Base::className((T*)NULL)<<"> : reserve("<<s<<")"<<std::endl;
//                         MemoryManager::deviceAlloc(d, &devicePointer[d], allocSize*sizeof ( T ) );
//                         if ( vectorSize > 0 && isDeviceValid(d)) MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], prevDevicePointer, vectorSize*sizeof ( T ) );
//                         if ( !MemoryManager::isNull(prevDevicePointer)) MemoryManager::deviceFree (d, prevDevicePointer );
//                 }
//         }

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
        reserve(s, WARP_SIZE);
        vectorSize = s;
        if (clearSize > vectorSize) clearSize = vectorSize;
        for (int d=0; d<MemoryManager::numDevices(); d++)
            if (deviceVectorSize[d] > vectorSize)
                deviceVectorSize[d] = vectorSize;
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

    void invalidateHost()
    {
        hostIsValid = 0;
        deviceIsValid = ALL_DEVICE_VALID;
    }

    void memsetDevice(int v = 0)
    {
        DEBUG_OUT_V(SPACEP << "memsetDevice " << std::endl);

        deviceIsValid = 0;
        clearSize = 0;
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            if (deviceAllocSize[d]>0)   /// if the vector has already been used
            {
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetDevice " << deviceAllocSize[d] << std::endl);
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
        clearSize = 0;
        hostIsValid = true;
        deviceIsValid = 0;
    }

    void resize ( size_type s,size_type WARP_SIZE=MemoryManager::BSIZE)
    {
        reserve(s, WARP_SIZE);
        if ( s == vectorSize ) return;
        DEBUG_OUT_V(SPACEP << "resize " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std::endl);
        if ( s > vectorSize )
        {
            if (sofa::defaulttype::DataTypeInfo<T>::ZeroConstructor )   // can use memset instead of constructors
            {
                if (hostIsValid)
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost (new data) " << (s-vectorSize) << std::endl);
                    MemoryManager::memsetHost(hostPointer+vectorSize,0,(s-vectorSize)*sizeof(T));
                }
                clearSize = s;
#ifndef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
                size_t deviceReserveSize = allocSize;
#endif
                for (int d=0; d<MemoryManager::numDevices(); d++)
                {
                    if (isDeviceValid(d))
                    {
                        if (deviceAllocSize[d] >= deviceReserveSize)
                        {
                            MemoryManager::memsetDevice(d, MemoryManager::deviceOffset(devicePointer[d], deviceVectorSize[d]), 0, (s-deviceVectorSize[d])*sizeof(T));
                            deviceVectorSize[d] = s;
                        }
                        // if the memory allocated on a device is not sufficient, we do not reallocate and memset the new data until it is requested on the device
                        // but the valid flag is kept set because we will be able to provide the data on the device without transfers
                        //else deviceIsValid &= ~(1<<d);
                    }
                }
            }
            else     // must use class constructors -> resize can only happen on the host
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
#ifndef SOFA_NO_OPENGL
                    if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
#endif // SOFA_NO_OPENGL

                    for (int d=0; d<MemoryManager::numDevices(); d++)
                    {
                        if (isDeviceValid(d) )
                        {
                            if (deviceVectorSize[d] == 0) // no data is currently on the device -> we simply invalidate it and the transfer will happen once the data is actually requested
                                deviceIsValid &= ~(1<<d);
                            else
                            {
                                allocate(d);
                                DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyHostToDevice " << vectorSize << "->" << ( s-vectorSize ) << std::endl);
                                MemoryManager::memcpyHostToDevice(d, MemoryManager::deviceOffset(devicePointer[d], deviceVectorSize[d]), hostPointer+deviceVectorSize[d], ( s-deviceVectorSize[d] ) *sizeof ( T ) );
                                deviceVectorSize[d] = s;
                            }
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
        if (clearSize > vectorSize) clearSize = vectorSize;
        for (int d=0; d<MemoryManager::numDevices(); d++)
            if (deviceVectorSize[d] > vectorSize)
                deviceVectorSize[d] = vectorSize;

        if ( !vectorSize )   // special case when the vector is now empty -> host and device are valid
        {
            deviceIsValid = ALL_DEVICE_VALID;
            hostIsValid = true;
        }

        DEBUG_OUT_V(SPACEM << "resize " << std::endl);
    }

    void swap ( vector<T,MemoryManager>& v )
    {
        DEBUG_OUT_V(SPACEP << "swap " << std::endl);
#define VSWAP(type, var) { type t = var; var = v.var; v.var = t; }
        VSWAP ( size_type, vectorSize );
        VSWAP ( size_type, allocSize );
        VSWAP ( int, clearSize );
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        VSWAP ( int, deviceReserveSize );
#endif
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            VSWAP ( void*    , devicePointer[d] );
            VSWAP ( int    ,  deviceVectorSize[d] );
            VSWAP ( int    ,  deviceAllocSize[d] );
        }
        VSWAP ( T*       , hostPointer );
#ifndef SOFA_NO_OPENGL
        VSWAP ( GLuint   , bufferObject );
#endif // SOFA_NO_OPENGL
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

    const device_pointer deviceRead ( int gpu = MemoryManager::getBufferDevice()) const
    {
        return deviceReadAt(0,gpu);
    }

    device_pointer deviceWriteAt ( int i ,int gpu = MemoryManager::getBufferDevice())
    {
        DEBUG_OUT_V(if (hostIsValid) {SPACEN << "deviceWrite" << std::endl;});
        copyToDevice(gpu);
        if(vectorSize>0)
            hostIsValid = false;
        deviceIsValid = 1<<gpu;
        return MemoryManager::deviceOffset(devicePointer[gpu],i);
    }

    device_pointer deviceWrite (int gpu = MemoryManager::getBufferDevice())
    {
        return deviceWriteAt(0,gpu);
    }

    const T* hostRead() const
    {
        return hostReadAt(0);
    }

    T* hostWrite()
    {
        return hostWriteAt(0);
    }

    const T* hostReadAt ( int i ) const
    {
        DEBUG_OUT_V(if (!hostIsValid) {SPACEN << "hostRead" << std::endl;});
        copyToHost();
        return hostPointer+i;
    }

    T* hostWriteAt ( int i )
    {
        DEBUG_OUT_V(if (deviceIsValid) {SPACEN << "hostWrite" << std::endl;});
        copyToHost();
        if(vectorSize>0)
            deviceIsValid = false;
        return hostPointer+i;
    }

    /// Get the OpenGL Buffer Object ID for reading
#ifndef SOFA_NO_OPENGL
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
#endif

    /// Get the OpenGL Buffer Object ID for writing
#ifndef SOFA_NO_OPENGL
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
            deviceIsValid = 1<<MemoryManager::getBufferDevice();
            return bufferObject;
        }
        else return 0;
    }
#endif

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
        return *hostReadAt(i);
    }

    T& operator[] ( size_type i )
    {
        checkIndex ( i );
        return *hostWriteAt(i);
    }

    const T* data( ) const
    {
        checkIndex ( 0 );
        return hostReadAt(0);
    }

    T* data( )
    {
        checkIndex ( 0 );
        return hostWriteAt(0);
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

    iterator erase(iterator position)
    {
        iterator p0 = begin();
        size_type i = position - p0;
        size_type n = size();
        if (i >= n) return end();
        for (size_type j=i+1; j<n; ++j)
            *(p0+(j-1)) = *(p0+j);
        resize(n-1);
        return begin()+i;
    }

    iterator insert(iterator position, const T& x)
    {
        size_type i = position - begin();
        size_type n = size();
        if (i > n) i = n;
        resize(n+1);
        iterator p0 = begin();
        for (size_type j=n; j>i; --j)
            *(p0+j) = *(p0+(j-1));
        *(p0+i) = x;
        return p0+i;
    }

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
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        size_t alloc = deviceReserveSize;
#else
        size_t alloc = allocSize;
#endif
        if (deviceAllocSize[d] < alloc)
        {
            DEBUG_OUT_V(SPACEP << "allocate device=" << d << " " << deviceAllocSize[d] << "->" << alloc << std::endl);
            device_pointer prevDevicePointer = devicePointer[d];
            //COMM : if (mycudaVerboseLevel>=LOG_INFO) std::cout << "CudaVector<"<<sofa::core::objectmodel::Base::className((T*)NULL)<<"> : reserve("<<s<<")"<<std::endl;
            MemoryManager::deviceAlloc(d, &devicePointer[d], alloc*sizeof ( T ) );
            deviceAllocSize[d] = alloc;
            if (isDeviceValid(d))
            {
                if (deviceVectorSize[d] > 0)
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid vector) " << 0 << "->" << deviceVectorSize[d] << std::endl);
                    MemoryManager::memcpyDeviceToDevice (d, devicePointer[d], prevDevicePointer, deviceVectorSize[d]*sizeof ( T ) );
                }
            }
            if ( !MemoryManager::isNull(prevDevicePointer)) MemoryManager::deviceFree (d, prevDevicePointer );
            deviceAllocSize[d] = alloc;
            DEBUG_OUT_V(SPACEM << "allocate " << std::endl);
        }
        if (isDeviceValid(d) && deviceVectorSize[d] < vectorSize)
        {
            if (clearSize > deviceVectorSize[d])
            {
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetDevice (clear new data) " << deviceVectorSize[d] << "->" << clearSize-deviceVectorSize[d] << std::endl);
                MemoryManager::memsetDevice(d,MemoryManager::deviceOffset(devicePointer[d],deviceVectorSize[d]), 0, (clearSize-deviceVectorSize[d])*sizeof(T));
            }
            deviceVectorSize[d] = vectorSize; // the device now contains a fully valid copy
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

#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
#endif // SOFA_NO_OPENGL

        /// if host is not valid data are valid and allocated on a device
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            if (isDeviceValid(d))
            {
                if (deviceVectorSize[d] > 0)
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToHost " << deviceVectorSize[d] << std::endl);
                    MemoryManager::memcpyDeviceToHost (d, hostPointer, devicePointer[d], deviceVectorSize[d]*sizeof ( T ) );
                }
                if (clearSize > deviceVectorSize[d])
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost " << deviceVectorSize[d] << " -> " << clearSize-deviceVectorSize[d] << std::endl);
                    MemoryManager::memsetHost(hostPointer+deviceVectorSize[d],0,(clearSize-deviceVectorSize[d])*sizeof(T));
                }
                hostIsValid = true;
                break;
            }
        }
        if (!hostIsValid) // if no device had valid data, we assume the only valid data are zeros
        {
            if (clearSize > 0)
            {
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost " << 0 << " -> " << clearSize << std::endl);
                MemoryManager::memsetHost(hostPointer,0,clearSize*sizeof(T));
            }
            hostIsValid = true;
        }

        DEBUG_OUT_V(SPACEM << "copyToHost " << std::endl);
    }

    void copyToDevice(int d = 0) const
    {
        allocate(d);
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
#endif // SOFA_NO_OPENGL
        if (isDeviceValid(d)) return;
        DEBUG_OUT_V(SPACEP << "copyToDevice " << std::endl);

//#ifndef NDEBUG
        //COMM : if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: CPU->GPU copy of "<<sofa::core::objectmodel::Base::decodeTypeName ( typeid ( *this ) ) <<": "<<vectorSize*sizeof ( T ) <<" B"<<std::endl;
//#endif
        if ( !hostIsValid ) copyToHost();
        DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyHostToDevice " << vectorSize << std::endl);
        MemoryManager::memcpyHostToDevice (d, devicePointer[d], hostPointer, vectorSize*sizeof ( T ) );
        deviceIsValid |= 1<<d;
        deviceVectorSize[d] = vectorSize;
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
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
#endif // SOFA_NO_OPENGL
        for (int d=0; d<MemoryManager::numDevices(); d++)
        {
            if (isDeviceValid(d))
            {
                if (i<deviceVectorSize[d]) {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToHost " << i << std::endl);
                    MemoryManager::memcpyDeviceToHost(d, ((T*)hostPointer)+i, MemoryManager::deviceOffset(devicePointer[d],i), sizeof ( T ) );
                } else {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost " << i << std::endl);
                    MemoryManager::memsetHost(((T*)hostPointer)+i,0,sizeof(T));
                }
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
        //assert ( i<this->size() );
        if (i>=this->size())
            vector_access_failure(this, this->size(), i, typeid(T));
    }
#endif

    void registerBuffer() const
    {
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (allocSize > 0 && !bufferIsRegistered)
            {
                bufferIsRegistered = MemoryManager::bufferRegister(bufferObject);
            }
        }
#endif // SOFA_NO_OPENGL
    }

    void mapBuffer() const
    {
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            registerBuffer();
            if (bufferIsRegistered)
            {
                int dev = MemoryManager::getBufferDevice();
                if (allocSize > 0 && MemoryManager::isNull(devicePointer[dev]))
                {
                    MemoryManager::bufferMapToDevice((device_pointer*)&(devicePointer[dev]), bufferObject);
                }
            }
            else
            {
                std::cout << "CUDA: Unable to map buffer to opengl" << std::endl;
            }
        }
#endif // SOFA_NO_OPENGL
    }

    void unmapBuffer() const
    {
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            int dev = MemoryManager::getBufferDevice();
            if (allocSize > 0 && !MemoryManager::isNull(devicePointer[dev]))
            {
                MemoryManager::bufferUnmapToDevice((device_pointer*)&(devicePointer[dev]), bufferObject);
                *((device_pointer*) &(devicePointer[dev])) = MemoryManager::null();
            }
        }
#endif // SOFA_NO_OPENGL
    }

    void unregisterBuffer() const
    {
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            unmapBuffer();
            if (allocSize > 0 && bufferIsRegistered)
            {
                //MemoryManager::bufferUnregister(bufferObject);
                MemoryManager::bufferFree(bufferObject);
                bufferIsRegistered = false;
            }
        }
#endif // SOFA_NO_OPENGL
    }

    void createBuffer()
    {
#ifndef SOFA_NO_OPENGL
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (bufferObject) return;

            if (allocSize > 0)
            {
                int dev = MemoryManager::getBufferDevice();
                if (isDeviceValid(dev))
                    copyToDevice(dev);//make sure data is on device MemoryManager::getBufferDevice()

                MemoryManager::bufferAlloc(&bufferObject,allocSize*sizeof(T));
                void* prevDevicePointer = devicePointer[dev];
                devicePointer[dev] = NULL;
                if (vectorSize>0 && isDeviceValid(dev))
                {
                    mapBuffer();
                    if (!MemoryManager::isNull(prevDevicePointer))
                        MemoryManager::memcpyDeviceToDevice( dev, devicePointer[dev], prevDevicePointer, vectorSize*sizeof ( T ) );
                }
                if (!MemoryManager::isNull(prevDevicePointer))
                    MemoryManager::deviceFree(dev, prevDevicePointer);
            }
        }
        #endif // SOFA_NO_OPENGL
    }
};

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
#endif //ndef PS3
#endif //SOFA_HELPER_VECTOR_DEVICE_H
