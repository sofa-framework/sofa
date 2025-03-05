/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/type/config.h>
#include <sofa/type/trait/Rebind.h>

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

namespace sofa::type
{

template <class T, class MemoryManager, class DataTypeInfoManager>
class vector_device
{
public:
    typedef T      value_type;
    typedef size_t Size;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef typename MemoryManager::device_pointer device_pointer;
    typedef typename MemoryManager::buffer_id_type buffer_id_type;

    typedef MemoryManager memory_manager;
    typedef DataTypeInfoManager datatypeinfo_manager;

    template<class T2>
    using rebind_to = vector_device<T2,
                                    sofa::type::rebind_to<memory_manager, T2>,
                                    sofa::type::rebind_to<datatypeinfo_manager, T2> >;

protected:
    Size     vectorSize;     ///< Current size of the vector
    Size     allocSize;      ///< Allocated size on host
    mutable Size      deviceVectorSize[MemoryManager::MAX_DEVICES];      ///< Initialized size on each device
    mutable Size      deviceAllocSize[MemoryManager::MAX_DEVICES];      ///< Allocated size on each device
    mutable device_pointer devicePointer[MemoryManager::MAX_DEVICES];  ///< Pointer to the data on the GPU side
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
    mutable Size      deviceReserveSize;      ///< Desired allocated size
#endif
    mutable Size      clearSize;  ///< when initializing missing device data, up to where entries should be set to zero ?
    T* hostPointer;    ///< Pointer to the data on the CPU side
    mutable int   deviceIsValid;  ///< True if the data on the GPU is currently valid (up to the given deviceVectorSize of each device, i.e. additional space may need to be allocated and/or initialized)
    mutable bool  hostIsValid;    ///< True if the data on the CPU is currently valid
    mutable bool  bufferIsRegistered;  ///< True if the buffer is registered with CUDA
    buffer_id_type  bufferObject;   ///< Optional associated buffer ID

    inline static int cptid = 0;

    enum { ALL_DEVICE_VALID = 0xFFFFFFFF };

    DEBUG_OUT_V(int id;)
        DEBUG_OUT_V(mutable int spaceDebug;)

public:

    vector_device()
        : vector_device(0)
    {}

    explicit vector_device(const Size n)
        : vectorSize(0), allocSize(0), hostPointer(nullptr), deviceIsValid(ALL_DEVICE_VALID), hostIsValid(true), bufferIsRegistered(false)
        , bufferObject(0)
    {
        DEBUG_OUT_V(id = cptid);
        DEBUG_OUT_V(cptid++);
        DEBUG_OUT_V(spaceDebug = 0);
        for (int d = 0; d < MemoryManager::numDevices(); d++)
        {
            devicePointer[d] = MemoryManager::null();
            deviceAllocSize[d] = 0;
            deviceVectorSize[d] = 0;
        }
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        deviceReserveSize = 0;
#endif
        clearSize = 0;

        if (n > 0)
        {
            resize(n);
        }
    }

    vector_device(const vector_device<T, MemoryManager, DataTypeInfoManager>& v)
        : vector_device()
    {
        * this = v;
    }

    vector_device(const std::initializer_list<T>& t) : vector_device()
    {
        if (!std::empty(t))
        {
            fastResize(t.size());
            std::copy(t.begin(), t.end(), hostPointer);
        }
    }

    bool isHostValid() const
    {
        return hostIsValid;
    }
    bool isDeviceValid(unsigned gpu) const
    {
        return (deviceIsValid & (1 << gpu)) != 0;
    }

    void clear()
    {
        DEBUG_OUT_V(SPACEP << "clear vector" << std::endl);
        vectorSize = 0;
        hostIsValid = true;
        deviceIsValid = ALL_DEVICE_VALID;
        for (int d = 0; d < MemoryManager::numDevices(); d++)
            deviceVectorSize[d] = 0;
        clearSize = 0;
        DEBUG_OUT_V(SPACEM << "clear vector " << std::endl);
    }

    void operator= (const vector_device<T, MemoryManager, DataTypeInfoManager >& v)
    {
        if (&v == this)
        {
            return;
        }
        DEBUG_OUT_V(SPACEP << "operator=, id is " << v.id << "(" << v.hostIsValid << "," << (v.deviceIsValid & 1) << ") " << std::endl);
        DEBUG_OUT_V(std::cout << v.id << " : " << "(" << v.hostIsValid << "," << (v.deviceIsValid & 1) << ") " << ". operator= param " << id << std::endl);

        const Size newSize = v.size();
        clear();

        fastResize(newSize);

        if (vectorSize > 0)
        {
            if (v.hostIsValid) std::copy(v.hostPointer, v.hostPointer + vectorSize, hostPointer);

            clearSize = v.clearSize;
            if (v.deviceIsValid)
            {
                if (MemoryManager::SUPPORT_GL_BUFFER)
                {
                    if (bufferObject) mapBuffer();//COMM : necessaire????
                    if (v.bufferObject) v.mapBuffer();
                }

                deviceIsValid = 0; /// we specify that we don't want to copy previous value of the current vector
                for (int d = 0; d < MemoryManager::numDevices(); d++)
                {
                    if (v.isDeviceValid(d) && v.deviceVectorSize[d] > 0)
                    {
                        //v.allocate(d); /// make sure that the device data are correct
                        allocate(d); /// device are not valid so it only allocate

                        if (vectorSize <= v.deviceVectorSize[d])
                        {
                            DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy full vector) " << 0 << "->" << vectorSize << std::endl);
                            MemoryManager::memcpyDeviceToDevice(d, devicePointer[d], v.devicePointer[d], vectorSize * sizeof(T));
                            deviceVectorSize[d] = vectorSize;
                        }
                        else
                        {
                            DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid vector) " << 0 << "->" << v.deviceAllocSize[d] << std::endl);
                            MemoryManager::memcpyDeviceToDevice(d, devicePointer[d], v.devicePointer[d], v.deviceVectorSize[d] * sizeof(T));
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

    ~vector_device()
    {
        if (hostPointer != nullptr) MemoryManager::hostFree(hostPointer);

        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject)
        {
            unregisterBuffer();
            MemoryManager::bufferFree(bufferObject);
            devicePointer[MemoryManager::getBufferDevice()] = MemoryManager::null(); // already free
        }
        else
        {
            for (int d = 0; d < MemoryManager::numDevices(); d++)
            {
                if (!MemoryManager::isNull(devicePointer[d]))
                    MemoryManager::deviceFree(d, (devicePointer[d]));
            }
        }
    }

    Size size() const
    {
        return vectorSize;
    }

    Size capacity() const
    {
        return allocSize;
    }

    bool empty() const
    {
        return vectorSize == 0;
    }

    void reserve(Size s, Size WARP_SIZE = MemoryManager::BSIZE)
    {
        s = ((s + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        if (s > deviceReserveSize)
        {
            // we double the reserved size except if the requested size is bigger or if we would allocate more memory than the configured step size
            if (s <= 2 * deviceReserveSize && 2 * deviceReserveSize <= s + SOFA_VECTOR_HOST_STEP_SIZE)
                deviceReserveSize *= 2;
            else
                deviceReserveSize = s;
            // always allocate multiples of WARP_SIZE values
            deviceReserveSize = ((deviceReserveSize + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        }
#endif
        if (s <= allocSize) return;
        DEBUG_OUT_V(SPACEP << "reserve " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std
            ::endl);
        // we double the reserved size except if the requested size is bigger or if we would allocate more memory than the configured step size
        if (s <= 2 * allocSize && 2 * allocSize <= s + SOFA_VECTOR_HOST_STEP_SIZE)
            allocSize *= 2;
        else
            allocSize = s + SOFA_VECTOR_HOST_STEP_SIZE;
        // always allocate multiples of WARP_SIZE values
        allocSize = ((allocSize + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject)
        {
            DEBUG_OUT_V(SPACEN << "BUFFEROBJ " << std::endl);

            hostRead(); // make sure the host copy is valid
            unregisterBuffer();

            MemoryManager::bufferAlloc(&bufferObject, allocSize * sizeof(T), false);

            if (vectorSize > 0) deviceIsValid = 0;
        }

        T* prevHostPointer = hostPointer;
        void* newHostPointer = nullptr;
        DEBUG_OUT_V(SPACEN << "MemoryManager::hostAlloc " << allocSize << std::endl);
        MemoryManager::hostAlloc(&newHostPointer, allocSize * sizeof(T));
        hostPointer = (T*)newHostPointer;
        if (vectorSize != 0 && hostIsValid) std::copy(prevHostPointer, prevHostPointer + vectorSize, hostPointer);
        if (prevHostPointer != nullptr) MemoryManager::hostFree(prevHostPointer);
        DEBUG_OUT_V(SPACEM << "reserve " << " (alloc=" << allocSize << ")" << std::endl);
    }

    /// resize the vector without calling constructors or destructors, and without synchronizing the device and host copy
    void fastResize(Size s, Size WARP_SIZE = MemoryManager::BSIZE)
    {
        if (s == vectorSize) return;
        DEBUG_OUT_V(SPACEP << "fastresize " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std::endl);
        reserve(s, WARP_SIZE);
        vectorSize = s;
        if (clearSize > vectorSize) clearSize = vectorSize;
        for (int d = 0; d < MemoryManager::numDevices(); d++)
            if (deviceVectorSize[d] > vectorSize)
                deviceVectorSize[d] = vectorSize;
        if (!vectorSize)
        {
            // special case when the vector is now empty -> host and device are valid
            deviceIsValid = ALL_DEVICE_VALID;
            hostIsValid = true;
        }
        DEBUG_OUT_V(SPACEM << "fastresize " << std::endl);
    }
    /// resize the vector discarding any old values, without calling constructors or destructors, and without synchronizing the device and host copy
    void recreate(Size s, Size WARP_SIZE = MemoryManager::BSIZE)
    {
        clear();
        fastResize(s, WARP_SIZE);
    }

    void invalidateDevice()
    {
        hostIsValid = true;
        deviceIsValid = 0;
    }

    void invalidateHost()
    {
        hostIsValid = false;
        deviceIsValid = ALL_DEVICE_VALID;
    }

    void memsetDevice(int v = 0)
    {
        DEBUG_OUT_V(SPACEP << "memsetDevice " << std::endl);

        deviceIsValid = 0;
        clearSize = 0;
        for (int d = 0; d < MemoryManager::numDevices(); d++)
        {
            if (deviceAllocSize[d] > 0)   /// if the vector has already been used
            {
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetDevice " << deviceAllocSize[d] << std::endl);
                allocate(d); /// make sure the size is correct device is not valid so it only resize if necessary
                MemoryManager::memsetDevice(d, devicePointer[d], v, vectorSize * sizeof(T));
                deviceIsValid |= 1 << d;
            }
        }

        /// if we found at least one device valid we invalidate the host else the host is memset, device will be set at next copytodevice
        if (deviceIsValid) hostIsValid = false;
        else memsetHost(v);

        DEBUG_OUT_V(SPACEM << "memsetDevice " << std::endl);
    }

    void memsetHost(int v = 0)
    {
        MemoryManager::memsetHost(hostPointer, v, vectorSize * sizeof(T));
        clearSize = 0;
        hostIsValid = true;
        deviceIsValid = 0;
    }

    void resize(Size s, Size WARP_SIZE = MemoryManager::BSIZE)
    {
        reserve(s, WARP_SIZE);
        if (s == vectorSize) return;
        DEBUG_OUT_V(SPACEP << "resize " << vectorSize << "->" << s << " (alloc=" << allocSize << ")" << std::endl);
        if (s > vectorSize)
        {
            if (datatypeinfo_manager::ZeroConstructor)   // can use memset instead of constructors
            {
                if (hostIsValid)
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost (new data) " << (s - vectorSize) << std::endl);
                    MemoryManager::memsetHost(hostPointer + vectorSize, 0, (s - vectorSize) * sizeof(T));
                }
                clearSize = s;
#ifndef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
                size_t deviceReserveSize = allocSize;
#endif
                for (int d = 0; d < MemoryManager::numDevices(); d++)
                {
                    if (isDeviceValid(d))
                    {
                        if (deviceAllocSize[d] >= deviceReserveSize)
                        {
                            MemoryManager::memsetDevice(d, MemoryManager::deviceOffset(devicePointer[d], deviceVectorSize[d]), 0, (s - deviceVectorSize[d]) * sizeof(T));
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
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost (new data) " << (s - vectorSize) << std::endl);
                MemoryManager::memsetHost(hostPointer + vectorSize, 0, (s - vectorSize) * sizeof(T));
                // Call the constructor for the new elements
                for (Size i = vectorSize; i < s; i++) ::new (hostPointer + i) T;

                if (vectorSize == 0)   // wait until the transfer is really necessary, as other modifications might follow
                {
                    deviceIsValid = 0;
                }
                else
                {
                    if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();

                    for (int d = 0; d < MemoryManager::numDevices(); d++)
                    {
                        if (isDeviceValid(d))
                        {
                            if (deviceVectorSize[d] == 0) // no data is currently on the device -> we simply invalidate it and the transfer will happen once the data is actually requested
                                deviceIsValid &= ~(1 << d);
                            else
                            {
                                allocate(d);
                                DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyHostToDevice " << vectorSize << "->" << (s - vectorSize) << std::endl);
                                MemoryManager::memcpyHostToDevice(d, MemoryManager::deviceOffset(devicePointer[d], deviceVectorSize[d]), hostPointer + deviceVectorSize[d], (s - deviceVectorSize[d]) * sizeof(T));
                                deviceVectorSize[d] = s;
                            }
                        }
                    }
                }
            }
        }
        else if (s < vectorSize && !(datatypeinfo_manager::SimpleCopy))     // need to call destructors
        {
            DEBUG_OUT_V(SPACEN << "SIMPLECOPY " << std::endl);
            copyToHost();
            // Call the destructor for the deleted elements
            for (Size i = s; i < vectorSize; i++)
            {
                hostPointer[i].~T();
            }
        }
        vectorSize = s;
        if (clearSize > vectorSize) clearSize = vectorSize;
        for (int d = 0; d < MemoryManager::numDevices(); d++)
            if (deviceVectorSize[d] > vectorSize)
                deviceVectorSize[d] = vectorSize;

        if (!vectorSize)   // special case when the vector is now empty -> host and device are valid
        {
            deviceIsValid = ALL_DEVICE_VALID;
            hostIsValid = true;
        }

        DEBUG_OUT_V(SPACEM << "resize " << std::endl);
    }

    void swap(vector_device<T, MemoryManager, DataTypeInfoManager>& v)
    {
        DEBUG_OUT_V(SPACEP << "swap " << std::endl);
#define VSWAP(type, var) { type t = var; var = v.var; v.var = t; }
        VSWAP(Size, vectorSize);
        VSWAP(Size, allocSize);
        VSWAP(int, clearSize);
#ifdef SOFA_VECTOR_DEVICE_CUSTOM_SIZE
        VSWAP(int, deviceReserveSize);
#endif
        for (int d = 0; d < MemoryManager::numDevices(); d++)
        {
            VSWAP(void*, devicePointer[d]);
            VSWAP(int, deviceVectorSize[d]);
            VSWAP(int, deviceAllocSize[d]);
        }
        VSWAP(T*, hostPointer);
        VSWAP(buffer_id_type, bufferObject);
        VSWAP(int, deviceIsValid);
        VSWAP(bool, hostIsValid);
        VSWAP(bool, bufferIsRegistered);
#undef VSWAP
        DEBUG_OUT_V(SPACEM << "swap " << std::endl);
    }

    const device_pointer deviceReadAt(int i, int gpu = MemoryManager::getBufferDevice()) const
    {
        DEBUG_OUT_V(if (!(deviceIsValid & (1 << gpu))) { SPACEN << "deviceRead" << std::endl; });
        copyToDevice(gpu);
        return MemoryManager::deviceOffset(devicePointer[gpu], i);
    }

    const device_pointer deviceRead(int gpu = MemoryManager::getBufferDevice()) const
    {
        return deviceReadAt(0, gpu);
    }

    device_pointer deviceWriteAt(int i, int gpu = MemoryManager::getBufferDevice())
    {
        DEBUG_OUT_V(if (hostIsValid) { SPACEN << "deviceWrite" << std::endl; });
        copyToDevice(gpu);
        if (vectorSize > 0)
            hostIsValid = false;
        deviceIsValid = 1 << gpu;
        return MemoryManager::deviceOffset(devicePointer[gpu], i);
    }

    device_pointer deviceWrite(int gpu = MemoryManager::getBufferDevice())
    {
        return deviceWriteAt(0, gpu);
    }

    const T* hostRead() const
    {
        return hostReadAt(0);
    }

    T* hostWrite()
    {
        return hostWriteAt(0);
    }

    const T* hostReadAt(int i) const
    {
        DEBUG_OUT_V(if (!hostIsValid) { SPACEN << "hostRead" << std::endl; });
        copyToHost();
        return hostPointer + i;
    }

    T* hostWriteAt(int i)
    {
        DEBUG_OUT_V(if (deviceIsValid) { SPACEN << "hostWrite" << std::endl; });
        copyToHost();
        if (vectorSize > 0)
            deviceIsValid = false;
        return hostPointer + i;
    }

    /// Get the Buffer Object ID for reading
    buffer_id_type bufferRead(bool create = false)
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

    /// Get the Buffer Object ID for writing
    buffer_id_type bufferWrite(bool create = false)
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
            deviceIsValid = 1 << MemoryManager::getBufferDevice();
            return bufferObject;
        }
        else return 0;
    }

    void push_back(const T& t)
    {
        Size i = size();
        copyToHost();
        deviceIsValid = 0;
        fastResize(i + 1);
        ::new (hostPointer + i) T(t);
    }

    void pop_back()
    {
        if (!empty()) resize(size() - 1);
    }

    const T& operator[] (Size i) const
    {
        checkIndex(i);
        return *hostReadAt(i);
    }

    T& operator[] (Size i)
    {
        checkIndex(i);
        return *hostWriteAt(i);
    }

    const T* data() const
    {
        checkIndex(0);
        return hostReadAt(0);
    }

    T* data()
    {
        checkIndex(0);
        return hostWriteAt(0);
    }

    const T& getCached(Size i) const
    {
        checkIndex(i);
        return hostPointer[i];
    }

    const T& getSingle(Size i) const
    {
        copyToHostSingle(i);
        return hostPointer[i];
    }

    const_iterator begin() const { return hostRead(); }
    const_iterator end() const { return hostRead() + size(); }

    iterator begin() { return hostWrite(); }
    iterator end() { return hostWrite() + size(); }

    iterator erase(iterator position)
    {
        iterator p0 = begin();
        Size i = position - p0;
        const Size n = size();
        if (i >= n) return end();
        for (Size j = i + 1; j < n; ++j)
            *(p0 + (j - 1)) = *(p0 + j);
        resize(n - 1);
        return begin() + i;
    }

    iterator insert(iterator position, const T& x)
    {
        Size i = position - begin();
        const Size n = size();
        if (i > n) i = n;
        resize(n + 1);
        iterator p0 = begin();
        for (Size j = n; j > i; --j)
            *(p0 + j) = *(p0 + (j - 1));
        *(p0 + i) = x;
        return p0 + i;
    }

    /// Output stream
    inline friend std::ostream& operator<< (std::ostream& os, const vector_device<T, MemoryManager, DataTypeInfoManager>& vec)
    {
        if (vec.size() > 0)
        {
            for (unsigned int i = 0; i < vec.size() - 1; ++i) os << vec[i] << " ";
            os << vec[vec.size() - 1];
        }
        return os;
    }

    /// Input stream
    inline friend std::istream& operator>> (std::istream& in, vector_device<T, MemoryManager, DataTypeInfoManager>& vec)
    {
        T t;
        vec.clear();
        while (in >> t)
        {
            vec.push_back(t);
        }
        if (in.rdstate() & std::ios_base::eofbit) { in.clear(); }
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

            MemoryManager::deviceAlloc(d, &devicePointer[d], alloc * sizeof(T));
            deviceAllocSize[d] = alloc;
            if (isDeviceValid(d))
            {
                if (deviceVectorSize[d] > 0)
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToDevice(copy valid vector) " << 0 << "->" << deviceVectorSize[d] << std::endl);
                    MemoryManager::memcpyDeviceToDevice(d, devicePointer[d], prevDevicePointer, deviceVectorSize[d] * sizeof(T));
                }
            }
            if (!MemoryManager::isNull(prevDevicePointer)) MemoryManager::deviceFree(d, prevDevicePointer);
            deviceAllocSize[d] = alloc;
            DEBUG_OUT_V(SPACEM << "allocate " << std::endl);
        }
        if (isDeviceValid(d) && deviceVectorSize[d] < vectorSize)
        {
            if (clearSize > deviceVectorSize[d])
            {
                DEBUG_OUT_V(SPACEN << "MemoryManager::memsetDevice (clear new data) " << deviceVectorSize[d] << "->" << clearSize - deviceVectorSize[d] << std::endl);
                MemoryManager::memsetDevice(d, MemoryManager::deviceOffset(devicePointer[d], deviceVectorSize[d]), 0, (clearSize - deviceVectorSize[d]) * sizeof(T));
            }
            deviceVectorSize[d] = vectorSize; // the device now contains a fully valid copy
        }
    }

    void copyToHost() const
    {
        if (hostIsValid) return;
        DEBUG_OUT_V(SPACEP << "copyToHost " << std::endl);

        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();

        /// if host is not valid data are valid and allocated on a device
        for (int d = 0; d < MemoryManager::numDevices(); d++)
        {
            if (isDeviceValid(d))
            {
                if (deviceVectorSize[d] > 0)
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToHost " << deviceVectorSize[d] << std::endl);
                    MemoryManager::memcpyDeviceToHost(d, hostPointer, devicePointer[d], deviceVectorSize[d] * sizeof(T));
                }
                if (clearSize > deviceVectorSize[d])
                {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost " << deviceVectorSize[d] << " -> " << clearSize - deviceVectorSize[d] << std::endl);
                    MemoryManager::memsetHost(hostPointer + deviceVectorSize[d], 0, (clearSize - deviceVectorSize[d]) * sizeof(T));
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
                MemoryManager::memsetHost(hostPointer, 0, clearSize * sizeof(T));
            }
            hostIsValid = true;
        }

        DEBUG_OUT_V(SPACEM << "copyToHost " << std::endl);
    }

    void copyToDevice(int d = 0) const
    {
        allocate(d);
        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
        if (isDeviceValid(d)) return;
        DEBUG_OUT_V(SPACEP << "copyToDevice " << std::endl);

        if (!hostIsValid) copyToHost();
        DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyHostToDevice " << vectorSize << std::endl);
        MemoryManager::memcpyHostToDevice(d, devicePointer[d], hostPointer, vectorSize * sizeof(T));
        deviceIsValid |= 1 << d;
        deviceVectorSize[d] = vectorSize;
        DEBUG_OUT_V(SPACEM << "copyToDevice " << std::endl);
    }

    void copyToHostSingle(Size i) const
    {
        if (hostIsValid) return;
        DEBUG_OUT_V(SPACEP << "copyToHostSingle " << std::endl);
        if (MemoryManager::SUPPORT_GL_BUFFER && bufferObject) mapBuffer();
        for (int d = 0; d < MemoryManager::numDevices(); d++)
        {
            if (isDeviceValid(d))
            {
                if (i < deviceVectorSize[d]) {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memcpyDeviceToHost " << i << std::endl);
                    MemoryManager::memcpyDeviceToHost(d, ((T*)hostPointer) + i, MemoryManager::deviceOffset(devicePointer[d], i), sizeof(T));
                }
                else {
                    DEBUG_OUT_V(SPACEN << "MemoryManager::memsetHost " << i << std::endl);
                    MemoryManager::memsetHost(((T*)hostPointer) + i, 0, sizeof(T));
                }
                break;
            }
        }
        DEBUG_OUT_V(SPACEM << "copyToHostSingle " << std::endl);
    }

#ifdef NDEBUG
    void checkIndex(Size) const {}
#else
    void checkIndex(Size i) const
    {
        //assert ( i<this->size() );
        if (i >= this->size())
            vector_access_failure(this, this->size(), i, typeid(T));
    }
#endif

    void registerBuffer() const
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (allocSize > 0 && !bufferIsRegistered)
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
                int dev = MemoryManager::getBufferDevice();
                if (allocSize > 0 && MemoryManager::isNull(devicePointer[dev]))
                {
                    MemoryManager::bufferMapToDevice((device_pointer*)&(devicePointer[dev]), bufferObject);
                }
            }
            else
            {
                msg_error("vector (with gl buffer)") << " Unable to map buffer to opengl";
            }
        }
    }

    void unmapBuffer() const
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            int dev = MemoryManager::getBufferDevice();
            if (allocSize > 0 && !MemoryManager::isNull(devicePointer[dev]))
            {
                MemoryManager::bufferUnmapToDevice((device_pointer*)&(devicePointer[dev]), bufferObject);
                *((device_pointer*)&(devicePointer[dev])) = MemoryManager::null();
            }
        }
    }

    void unregisterBuffer() const
    {
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
    }

    void createBuffer()
    {
        if (MemoryManager::SUPPORT_GL_BUFFER)
        {
            if (bufferObject) return;

            if (allocSize > 0)
            {
                int dev = MemoryManager::getBufferDevice();
                if (isDeviceValid(dev))
                    copyToDevice(dev);//make sure data is on device MemoryManager::getBufferDevice()

                MemoryManager::bufferAlloc(&bufferObject, allocSize * sizeof(T));
                void* prevDevicePointer = devicePointer[dev];
                devicePointer[dev] = nullptr;
                if (vectorSize > 0 && isDeviceValid(dev))
                {
                    mapBuffer();
                    if (!MemoryManager::isNull(prevDevicePointer))
                        MemoryManager::memcpyDeviceToDevice(dev, devicePointer[dev], prevDevicePointer, vectorSize * sizeof(T));
                }
                if (!MemoryManager::isNull(prevDevicePointer))
                    MemoryManager::deviceFree(dev, prevDevicePointer);
            }
        }
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

} // namespace sofa::type
