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
#ifndef SOFA_HELPER_MEMORYMANAGER_H
#define SOFA_HELPER_MEMORYMANAGER_H

#include <sofa/helper/config.h>
#include <sofa/type/config.h>
#include <cstring>

namespace sofa::helper
{

#ifndef MAXIMUM_NUMBER_OF_DEVICES
#define MAXIMUM_NUMBER_OF_DEVICES 8
#endif

/** Generic MemoryManager
 * Its use is informative only and it cannot be instancied (linkage error otherwise).
 */
template <class T>
class MemoryManager
{
public :
    typedef T* host_pointer;

    //have to be changed according of the type of device
    typedef void* device_pointer;

    typedef unsigned int buffer_id_type;

    enum { MAX_DEVICES = 0 };
    enum { BSIZE = 32 };
    enum { SUPPORT_GL_BUFFER = 0 };

    static int numDevices();
    //
    static void hostAlloc(host_pointer* hPointer,int n) { *hPointer = new T[n/sizeof(T)]; }
    static void memsetHost(host_pointer hPointer, int value,size_t n) { memset((void*) hPointer, value, n); }
    static void hostFree(const host_pointer hSrcPointer);

    static void deviceAlloc(int d,device_pointer* dPointer, int n);
    static void deviceFree(int d,const device_pointer dSrcPointer);
    static void memcpyHostToDevice(int d, device_pointer dDestPointer, const host_pointer hSrcPointer, size_t n);
    static void memcpyDeviceToHost(int d, host_pointer hDestPointer, const void * dSrcPointer , size_t n);
    static void memcpyDeviceToDevice(int d, device_pointer dDestPointer, const device_pointer dSrcPointer , size_t n);
    static void memsetDevice(int d, device_pointer dDestPointer, int value,size_t n);

    static int getBufferDevice();

    static bool bufferAlloc(buffer_id_type* /*bId*/, int /*n*/, bool /* createBuffer = true */) { return false; }
    static void bufferFree(const buffer_id_type /*bId*/) {}

    static bool bufferRegister(const buffer_id_type /*bId*/) { return false; }
    static void bufferUnregister(const buffer_id_type /*bId*/) {}
    static bool bufferMapToDevice(device_pointer* /*dDestPointer*/, const buffer_id_type /*bSrcId*/) { return false; }
    static void bufferUnmapToDevice(device_pointer* /*dDestPointer*/, const buffer_id_type /*bSrcId*/) {}

    static device_pointer deviceOffset(device_pointer dPointer,size_t offset) {return (T*)dPointer+offset;}

    static device_pointer null() {return nullptr;}
    static bool isNull(device_pointer p) {return p==nullptr;}
};

}

#endif //SOFA_HELPER_MEMORYMANAGER_H
