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
#ifndef SOFA_HELPER_MEMORYMANAGER_H
#define SOFA_HELPER_MEMORYMANAGER_H

#include <sofa/helper/helper.h>
#include <cstring>

namespace sofa
{

namespace helper
{

#ifndef MAXIMUM_NUMBER_OF_DEVICES
#define MAXIMUM_NUMBER_OF_DEVICES 8
#endif

//Generic MemoryManager
template <class T>
class MemoryManager
{
    typedef T* host_pointer;
    typedef void* device_pointer;

    enum { MAX_DEVICES = MAXIMUM_NUMBER_OF_DEVICES };

    static int numDevices();

    host_pointer hostAlloc(int n) { return new T[n]; }
    void memsetHost(host_pointer hPointer, int value,size_t n) { memset((void*) hPointer, value, n); }

    device_pointer deviceAlloc(int d, int n) { return NULL; }
    void memcpyHostToDevice(int d, device_pointer dDestPointer, const host_pointer hSrcPointer, size_t n) { return ;}
    void memcpyDeviceToHost(int d, host_pointer hDestPointer, const device_pointer dSrcPointer , size_t n) { return ;}
    void memcpyDeviceToDevice(int dDest, int dSrc, device_pointer dDestPointer, const device_pointer dSrcPointer , size_t n) { return ;}
    void memsetDevice(int d, int value,size_t n) { return ;}
};

//CPU MemoryManager
template <class T >
class CPUMemoryManager : public MemoryManager<T>
{
public:

    typedef T* host_pointer;
    typedef void* device_pointer;

    static int numDevices() { return 0 ; }
};

}

}

#endif //SOFA_HELPER_MEMORYMANAGER_H


