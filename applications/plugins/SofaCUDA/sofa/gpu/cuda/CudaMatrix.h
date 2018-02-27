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
#ifndef SOFA_GPU_CUDA_CUDAMATRIX_H
#define SOFA_GPU_CUDA_CUDAMATRIX_H

//#include "host_runtime.h" // CUDA
#include "CudaTypes.h"
#include <iostream>

//#define DEBUG_OUT_MATRIX

#ifdef DEBUG_OUT_MATRIX
#define DEBUG_OUT_M(a) a
#define SPACEP std::cout << "(" << hostIsValid << "," << (deviceIsValid) << ") " ;for(int espaceaff=0;espaceaff<spaceDebug;espaceaff++) std::cout << "  ";spaceDebug++; std::cout << ">"
#define SPACEM std::cout << "(" << hostIsValid << "," << (deviceIsValid) << ") " ;spaceDebug--;for(int espaceaff=0;espaceaff<spaceDebug;espaceaff++) std::cout << "  "; std::cout << "<"
#define SPACEN std::cout << "(" << hostIsValid << "," << (deviceIsValid) << ") " ;for(int espaceaff=0;espaceaff<spaceDebug;espaceaff++) std::cout << "  "; std::cout << "|"
#else
#define DEBUG_OUT_M(a)
#endif


namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class T, class MemoryManager = CudaMemoryManager<T> >
class CudaMatrix
{
public:
    typedef CudaMatrix<T> Matrix;
    typedef T      value_type;
    typedef size_t size_type;

private:
    size_type    sizeX;     ///< Current size of the vector
    size_type    sizeY;     ///< Current size of the vector
    size_type    pitch_device;     ///< Row alignment on the GPU
    size_type    pitch_host;     ///< Row alignment on the GPU
    size_type    allocSizeX;  ///< Allocated size
    size_type    allocSizeY;  ///< Allocated size
    void*        devicePointer;  ///< Pointer to the data on the GPU side
    T*           hostPointer;    ///< Pointer to the data on the CPU side
    mutable bool deviceIsValid;  ///< True if the data on the GPU is currently valid
    mutable bool hostIsValid;    ///< True if the data on the CPU is currently valid
    DEBUG_OUT_M(mutable int spaceDebug;)
public:

    CudaMatrix()
        : sizeX ( 0 ), sizeY( 0 ), pitch_device(0), pitch_host(0),  allocSizeX(0), allocSizeY(0), devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {
        DEBUG_OUT_M(spaceDebug = 0);
    }

    CudaMatrix(size_t x, size_t y, size_t size)
        : sizeX ( 0 ), sizeY ( 0 ), pitch_device(0),pitch_host(0),  allocSizeX(0), allocSizeY(0),  devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {
        resize (x,y,size);
        DEBUG_OUT_M(spaceDebug = 0);
    }

    CudaMatrix(const CudaMatrix<T>& v )
        : sizeX ( 0 ), sizeY ( 0 ), pitch_device(0),pitch_host(0),  allocSizeX(0), allocSizeY(0),  devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {
        *this = v;
        DEBUG_OUT_M(spaceDebug = 0);
    }

    unsigned capacityX() const {
        return allocSizeX;
    }

    unsigned capacityY() const {
        return allocSizeY;
    }

    void resize_allocated(size_type y,size_t x,size_t WARP_SIZE) {
        size_type d_x = x;
        size_type d_y = y;

        if (WARP_SIZE==0) {
            d_x = x;
            d_y = y;
        } else {
            d_x = ((d_x+WARP_SIZE-1)/WARP_SIZE)*WARP_SIZE;
            d_y = ((d_y+WARP_SIZE-1)/WARP_SIZE)*WARP_SIZE;
        }

        DEBUG_OUT_M(SPACEN << "Ask resize from " << sizeX << "(" << allocSizeX << ")," << sizeY << "(" << allocSizeY << ")" << " to " << d_x << " " << d_y << std::endl);

        if (d_x <= allocSizeX) {
            //We keep the same pitch!

            if (d_y > allocSizeY) {
                DEBUG_OUT_M(SPACEN << "Is in d_y >= allocSizeY" << std::endl);

                allocSizeY = d_y;

                void* prevDevicePointer = devicePointer;
                T* prevHostPointer = hostPointer;

                size_t newpitch; // newpitch should be = to pitch_device !
                mycudaMallocPitch(&devicePointer, &newpitch, pitch_device, d_y);
                MemoryManager::hostAlloc( (void **) &hostPointer, pitch_host*d_y);

                if (sizeX!=0 && sizeY!=0) {
                    if (deviceIsValid) {
                        DEBUG_OUT_M(SPACEN << "MemcpyDevice from 0 to " << (pitch_device*sizeY) << std::endl);
                        MemoryManager::memcpyDeviceToDevice (0, devicePointer, prevDevicePointer, pitch_device*sizeY );
                    }
                    if (hostIsValid) {
                        DEBUG_OUT_M(SPACEN << "MemcpyHost from 0 to " << (pitch_host*sizeY) << std::endl);
                        std::copy ( prevHostPointer, (T*) (((char*)prevHostPointer)+(pitch_host*sizeY)), hostPointer);
                    }
                }

                if ( prevHostPointer != NULL ) MemoryManager::hostFree( prevHostPointer );
                if ( prevDevicePointer != NULL ) mycudaFree ( prevDevicePointer );
            }
        } else { //d_x > allocSizeX
            DEBUG_OUT_M(SPACEN << "Is in d_x >= allocSizeX" << std::endl);

            allocSizeX = d_x;
            if (d_y > allocSizeY) allocSizeY = d_y;

            void* prevDevicePointer = devicePointer;
            T* prevHostPointer = hostPointer;

            size_t oldpitch_device = pitch_device;
            size_t oldpitch_host = pitch_host;
            pitch_host = d_x * sizeof(T);// new pitch_host larger than oldpitch_host : guarantee that data on the host are continuous

            mycudaMallocPitch(&devicePointer, &pitch_device, d_x*sizeof(T), allocSizeY);// new pitch_device biger than oldpitch_device
            MemoryManager::hostAlloc( (void **) &hostPointer, pitch_host*allocSizeY);

            if (sizeX!=0 && sizeY!=0) {
                if (deviceIsValid && prevDevicePointer!= NULL) {
                    for (unsigned j=0;j<sizeY;j++) {
                        DEBUG_OUT_M(SPACEN << "MemcpyDevice from line " << j << " : from " << (oldpitch_device*j) << " to " << ((oldpitch_device*j) + (sizeX*sizeof(T))) << "(" << (sizeX*sizeof(T)) << " data)" << std::endl);
                        MemoryManager::memcpyDeviceToDevice (0, ((char*)devicePointer) + (pitch_device*j), ((char*)prevDevicePointer) + (oldpitch_device*j), sizeX * sizeof(T));
                    }
                }
                if (hostIsValid && prevHostPointer!= NULL) {
                    for (unsigned j=0;j<sizeY;j++) {
                        DEBUG_OUT_M(SPACEN << "MemcpyHost from line " << j << " : from " << (oldpitch_host*j) << " to " << ((oldpitch_host*j) + (sizeX*sizeof(T))) << "(" << (sizeX*sizeof(T)) << " data)" << std::endl);
                        std::copy ((T*) ((char*)prevHostPointer+ (oldpitch_host*j)), (T*) (((char*)prevHostPointer) + (oldpitch_host*j) + (sizeX*sizeof(T))), (T*) ((char*)hostPointer+ (pitch_host*j)));
                    }
                }
            }

            if ( prevHostPointer != NULL ) MemoryManager::hostFree( prevHostPointer );
            if ( prevDevicePointer != NULL ) mycudaFree ( prevDevicePointer );
        }
    }

    void clear() {
        DEBUG_OUT_M(SPACEP << "Clear" << std::endl);
        sizeX = 0;
        sizeY = 0;
        deviceIsValid = true;
        hostIsValid = true;
        DEBUG_OUT_M(SPACEM << "Clear" << std::endl);
    }

    ~CudaMatrix() {
        if (hostPointer!=NULL) mycudaFreeHost(hostPointer);
        if (devicePointer!=NULL) mycudaFree(devicePointer);
    }

    size_type getSizeX() const {
        return sizeX;
    }

    size_type getSizeY() const {
        return sizeY;
    }

    size_type getPitchDevice() const {
        return pitch_device;
    }

    size_type getPitchHost() const {
        return pitch_host;
    }

    bool isHostValid() const {
        return hostIsValid;
    }

    bool isDeviceValid() const {
        return deviceIsValid;
    }

    bool empty() const {
        return sizeX==0 || sizeY==0;
    }

    void memsetHost(int v = 0) {
        DEBUG_OUT_M(SPACEP << "memsetHost" << std::endl);
        MemoryManager::memsetHost(hostPointer,v,pitch_host*sizeY);
        hostIsValid = true;
        deviceIsValid = false;
        DEBUG_OUT_M(SPACEM << "memsetHost" << std::endl);
    }

    void memsetDevice(int v = 0) {
        DEBUG_OUT_M(SPACEP << "memsetHost" << std::endl);
        MemoryManager::memsetDevice(0,devicePointer, v, pitch_device*sizeY);
        hostIsValid = false;
        deviceIsValid = true;
        DEBUG_OUT_M(SPACEM << "memsetHost" << std::endl);
    }

    void invalidateDevices() {
        hostIsValid = true;
        deviceIsValid = false;
    }

    void invalidatehost() {
        hostIsValid = false;
        deviceIsValid = true;
    }

    void recreate(int nbRow,int nbCol) {
        clear();
        fastResize(nbRow,nbCol);
    }

    void fastResize(size_type y,size_type x,size_type WARP_SIZE=MemoryManager::BSIZE) {
        DEBUG_OUT_M(SPACEP << "fastResize : " << x << " " << y << " WArp_Size=" << WARP_SIZE << " sizeof(T)=" << sizeof(T) << std::endl);

        if ( x==0 || y==0) {
            clear();
            DEBUG_OUT_M(SPACEM << std::endl);
            return;
        }

        if ( sizeX==x && sizeY==y) {
            DEBUG_OUT_M(SPACEM << std::endl);
            return;
        }

        resize_allocated(y,x,WARP_SIZE);

        sizeX = x;
        sizeY = y;

        DEBUG_OUT_M(SPACEM << "fastResize" << std::endl);
    }

    void resize (size_type y,size_type x,size_t WARP_SIZE=MemoryManager::BSIZE) {
        DEBUG_OUT_M(SPACEP << "reisze : " << x << " " << y << " WArp_Size=" << WARP_SIZE << " sizeof(T)=" << sizeof(T) << std::endl);

        if ((x==0) || (y==0)) {
            clear();
            DEBUG_OUT_M(SPACEM << std::endl);
            return;
        }

        if ((sizeX==x) && (sizeY==y)) {
            DEBUG_OUT_M(SPACEM << std::endl);
            return;
        }

        if ( !sizeX && !sizeY) {//special case anly reserve
            DEBUG_OUT_M(SPACEN << "Is in ( !sizeX && !sizeY)" << std::endl);

            resize_allocated(y,x,WARP_SIZE);

            if (hostIsValid) {
                DEBUG_OUT_M(SPACEN << "MemsetHost from 0 to " << (pitch_host*y) << std::endl);
                //set all the matrix to zero
                MemoryManager::memsetHost(hostPointer,0,pitch_host*y);
            }

            if (deviceIsValid) {
                DEBUG_OUT_M(SPACEN << "MemsetDevice from 0 to " << (pitch_device*y) << std::endl);
                //set all the matrix to zero
                MemoryManager::memsetDevice(0,devicePointer, 0, pitch_device*y);
            }
        } else { // there is data in the matrix that we want to keep
            DEBUG_OUT_M(SPACEN << "Is in (x <= pitch)" << std::endl);

            resize_allocated(y,x,WARP_SIZE);

            if (x>sizeX) {
                if (hostIsValid) {
                    DEBUG_OUT_M(SPACEN << "MemsetHost from " << pitch_host*sizeY << " to " << (pitch_host*y) << std::endl);
                    //set all the end of line to Zero
                    for (unsigned j=0;j<sizeY;j++) MemoryManager::memsetHost((T*) (((char*)hostPointer)+pitch_host*j),0,(x-sizeX)*sizeof(T));
                }

                if (deviceIsValid) {
                    DEBUG_OUT_M(SPACEN << "MemsetDevice from " << pitch_device*sizeY << " to " << (pitch_device*y) << std::endl);
                    //set all the end of line to Zero
                    for (unsigned j=0;j<sizeY;j++) MemoryManager::memsetDevice(0,((char*)devicePointer) + (pitch_device*j), 0, (x-sizeX)*sizeof(T));
                }
            }

            if (y>sizeY) {
                if (hostIsValid) {
                    DEBUG_OUT_M(SPACEN << "MemsetHost from " << pitch_host*sizeY << " to " << (pitch_host*y) << std::endl);
                    //set the end of the matrix to zero
                    MemoryManager::memsetHost((T*) (((char*)hostPointer)+pitch_host*sizeY),0,pitch_host*(y-sizeY));
                }

                if (deviceIsValid) {
                    DEBUG_OUT_M(SPACEN << "MemsetDevice from " << pitch_device*sizeY << " to " << (pitch_device*y) << std::endl);
                    //set the end of the matrix to zero
                    MemoryManager::memsetDevice(0,((char*)devicePointer) + (pitch_device*sizeY), 0, pitch_device*(y-sizeY));
                }
            }
        }

        sizeX = x;
        sizeY = y;

        DEBUG_OUT_M(SPACEM << "reisze" << std::endl);
    }

    void swap ( CudaMatrix<T>& v ) {
#define VSWAP(type, var) { type t = var; var = v.var; v.var = t; }
        VSWAP ( size_type, sizeX );
        VSWAP ( size_type, sizeY );
        VSWAP ( size_type, pitch_device );
        VSWAP ( size_type, pitch_host );
        VSWAP ( size_type, allocSizeX );
        VSWAP ( size_type, allocSizeY );
        VSWAP ( void*    , devicePointer );
        VSWAP ( T*       , hostPointer );
        VSWAP ( bool     , deviceIsValid );
        VSWAP ( bool     , hostIsValid );
#undef VSWAP
    }

    void operator= ( const CudaMatrix<T,MemoryManager >& m ) {
        if (&m == this) return;

        std::cerr << "operator= is not handeled, you have to copy data manually" << std::endl;
//        sizeX = m.sizeX;
//        sizeY = m.sizeY;

//        if (sizeY*pitch_host<m.sizeY*m.pitch_host)   //simple case, we simply copy data with the same attribute
//        {
//            T* prevHostPointer = hostPointer;
//            MemoryManager::hostAlloc( (void **) &hostPointer, m.pitch_host * sizeY);
//            if ( prevHostPointer != NULL ) MemoryManager::hostFree( prevHostPointer );

//            void* prevDevicePointer = devicePointer;
//            mycudaMallocPitch(&devicePointer, &pitch_device, m.pitch_device, sizeY);
//            if ( prevDevicePointer != NULL ) mycudaFree ( prevDevicePointer );

//            allocSizeY = sizeY;
//        } else {
//            int allocline = (allocSizeY*pitch_host) / m.pitch_host;
//            allocSizeY = allocline * m.pitch_host;
//            // Here it's possible that the allocSizeY is < the the real memory allocated, but it's not a problem, it will we deleted at the next resize;
//        }

//        pitch_host = m.pitch_host;
//        pitch_device = m.pitch_device;

//        if (m.hostIsValid) std::copy ( m.hostPointer, ((T*) (((char*) m.hostPointer)+(m.pitch_host*m.sizeY))), hostPointer);
//        if (m.deviceIsValid) MemoryManager::memcpyDeviceToDevice (0, devicePointer, m.devicePointer, m.pitch_device*m.sizeY );

//        hostIsValid = m.hostIsValid;
//        deviceIsValid = m.deviceIsValid; /// finally we get the correct device valid
    }

    const void* deviceRead ( int y=0, int x=0 ) const {
        copyToDevice();
        return ((const T*) (((const char*)devicePointer) + pitch_device*y)) + x;
    }

    void* deviceWrite ( int y=0, int x=0 ) {
        copyToDevice();
        hostIsValid = false;
        return ((T*) (((char*)devicePointer) + pitch_device*y)) + x;
    }

    const T* hostRead ( int y=0, int x=0 ) const {
        copyToHost();
        return ((const T*) (((const char*) hostPointer) + pitch_host*y)) + x;
    }

    T* hostWrite ( int y=0, int x=0 ) {
        copyToHost();
        deviceIsValid = false;
        return ((T*) (((char*) hostPointer) + pitch_host*y)) + x;
    }

    T getSingle( int y=0, int x=0 ) const {
        copyToHostSingle(y,x);
        return ((const T*) (((const char*) hostPointer) + pitch_host*y))[x];
    }

    const T& operator() (size_type y,size_type x) const {
#ifdef DEBUG_OUT_MATRIX
        checkIndex (y,x);
#endif
        return hostRead(y,x);
    }

    T& operator() (size_type y,size_type x) {
#ifdef DEBUG_OUT_MATRIX
        checkIndex (y,x);
#endif
        return hostWrite(y,x);
    }

    const T* operator[] (size_type y) const {
#ifdef DEBUG_OUT_MATRIX
        checkIndex (y,0);
#endif
        return hostRead(y,0);
    }

    T* operator[] (size_type y) {
#ifdef DEBUG_OUT_MATRIX
        checkIndex (y,0);
#endif
        return hostWrite(y,0);
    }

    const T& getCached (size_type y,size_type x) const {
#ifdef DEBUG_OUT_MATRIX
        checkIndex (y,x);
#endif
        return ((T*) (((char*) hostPointer)+(y*pitch_host))) + x;
    }

    friend std::ostream& operator<< ( std::ostream& os, const Matrix & mat ) {
        mat.hostRead();
        os << "[\n";
        for (unsigned j=0; j<mat.getSizeY(); j++)
        {
            os << "[ ";
            for (unsigned i=0; i<mat.getSizeX(); i++)
            {
                os << " " << mat[j][i];
            }
            os << "]\n";
        }
        os << "]\n";
        return os;
    }

protected:
    void copyToHost() const {
        if ( hostIsValid ) return;
        DEBUG_OUT_M(SPACEN << "copyToHost" << std::endl);

//#ifndef NDEBUG
        if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: GPU->CPU copy of "<<sofa::core::objectmodel::BaseClass::decodeTypeName ( typeid ( *this ) ) <<": "<<sizeX*sizeof(T) <<" B"<<std::endl;
//#endif
        DEBUG_OUT_M(SPACEN << "copyToHost host : " << ((unsigned long) hostPointer) << " pitchH : " << pitch_host << " | device : " << ((unsigned long)devicePointer) << " pitchD : " << pitch_device << " | (" << sizeX*sizeof(T) << "," << sizeY << ")" << std::endl);
        mycudaMemcpyDeviceToHost2D ( hostPointer, pitch_host, devicePointer, pitch_device, sizeX*sizeof(T), sizeY);
        hostIsValid = true;
    }

    void copyToHostSingle(size_type y,size_type x) const
    {
        if ( hostIsValid ) return;
        mycudaMemcpyDeviceToHost(((T*)(((char *) hostPointer)+(pitch_host*y))) + x, ((T*)(((char *) devicePointer)+(pitch_device*y))) + x, sizeof ( T ) );
    }

    void copyToDevice() const {        
        if ( deviceIsValid ) return;


//#ifndef NDEBUG
        if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: CPU->GPU copy of "<<sofa::core::objectmodel::BaseClass::decodeTypeName ( typeid ( *this ) ) <<": "<<sizeX*sizeof(T) <<" B"<<std::endl;
//#endif
        DEBUG_OUT_M(SPACEN << "copyToDevice device : " << ((unsigned long)devicePointer) << " pitchD : " << pitch_device << " | host : " << ((unsigned long) hostPointer) << " pitchH : " << pitch_host << " | (" << sizeX*sizeof(T) << "," << sizeY << ")" << std::endl);
        mycudaMemcpyHostToDevice2D ( devicePointer, pitch_device, hostPointer, pitch_host,  sizeX*sizeof(T), sizeY);
        deviceIsValid = true;
    }

#ifdef DEBUG_OUT_MATRIX
    void checkIndex ( size_type y,size_type x) const {
        if (x>=sizeX) assert(0);
        if (y>=sizeY) assert(0);
    }
#endif
};

#ifdef DEBUG_OUT_MATRIX
#undef DEBUG_OUT_M
#undef SPACEP
#undef SPACEM
#undef SPACEN
#undef DEBUG_OUT_MATRIX
#else
#undef DEBUG_OUT_M
#endif

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
