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
#ifndef SOFA_GPU_CUDA_CUDATYPES_H
#define SOFA_GPU_CUDA_CUDATYPES_H

//#include "host_runtime.h" // CUDA
#include "CudaCommon.h"
#include "mycuda.h"
#include <sofa/helper/system/gl.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/accessor.h>
//#include <sofa/helper/BackTrace.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <sofa/defaulttype/SparseConstraintTypes.h>
#include <iostream>
#include <sofa/gpu/cuda/CudaMemoryManager.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class T>
class CudaVector : public helper::vector<T,CudaMemoryManager<T> >
{
public :
    typedef size_t size_type;

    CudaVector() : helper::vector<T,CudaMemoryManager<T> >() {}

    CudaVector(size_type n) : helper::vector<T,CudaMemoryManager<T> >(n) {}

    CudaVector(const helper::vector<T,CudaMemoryManager< T > >& v) : helper::vector<T,CudaMemoryManager<T> >(v) {}

};

template<class T>
class CudaMatrix
{
public:
    typedef T      value_type;
    typedef size_t size_type;

protected:
    size_type    sizeX;     ///< Current size of the vector
    size_type    sizeY;     ///< Current size of the vector
    size_type    pitch;     ///< Row alignment on the GPU
    size_type    hostAllocSize;  ///< Allocated size
    size_type    deviceAllocSize;///< Allocated size
    void*        devicePointer;  ///< Pointer to the data on the GPU side
    T*           hostPointer;    ///< Pointer to the data on the CPU side
    mutable bool deviceIsValid;  ///< True if the data on the GPU is currently valid
    mutable bool hostIsValid;    ///< True if the data on the CPU is currently valid
public:

    CudaMatrix()
        : sizeX ( 0 ), sizeY( 0 ), pitch(0), hostAllocSize ( 0 ), deviceAllocSize ( 0 ), devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {}

    CudaMatrix(size_t x, size_t y, size_t size)
        : sizeX ( 0 ), sizeY ( 0 ), pitch(0), hostAllocSize ( 0 ), deviceAllocSize ( 0 ), devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {
        resize (x,y,size);
    }

    CudaMatrix(const CudaMatrix<T>& v )
        : sizeX ( 0 ), sizeY ( 0 ), pitch(0), hostAllocSize ( 0 ), deviceAllocSize ( 0 ), devicePointer ( NULL ), hostPointer ( NULL ), deviceIsValid ( true ), hostIsValid ( true )
    {
        *this = v;
    }

    void clear()
    {
        sizeY = 0;
        sizeY = 0;
        pitch = 0;
        deviceIsValid = true;
        hostIsValid = true;
    }

    ~CudaMatrix()
    {
        if (hostPointer!=NULL) mycudaFreeHost(hostPointer);
        if (devicePointer!=NULL) mycudaFree(devicePointer);
    }

    size_type getSizeX() const
    {
        return sizeX;
    }

    size_type getSizeY() const
    {
        return sizeY;
    }

    size_type getPitch() const
    {
        return pitch;
    }
    bool empty() const
    {
        return sizeX==0 || sizeY==0;
    }

    void memsetHost(int v = 0)
    {
        memset(hostPointer,v,sizeY*sizeX*sizeof(T));
        hostIsValid = true;
        deviceIsValid = false;
    }

    void invalidateDevices()
    {
        hostIsValid = true;
        deviceIsValid = false;
    }

    void invalidatehost()
    {
        hostIsValid = false;
        deviceIsValid = true;
    }

    void fastResize(size_type x,size_type y,size_type WARP_SIZE)
    {
        size_type s = x*y;

        if (s > hostAllocSize)
        {
            hostAllocSize = ( s>2*hostAllocSize || s > 1024*1024 ) ? s : 2*hostAllocSize;
            // always allocate multiples of BSIZE values
            //hostAllocSize = ( hostAllocSize+WARP_SIZE-1 )/WARP_SIZE * WARP_SIZE;
            T* prevHostPointer = hostPointer;

            if ( prevHostPointer != NULL ) mycudaFreeHost ( prevHostPointer );
            void* newHostPointer = NULL;
            mycudaMallocHost ( &newHostPointer, hostAllocSize*sizeof ( T ) );
            hostPointer = (T*)newHostPointer;
        }


        if (WARP_SIZE==0) pitch = x*sizeof(T);
        else pitch = ((x+WARP_SIZE-1)/WARP_SIZE)*WARP_SIZE*sizeof(T);

        size_type ypitch = y;
        if (WARP_SIZE==0) ypitch = y;
        else ypitch = ((y+WARP_SIZE-1)/WARP_SIZE)*WARP_SIZE;

        if ( ypitch*pitch > deviceAllocSize )
        {
            if (ypitch < 2 * ((deviceAllocSize+pitch-1) / pitch) && ypitch < 1024)
            {
                ypitch = 2 * ((deviceAllocSize+pitch-1) / pitch);
            }

            void* prevDevicePointer = devicePointer;
            if (prevDevicePointer != NULL ) mycudaFree ( prevDevicePointer );

            mycudaMallocPitch(&devicePointer, &pitch, pitch, ypitch);
            deviceAllocSize = ypitch*pitch;
        }
        /*
        if (y*x > deviceAllocSize) {
         void* prevDevicePointer = devicePointer;
         if (prevDevicePointer != NULL ) mycudaFree ( prevDevicePointer );

         mycudaMallocPitch(&devicePointer, &pitch, x*sizeof(T), y);
         deviceAllocSize = y*x;
        }
        */
        sizeX = x;
        sizeY = y;
        deviceIsValid = true;
        hostIsValid = true;
    }

    void resize (size_type x,size_type y,size_t WARP_SIZE=BSIZE)
    {
        fastResize(x,y,WARP_SIZE);
    }

    void swap ( CudaMatrix<T>& v )
    {
#define VSWAP(type, var) { type t = var; var = v.var; v.var = t; }
        VSWAP ( size_type, sizeX );
        VSWAP ( size_type, sizeY );
        VSWAP ( size_type, pitch );
        VSWAP ( size_type, hostAllocSize );
        VSWAP ( size_type, deviceAllocSize );
        VSWAP ( void*    , devicePointer );
        VSWAP ( T*       , hostPointer );
        VSWAP ( bool     , deviceIsValid );
        VSWAP ( bool     , hostIsValid );
#undef VSWAP
    }

    const void* deviceRead ( int x=0, int y=0 ) const
    {
        copyToDevice();
        return (const T*)(( ( const char* ) devicePointer ) +(y*pitch+x*sizeof(T)));
    }

    void* deviceWrite ( int x=0, int y=0 )
    {
        copyToDevice();
        hostIsValid = false;
        return (T*)(( ( const char* ) devicePointer ) +(y*pitch+x*sizeof(T)));
    }

    const T* hostRead ( int x=0, int y=0 ) const
    {
        copyToHost();
        return hostPointer+(y*sizeX+x);
    }

    T* hostWrite ( int x=0, int y=0 )
    {
        copyToHost();
        deviceIsValid = false;
        return hostPointer+(y*sizeX+x);
    }

    bool isHostValid() const
    {
        return hostIsValid;
    }

    bool isDeviceValid() const
    {
        return deviceIsValid;
    }

    const T& operator() (size_type x,size_type y) const
    {
        checkIndex (x,y);
        return hostRead(x,y);
    }

    T& operator() (size_type x,size_type y)
    {
        checkIndex (x,y);
        return hostWrite(x,y);
    }

    const T* operator[] (size_type y) const
    {
        checkIndex (0,y);
        return hostRead(0,y);
    }

    T* operator[] (size_type y)
    {
        checkIndex (0,y);
        return hostWrite(0,y);
    }

    const T& getCached (size_type x,size_type y) const
    {
        checkIndex (x,y);
        return hostPointer[y*sizeX+x];
    }

protected:
    void copyToHost() const
    {
        if ( hostIsValid ) return;
//#ifndef NDEBUG
        if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: GPU->CPU copy of "<<sofa::core::objectmodel::Base::decodeTypeName ( typeid ( *this ) ) <<": "<<sizeX*sizeof(T) <<" B"<<std::endl;
//#endif
        mycudaMemcpyDeviceToHost2D ( hostPointer, sizeX*sizeof(T), devicePointer, pitch, sizeX*sizeof(T), sizeY);
        hostIsValid = true;
    }
    void copyToDevice() const
    {
        if ( deviceIsValid ) return;
//#ifndef NDEBUG
        if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: CPU->GPU copy of "<<sofa::core::objectmodel::Base::decodeTypeName ( typeid ( *this ) ) <<": "<<sizeX*sizeof(T) <<" B"<<std::endl;
//#endif
        mycudaMemcpyHostToDevice2D ( devicePointer, pitch, hostPointer, sizeX*sizeof(T),  sizeX*sizeof(T), sizeY);
        deviceIsValid = true;
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

template<class TCoord, class TDeriv, class TReal = typename TCoord::value_type>
class CudaVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef TReal Real;
    typedef CudaVector<Coord> VecCoord;
    typedef CudaVector<Deriv> VecDeriv;
    typedef CudaVector<Real> VecReal;
    typedef defaulttype::MapMapSparseMatrix<Deriv> MatrixDeriv;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef Coord CPos;
    static const CPos& getCPos(const Coord& c) { return c; }
    static void setCPos(Coord& c, const CPos& v) { c = v; }
    typedef Deriv DPos;
    static const DPos& getDPos(const Deriv& d) { return d; }
    static void setDPos(Deriv& d, const DPos& v) { d = v; }

    template<class C, typename T>
    static void set( C& c, T x, T y, T z )
    {
        if ( c.size() >0 )
            c[0] = (Real) x;
        if ( c.size() >1 )
            c[1] = (Real) y;
        if ( c.size() >2 )
            c[2] = (Real) z;
    }

    template<class C, typename T>
    static void get( T& x, T& y, T& z, const C& c )
    {
        x = ( c.size() >0 ) ? (T) c[0] : (T) 0.0;
        y = ( c.size() >1 ) ? (T) c[1] : (T) 0.0;
        z = ( c.size() >2 ) ? (T) c[2] : (T) 0.0;
    }

    template<class C, typename T>
    static void add( C& c, T x, T y, T z )
    {
        if ( c.size() >0 )
            c[0] += (Real) x;
        if ( c.size() >1 )
            c[1] += (Real) y;
        if ( c.size() >2 )
            c[2] += (Real) z;
    }

    template<class C>
    static C interpolate(const helper::vector< C > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        C coord;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            coord += ancestors[i] * coefs[i];
        }

        return coord;
    }

    static const char* Name();
};

typedef sofa::defaulttype::Vec3f Vec3f;
typedef sofa::defaulttype::Vec2f Vec2f;

using defaulttype::Vec;
using defaulttype::NoInit;
using defaulttype::NOINIT;

template<class Real>
class Vec3r1 : public sofa::defaulttype::Vec<3,Real>
{
public:
    typedef sofa::defaulttype::Vec<3,Real> Inherit;
    typedef Real real;
    enum { N=3 };
    Vec3r1() : dummy(0.0f) {}
    template<class real2>
    Vec3r1(const Vec<N,real2>& v): Inherit(v), dummy(0.0f) {}
    Vec3r1(real x, real y, real z) : Inherit(x,y,z), dummy(0.0f) {}

    /// Fast constructor: no initialization
    explicit Vec3r1(NoInit n) : Inherit(n), dummy(0.0f)
    {
    }

    // LINEAR ALGEBRA

    /// Multiplication by a scalar f.
    template<class real2>
    Vec3r1 operator*(real2 f) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i] = this->elems[i]*(real)f;
        return r;
    }

    /// On-place multiplication by a scalar f.
    template<class real2>
    void operator*=(real2 f)
    {
        for (int i=0; i<N; i++)
            this->elems[i]*=(real)f;
    }

    template<class real2>
    Vec3r1 operator/(real2 f) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i] = this->elems[i]/(real)f;
        return r;
    }

    /// On-place division by a scalar f.
    template<class real2>
    void operator/=(real2 f)
    {
        for (int i=0; i<N; i++)
            this->elems[i]/=(real)f;
    }

    /// Dot product.
    template<class real2>
    real operator*(const Vec<N,real2>& v) const
    {
        real r = (real)(this->elems[0]*v[0]);
        for (int i=1; i<N; i++)
            r += (real)(this->elems[i]*v[i]);
        return r;
    }


    /// Dot product.
    template<class real2>
    inline friend real operator*(const Vec<N,real2>& v1, const Vec3r1<real>& v2)
    {
        real r = (real)(v1[0]*v2[0]);
        for (int i=1; i<N; i++)
            r += (real)(v1[i]*v2[i]);
        return r;
    }


    /// Dot product.
    real operator*(const Vec3r1& v) const
    {
        real r = (real)(this->elems[0]*v[0]);
        for (int i=1; i<N; i++)
            r += (real)(this->elems[i]*v[i]);
        return r;
    }

    /// linear product.
    template<class real2>
    Vec3r1 linearProduct(const Vec<N,real2>& v) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]*(real)v[i];
        return r;
    }

    /// linear product.
    Vec3r1 linearProduct(const Vec3r1& v) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]*(real)v[i];
        return r;
    }

    /// Vector addition.
    template<class real2>
    Vec3r1 operator+(const Vec<N,real2>& v) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]+(real)v[i];
        return r;
    }

    /// Vector addition.
    Vec3r1 operator+(const Vec3r1& v) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]+(real)v[i];
        return r;
    }

    /// On-place vector addition.
    template<class real2>
    void operator+=(const Vec<N,real2>& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]+=(real)v[i];
    }

    /// On-place vector addition.
    void operator+=(const Vec3r1& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]+=(real)v[i];
    }

    /// Vector subtraction.
    template<class real2>
    Vec3r1 operator-(const Vec<N,real2>& v) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]-(real)v[i];
        return r;
    }

    /// Vector subtraction.
    Vec3r1 operator-(const Vec3r1& v) const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]-(real)v[i];
        return r;
    }

    /// On-place vector subtraction.
    template<class real2>
    void operator-=(const Vec<N,real2>& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]-=(real)v[i];
    }

    /// On-place vector subtraction.
    void operator-=(const Vec3r1& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]-=(real)v[i];
    }

    /// Vector negation.
    Vec3r1 operator-() const
    {
        Vec3r1 r(NOINIT);
        for (int i=0; i<N; i++)
            r[i]=-this->elems[i];
        return r;
    }

    Vec3r1 cross( const Vec3r1& b ) const
    {
        BOOST_STATIC_ASSERT(N == 3);
        return Vec3r1(
                (*this)[1]*b[2] - (*this)[2]*b[1],
                (*this)[2]*b[0] - (*this)[0]*b[2],
                (*this)[0]*b[1] - (*this)[1]*b[0]
                );
    }

protected:
    Real dummy;
};

typedef Vec3r1<float> Vec3f1;

typedef CudaVectorTypes<Vec3f,Vec3f,float> CudaVec3fTypes;
typedef CudaVec3fTypes CudaVec3Types;

template<>
inline const char* CudaVec3fTypes::Name()
{
    return "CudaVec3f";
}

typedef CudaVectorTypes<Vec2f,Vec2f,float> CudaVec2fTypes;
typedef CudaVec2fTypes CudaVec2Types;

template<>
inline const char* CudaVec2fTypes::Name()
{
    return "CudaVec2f";
}

typedef CudaVectorTypes<Vec3f1,Vec3f1,float> CudaVec3f1Types;

template<>
inline const char* CudaVec3f1Types::Name()
{
    return "CudaVec3f1";
}

template<int N, typename real>
class CudaRigidTypes;

template<typename real>
class CudaRigidTypes<3, real>
{
public:
    typedef real Real;
    typedef sofa::defaulttype::RigidCoord<3,real> Coord;
    typedef sofa::defaulttype::RigidDeriv<3,real> Deriv;
    typedef typename Coord::Vec3 Vec3;
    typedef typename Coord::Quat Quat;
    typedef CudaVector<Coord> VecCoord;
    typedef CudaVector<Deriv> VecDeriv;
    typedef CudaVector<Real> VecReal;
    typedef defaulttype::MapMapSparseMatrix<Deriv> MatrixDeriv;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef typename Deriv::Pos DPos;
    typedef typename Deriv::Rot DRot;
    static const DPos& getDPos(const Deriv& d) { return d.getVCenter(); }
    static void setDPos(Deriv& d, const DPos& v) { d.getVCenter() = v; }
    static const DRot& getDRot(const Deriv& d) { return d.getVOrientation(); }
    static void setDRot(Deriv& d, const DRot& v) { d.getVOrientation() = v; }

    template<typename T>
    static void set(Coord& r, T x, T y, T z)
    {
        Vec3& c = r.getCenter();
        if ( c.size() >0 )
            c[0] = (Real) x;
        if ( c.size() >1 )
            c[1] = (Real) y;
        if ( c.size() >2 )
            c[2] = (Real) z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& r)
    {
        const Vec3& c = r.getCenter();
        x = ( c.size() >0 ) ? (T) c[0] : (T) 0.0;
        y = ( c.size() >1 ) ? (T) c[1] : (T) 0.0;
        z = ( c.size() >2 ) ? (T) c[2] : (T) 0.0;
    }

    template<typename T>
    static void add(Coord& r, T x, T y, T z)
    {
        Vec3& c = r.getCenter();
        if ( c.size() >0 )
            c[0] += (Real) x;
        if ( c.size() >1 )
            c[1] += (Real) y;
        if ( c.size() >2 )
            c[2] += (Real) z;
    }

    template<typename T>
    static void set(Deriv& r, T x, T y, T z)
    {
        Vec3& c = r.getVCenter();
        if ( c.size() >0 )
            c[0] = (Real) x;
        if ( c.size() >1 )
            c[1] = (Real) y;
        if ( c.size() >2 )
            c[2] = (Real) z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Deriv& r)
    {
        const Vec3& c = r.getVCenter();
        x = ( c.size() >0 ) ? (T) c[0] : (T) 0.0;
        y = ( c.size() >1 ) ? (T) c[1] : (T) 0.0;
        z = ( c.size() >2 ) ? (T) c[2] : (T) 0.0;
    }

    template<typename T>
    static void add(Deriv& r, T x, T y, T z)
    {
        Vec3& c = r.getVCenter();
        if ( c.size() >0 )
            c[0] += (Real) x;
        if ( c.size() >1 )
            c[1] += (Real) y;
        if ( c.size() >2 )
            c[2] += (Real) z;
    }

    static Coord interpolate(const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            // Position interpolation.
            c.getCenter() += ancestors[i].getCenter() * coefs[i];

            // Angle extraction from the orientation quaternion.
            helper::Quater<Real> q = ancestors[i].getOrientation();
            Real angle = acos(q[3]) * 2;

            // Axis extraction from the orientation quaternion.
            defaulttype::Vec<3,Real> v(q[0], q[1], q[2]);
            Real norm = v.norm();
            if (norm > 0.0005)
            {
                v.normalize();

                // The scale factor is applied to the angle
                angle *= coefs[i];

                // Corresponding quaternion is computed, then added to the interpolated point orientation.
                q.axisToQuat(v, angle);
                q.normalize();

                c.getOrientation() += q;
            }
        }

        c.getOrientation().normalize();

        return c;
    }

    static Deriv interpolate(const helper::vector< Deriv > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    static const char* Name();
};

typedef CudaRigidTypes<3,float> CudaRigid3fTypes;
typedef CudaRigid3fTypes CudaRigid3Types;

template<>
inline const char* CudaRigid3fTypes::Name()
{
    return "CudaRigid3f";
}


// support for double precision
//#define SOFA_GPU_CUDA_DOUBLE

#ifdef SOFA_GPU_CUDA_DOUBLE
using sofa::defaulttype::Vec3d;
using sofa::defaulttype::Vec2d;
typedef Vec3r1<double> Vec3d1;

typedef CudaVectorTypes<Vec3d,Vec3d,double> CudaVec3dTypes;
//typedef CudaVec3dTypes CudaVec3Types;

template<>
inline const char* CudaVec3dTypes::Name()
{
    return "CudaVec3d";
}

typedef CudaVectorTypes<Vec2d,Vec2d,double> CudaVec2dTypes;
//typedef CudaVec2dTypes CudaVec2Types;

template<>
inline const char* CudaVec2dTypes::Name()
{
    return "CudaVec2d";
}

typedef CudaVectorTypes<Vec3d1,Vec3d1,double> CudaVec3d1Types;

template<>
inline const char* CudaVec3d1Types::Name()
{
    return "CudaVec3d1";
}

typedef CudaRigidTypes<3,double> CudaRigid3dTypes;
//typedef CudaRigid3dTypes CudaRigid3Types;

template<>
inline const char* CudaRigid3dTypes::Name()
{
    return "CudaRigid3d";
}
#endif



template<class real, class real2>
inline real operator*(const sofa::defaulttype::Vec<3,real>& v1, const sofa::gpu::cuda::Vec3r1<real2>& v2)
{
    real r = (real)(v1[0]*v2[0]);
    for (int i=1; i<3; i++)
        r += (real)(v1[i]*v2[i]);
    return r;
}

template<class real, class real2>
inline real operator*(const sofa::gpu::cuda::Vec3r1<real>& v1, const sofa::defaulttype::Vec<3,real2>& v2)
{
    real r = (real)(v1[0]*v2[0]);
    for (int i=1; i<3; i++)
        r += (real)(v1[i]*v2[i]);
    return r;
}

} // namespace cuda

} // namespace gpu

// Overload helper::ReadAccessor and helper::WriteAccessor on CudaVector

namespace helper
{

template<class T>
class ReadAccessor< gpu::cuda::CudaVector<T> >
{
public:
    typedef gpu::cuda::CudaVector<T> container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    const container_type& vref;
    const value_type* data;
public:
    ReadAccessor(const container_type& container) : vref(container), data(container.hostRead()) {}
    ~ReadAccessor() {}

    size_type size() const { return vref.size(); }
    bool empty() const { return vref.empty(); }

    const container_type& ref() const { return vref; }

    const_reference operator[](size_type i) const { return data[i]; }

    const_iterator begin() const { return data; }
    const_iterator end() const { return data+vref.size(); }

    inline friend std::ostream& operator<< ( std::ostream& os, const ReadAccessor<container_type>& vec )
    {
        return os << vec.vref;
    }
};

template<class T>
class WriteAccessor< gpu::cuda::CudaVector<T> >
{
public:
    typedef gpu::cuda::CudaVector<T> container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    container_type& vref;
    T* data;

public:
    WriteAccessor(container_type& container) : vref(container), data(container.hostWrite()) {}
    ~WriteAccessor() {}

    size_type size() const { return vref.size(); }
    bool empty() const { return vref.empty(); }

    const_reference operator[](size_type i) const { return data[i]; }
    reference operator[](size_type i) { return data[i]; }

    const container_type& ref() const { return vref; }
    container_type& wref() { return vref; }

    const_iterator begin() const { return data; }
    iterator begin() { return data; }
    const_iterator end() const { return data+vref.size(); }
    iterator end() { return data+vref.size(); }

    void clear() { vref.clear(); }
    void resize(size_type s, bool init = true) { if (init) vref.resize(s); else vref.fastResize(s); }
    void reserve(size_type s) { vref.reserve(s); }
    void push_back(const_reference v) { vref.push_back(v); }

    inline friend std::ostream& operator<< ( std::ostream& os, const WriteAccessor<container_type>& vec )
    {
        return os << vec.vref;
    }

    inline friend std::istream& operator>> ( std::istream& in, WriteAccessor<container_type>& vec )
    {
        return in >> vec.vref;
    }

};


}

} // namespace sofa

// Specialization of the defaulttype::DataTypeInfo type traits template

namespace sofa
{

namespace defaulttype
{

template<class T>
struct DataTypeInfo< sofa::gpu::cuda::CudaVector<T> > : public VectorTypeInfo<sofa::gpu::cuda::CudaVector<T> >
{
    static std::string name() { std::ostringstream o; o << "CudaVector<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

template<typename real>
struct DataTypeInfo< sofa::gpu::cuda::Vec3r1<real> > : public FixedArrayTypeInfo<sofa::gpu::cuda::Vec3r1<real> >
{
    static std::string name() { std::ostringstream o; o << "Vec3r1<" << DataTypeName<real>::name() << ">"; return o.str(); }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName<sofa::gpu::cuda::Vec3f1> { static const char* name() { return "Vec3f1"; } };
#ifdef SOFA_GPU_CUDA_DOUBLE
template<> struct DataTypeName<sofa::gpu::cuda::Vec3d1> { static const char* name() { return "Vec3d1"; } };
#endif

/// \endcond

} // namespace defaulttype

} // namespace sofa

#endif
