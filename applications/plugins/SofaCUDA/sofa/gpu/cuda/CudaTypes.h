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
#include <sofa/helper/vector_device.h>

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


    /// @internal size dependant specializations
    /// @{

    /// default implementation for size >= 3
    template<int N, class T>
    struct Impl
    {
        static void set( Coord& c, T x, T y, T z )
        {
            c[0] = (Real)x;
            c[1] = (Real)y;
            c[2] = (Real)z;
        }

        static void get( T& x, T& y, T& z, const Coord& c )
        {
            x = (T) c[0];
            y = (T) c[1];
            z = (T) c[2];
        }

        static void add( Coord& c, T x, T y, T z )
        {
            c[0] += (Real)x;
            c[1] += (Real)y;
            c[2] += (Real)z;
        }
    };

    /// specialization for size == 2
    template<class T>
    struct Impl<2,T>
    {
        static void set( Coord& c, T x, T y, T )
        {
            c[0] = (Real)x;
            c[1] = (Real)y;
        }

        static void get( T& x, T& y, T& z, const Coord& c )
        {
            x = (T) c[0];
            y = (T) c[1];
            z = (T) 0;
        }

        static void add( Coord& c, T x, T y, T )
        {
            c[0] += (Real)x;
            c[1] += (Real)y;
        }
    };

    /// specialization for size == 1
    template<class T>
    struct Impl<1,T>
    {
        static void set( Coord& c, T x, T, T )
        {
            c[0] = (Real)x;
        }

        static void get( T& x, T& y, T& z, const Coord& c )
        {
            x = (T) c[0];
            y = (T) 0;
            z = (T) 0;
        }

        static void add( Coord& c, T x, T, T )
        {
            c[0] += (Real)x;
        }
    };

    ///@}

    template<class C, typename T>
    static void set( C& c, T x, T y, T z )
    {
        Impl<spatial_dimensions,T>::set(c,x,y,z);
    }

    template<class C, typename T>
    static void get( T& x, T& y, T& z, const C& c )
    {
        Impl<spatial_dimensions,T>::get(x,y,z,c);
    }

    template<class C, typename T>
    static void add( C& c, T x, T y, T z )
    {
        Impl<spatial_dimensions,T>::add(c,x,y,z);
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
typedef sofa::defaulttype::Vec1f Vec1f;
typedef sofa::defaulttype::Vec2f Vec2f;
typedef sofa::defaulttype::Vec6f Vec6f;

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
    Vec3r1() : dummy((Real) 0.0) {}
    template<class real2>
    Vec3r1(const Vec<N,real2>& v): Inherit(v), dummy((Real) 0.0) {}
    Vec3r1(real x, real y, real z) : Inherit(x,y,z), dummy((Real) 0.0) {}

    /// Fast constructor: no initialization
    explicit Vec3r1(NoInit n) : Inherit(n), dummy((Real) 0.0)
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
        static_assert(N == 3, "");
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


typedef CudaVectorTypes<Vec1f,Vec1f,float> CudaVec1fTypes;
typedef CudaVec1fTypes CudaVec1Types;

template<>
inline const char* CudaVec1fTypes::Name()
{
    return "CudaVec1f";
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

typedef CudaVectorTypes<Vec6f,Vec6f,float> CudaVec6fTypes;
typedef CudaVec6fTypes CudaVec6Types;

template<>
inline const char* CudaVec6fTypes::Name()
{
    return "CudaVec6f";
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
    typedef Vec3 AngularVector;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef typename sofa::defaulttype::StdRigidTypes<3,Real>::DPos DPos;
    typedef typename sofa::defaulttype::StdRigidTypes<3,Real>::DRot DRot;
    static const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    template<typename T>
    static void set(Coord& r, T x, T y, T z)
    {
        Vec3& c = r.getCenter();
        c[0] = (Real) x;
        c[1] = (Real) y;
        c[2] = (Real) z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& r)
    {
        const Vec3& c = r.getCenter();
        x = (T) c[0];
        y = (T) c[1];
        z = (T) c[2];
    }

    template<typename T>
    static void add(Coord& r, T x, T y, T z)
    {
        Vec3& c = r.getCenter();
        c[0] += (Real) x;
        c[1] += (Real) y;
        c[2] += (Real) z;
    }

    template<typename T>
    static void set(Deriv& r, T x, T y, T z)
    {
        Vec3& c = getVCenter(r);
        c[0] = (Real) x;
        c[1] = (Real) y;
        c[2] = (Real) z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Deriv& r)
    {
        const Vec3& c = getVCenter(r);
        x = (T) c[0];
        y = (T) c[1];
        z = (T) c[2];
    }

    template<typename T>
    static void add(Deriv& r, T x, T y, T z)
    {
        Vec3& c = getVCenter(r);
        c[0] += (Real) x;
        c[1] += (Real) y;
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

    /// double cross product: a * ( b * c )
    static Vec3 crosscross ( const Vec3& a, const Vec3& b, const Vec3& c)
    {
        return cross( a, cross( b,c ));
    }
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
using sofa::defaulttype::Vec1d;
using sofa::defaulttype::Vec2d;
using sofa::defaulttype::Vec6d;
typedef Vec3r1<double> Vec3d1;

typedef CudaVectorTypes<Vec3d,Vec3d,double> CudaVec3dTypes;
//typedef CudaVec3dTypes CudaVec3Types;

template<>
inline const char* CudaVec3dTypes::Name()
{
    return "CudaVec3d";
}

typedef CudaVectorTypes<Vec1d,Vec1d,double> CudaVec1dTypes;
//typedef CudaVec1dTypes CudaVec1Types;

template<>
inline const char* CudaVec1dTypes::Name()
{
    return "CudaVec1d";
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

typedef CudaVectorTypes<Vec6d,Vec6d,double> CudaVec6dTypes;
// typedef CudaVec6dTypes CudaVec6Types;

template<>
inline const char* CudaVec6dTypes::Name()
{
    return "CudaVec6d";
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
    void resize(size_type s, bool init = true) { if (init) vref.resize(s); else vref.fastResize(s); data = vref.hostWrite(); }
    void reserve(size_type s) { vref.reserve(s); data = vref.hostWrite(); }
    void push_back(const_reference v) { vref.push_back(v); data = vref.hostWrite(); }

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
