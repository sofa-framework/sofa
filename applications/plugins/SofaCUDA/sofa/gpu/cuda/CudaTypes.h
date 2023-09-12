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
#ifndef SOFA_GPU_CUDA_CUDATYPES_H
#define SOFA_GPU_CUDA_CUDATYPES_H

#include <sofa/gpu/cuda/CudaCommon.h>
#include "mycuda.h"
#include <sofa/core/objectmodel/Base.h>
#include <sofa/gl/gl.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>
#include <sofa/type/vector_device.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraint.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/accessor.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/gpu/cuda/CudaMemoryManager.h>
#include <sofa/component/mass/MassType.h>
#include <iostream>

namespace sofa
{

namespace gpu
{

namespace cuda
{

// Empty class to be used to highlight deprecated objects in SofaCUDA plugin at compilation time.
class CudaDeprecatedAndRemoved {};

#define SOFA_CUDA_ATTRIBUTE_DEPRECATED(removeDate, toFixMsg) \
    [[deprecated( \
        "Has been DEPRECATED and removed since " removeDate ". " \
        " To fix your code use " toFixMsg)]]

template<typename T>
struct DataTypeInfoManager
{
    static const bool ZeroConstructor = sofa::defaulttype::DataTypeInfo<T>::ZeroConstructor;
    static const bool SimpleCopy = sofa::defaulttype::DataTypeInfo<T>::SimpleCopy;
};

template<class T>
class CudaVector : public type::vector_device<T,CudaMemoryManager<T>, DataTypeInfoManager<T> >
{
public :
    using Inherit = type::vector_device<T, CudaMemoryManager<T>, DataTypeInfoManager<T> >;
    typedef size_t Size;

    CudaVector() : Inherit() {}

    CudaVector(Size n) : Inherit(n) {}

    CudaVector(const Inherit& v) : Inherit(v) {}

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
    typedef linearalgebra::CompressedRowSparseMatrixConstraint<Deriv> MatrixDeriv;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;
    static constexpr sofa::Size deriv_total_size = Deriv::total_size;

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
    static C interpolate(const type::vector< C > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        C coord;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            coord += ancestors[i] * coefs[i];
        }

        return coord;
    }

    static constexpr const char* Name();
};

typedef sofa::type::Vec3f Vec3f;
typedef sofa::type::Vec1f Vec1f;
typedef sofa::type::Vec2f Vec2f;
typedef sofa::type::Vec6f Vec6f;

using type::Vec;
using type::NoInit;
using type::NOINIT;

template<class Real>
class Vec3r1 : public sofa::type::Vec<3,Real>
{
public:
    typedef sofa::type::Vec<3,Real> Inherit;
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

    explicit operator sofa::type::Vec<3, Real>()
    {
        return { (*this)[0], (*this)[1] , (*this)[2] };
    }

protected:
    Real dummy;
};

typedef Vec3r1<float> Vec3f1;

typedef CudaVectorTypes<Vec3f,Vec3f,float> CudaVec3fTypes;
typedef CudaVec3fTypes CudaVec3Types;

template<>
constexpr const char* CudaVec3fTypes::Name()
{
    return "CudaVec3f";
}


typedef CudaVectorTypes<Vec1f,Vec1f,float> CudaVec1fTypes;
typedef CudaVec1fTypes CudaVec1Types;

template<>
constexpr const char* CudaVec1fTypes::Name()
{
    return "CudaVec1f";
}

typedef CudaVectorTypes<Vec2f,Vec2f,float> CudaVec2fTypes;
typedef CudaVec2fTypes CudaVec2Types;

template<>
constexpr const char* CudaVec2fTypes::Name()
{
    return "CudaVec2f";
}

typedef CudaVectorTypes<Vec3f1,Vec3f1,float> CudaVec3f1Types;

template<>
constexpr const char* CudaVec3f1Types::Name()
{
    return "CudaVec3f1";
}

typedef CudaVectorTypes<Vec6f,Vec6f,float> CudaVec6fTypes;
typedef CudaVec6fTypes CudaVec6Types;

template<>
constexpr const char* CudaVec6fTypes::Name()
{
    return "CudaVec6f";
}

//=============================================================================
// 3D Rigids
//=============================================================================

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
    typedef linearalgebra::CompressedRowSparseMatrixConstraint<Deriv> MatrixDeriv;
    typedef Vec3 AngularVector;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;
    static constexpr sofa::Size deriv_total_size = Deriv::total_size;

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

    static Coord interpolate(const type::vector< Coord > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            // Position interpolation.
            c.getCenter() += ancestors[i].getCenter() * coefs[i];

            // Angle extraction from the orientation quaternion.
            type::Quat<Real> q = ancestors[i].getOrientation();
            Real angle = acos(q[3]) * 2;

            // Axis extraction from the orientation quaternion.
            type::Vec<3,Real> v(q[0], q[1], q[2]);
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

    static Deriv interpolate(const type::vector< Deriv > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    static constexpr const char* Name();

    /// double cross product: a * ( b * c )
    static Vec3 crosscross ( const Vec3& a, const Vec3& b, const Vec3& c)
    {
        return cross( a, cross( b,c ));
    }
};

typedef CudaRigidTypes<3,float> CudaRigid3fTypes;
typedef CudaRigid3fTypes CudaRigid3Types;

template<>
constexpr const char* CudaRigid3fTypes::Name()
{
    return "CudaRigid3f";
}

//=============================================================================
// 2D Rigids
//=============================================================================

template<typename real>
class CudaRigidTypes<2, real>
{
public:
    typedef real Real;
    typedef typename sofa::defaulttype::StdRigidTypes<2,Real>::Vec2 Vec2;

    typedef sofa::defaulttype::RigidDeriv<2,real> Deriv;
    typedef sofa::defaulttype::RigidCoord<2,real> Coord;
    typedef Real AngularVector;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef typename sofa::defaulttype::StdRigidTypes<2,Real>::DPos DPos;
    typedef real DRot;
    static const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    static const char* Name();

    typedef CudaVector<Coord> VecCoord;
    typedef CudaVector<Deriv> VecDeriv;
    typedef CudaVector<Real> VecReal;

    typedef linearalgebra::CompressedRowSparseMatrixConstraint<Deriv> MatrixDeriv;

    template<typename T>
    static void set(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] = (Real)x;
        c.getCenter()[1] = (Real)y;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& c)
    {
        x = (T)c.getCenter()[0];
        y = (T)c.getCenter()[1];
        z = (T)0;
    }

    template<typename T>
    static void add(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] += (Real)x;
        c.getCenter()[1] += (Real)y;
    }

    template<typename T>
    static void set(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] = (Real)x;
        c.getVCenter()[1] = (Real)y;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = (T)c.getVCenter()[0];
        y = (T)c.getVCenter()[1];
        z = (T)0;
    }

    // Set linear and angular velocities, in 6D for uniformity with 3D
    template<typename T>
    static void set(Deriv& c, T x, T y, T, T vrot, T, T )
    {
        c.getVCenter()[0] = (Real)x;
        c.getVCenter()[1] = (Real)y;
        c.getVOrientation() = (Real) vrot;
    }

    template<typename T>
    static void add(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] += (Real)x;
        c.getVCenter()[1] += (Real)y;
    }

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude )
    {
        Deriv result;
        set( result, Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)),
                     Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)));
        return result;
    }

    static Coord interpolate(const type::vector< Coord > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            c += ancestors[i] * coefs[i];
        }

        return c;
    }

    static Deriv interpolate(const type::vector< Deriv > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    /// specialized version of the double cross product: a * ( b * c ) for the variation of torque applied to the frame due to a small rotation with constant force.
    static Real crosscross ( const Vec2& f, const Real& dtheta, const Vec2& OP)
    {
        return dtheta * dot( f,OP );
    }

    /// specialized version of the double cross product: a * ( b * c ) for point acceleration
    static Vec2 crosscross ( const Real& omega, const Real& dtheta, const Vec2& OP)
    {
        return OP * omega * (-dtheta);
    }

    /// create a rotation from Euler angles (only the first is used). For homogeneity with 3D.
    static CRot rotationEuler( Real x, Real , Real ){ return CRot(x); }

};

typedef CudaRigidTypes<2,float> CudaRigid2fTypes;
typedef CudaRigid2fTypes CudaRigid2Types;

template<>
inline const char* CudaRigid2fTypes::Name()
{
    return "CudaRigid2f";
}


// support for double precision
//#define SOFA_GPU_CUDA_DOUBLE

#ifdef SOFA_GPU_CUDA_DOUBLE
using sofa::type::Vec3d;
using sofa::type::Vec1d;
using sofa::type::Vec2d;
using sofa::type::Vec6d;
typedef Vec3r1<double> Vec3d1;

typedef CudaVectorTypes<Vec3d,Vec3d,double> CudaVec3dTypes;
//typedef CudaVec3dTypes CudaVec3Types;

template<>
constexpr const char* CudaVec3dTypes::Name()
{
    return "CudaVec3d";
}

typedef CudaVectorTypes<Vec1d,Vec1d,double> CudaVec1dTypes;
//typedef CudaVec1dTypes CudaVec1Types;

template<>
constexpr const char* CudaVec1dTypes::Name()
{
    return "CudaVec1d";
}

typedef CudaVectorTypes<Vec2d,Vec2d,double> CudaVec2dTypes;
//typedef CudaVec2dTypes CudaVec2Types;

template<>
constexpr const char* CudaVec2dTypes::Name()
{
    return "CudaVec2d";
}

typedef CudaVectorTypes<Vec3d1,Vec3d1,double> CudaVec3d1Types;

template<>
constexpr const char* CudaVec3d1Types::Name()
{
    return "CudaVec3d1";
}

typedef CudaVectorTypes<Vec6d,Vec6d,double> CudaVec6dTypes;
// typedef CudaVec6dTypes CudaVec6Types;

template<>
constexpr const char* CudaVec6dTypes::Name()
{
    return "CudaVec6d";
}


typedef CudaRigidTypes<3,double> CudaRigid3dTypes;
//typedef CudaRigid3dTypes CudaRigid3Types;

template<>
constexpr const char* CudaRigid3dTypes::Name()
{
    return "CudaRigid3d";
}

typedef CudaRigidTypes<2,double> CudaRigid2dTypes;
//typedef CudaRigid2dTypes CudaRigid2Types;

template<>
inline const char* CudaRigid2dTypes::Name()
{
    return "CudaRigid2d";
}

#endif



template<class real, class real2>
inline real operator*(const sofa::type::Vec<3,real>& v1, const sofa::gpu::cuda::Vec3r1<real2>& v2)
{
    real r = (real)(v1[0]*v2[0]);
    for (int i=1; i<3; i++)
        r += (real)(v1[i]*v2[i]);
    return r;
}

template<class real, class real2>
inline real operator*(const sofa::gpu::cuda::Vec3r1<real>& v1, const sofa::type::Vec<3,real2>& v2)
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
class ReadAccessorVector< gpu::cuda::CudaVector<T>>
{
public:
    typedef gpu::cuda::CudaVector<T> container_type;
    typedef typename container_type::Size Size;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    const container_type& vref;
    const value_type* data;
public:
    ReadAccessorVector(const container_type& container) : vref(container), data(container.hostRead()) {}
    ~ReadAccessorVector() {}

    Size size() const { return vref.size(); }
    bool empty() const { return vref.empty(); }

    const container_type& ref() const { return vref; }

    const_reference operator[](Size i) const { return data[i]; }

    const_iterator begin() const { return data; }
    const_iterator end() const { return data+vref.size(); }
};

template<class T>
class WriteAccessorVector< gpu::cuda::CudaVector<T> >
{
public:
    typedef gpu::cuda::CudaVector<T> container_type;
    typedef typename container_type::Size Size;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    container_type& vref;
    T* data;

public:
    WriteAccessorVector(container_type& container) : vref(container), data(container.hostWrite()) {}
    ~WriteAccessorVector() {}

    ////// Capacity //////
    bool empty() const { return vref.empty(); }
    Size size() const { return vref.size(); }
    void reserve(Size s) { vref.reserve(s); data = vref.hostWrite(); }

    ////// Element access //////
    const_reference operator[](Size i) const { return data[i]; }
    reference operator[](Size i) { return data[i]; }

    const container_type& ref() const { return vref; }
    container_type& wref() { return vref; }

    ////// Iterators //////
    const_iterator begin() const { return data; }
    iterator begin() { return data; }
    const_iterator end() const { return data+vref.size(); }
    iterator end() { return data+vref.size(); }

    ////// Modifiers //////
    void clear() { vref.clear(); }
    void resize(Size s, bool init = true) { 
        if (init) 
            vref.resize(s); 
        else 
            vref.fastResize(s); 
        data = vref.hostWrite(); 
    }
    
    iterator erase(iterator pos) { 
        iterator it = vref.erase(pos); 
        data = vref.hostWrite();
        return it;
    }

    void push_back(const_reference v) { vref.push_back(v); data = vref.hostWrite(); }
    void pop_back() { vref.pop_back(); data = vref.hostWrite(); }
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

// define MassType for CudaTypes
namespace sofa::component::mass
{
    template<class TCoord, class TDeriv, class TReal>
    struct MassType<sofa::gpu::cuda::CudaVectorTypes< TCoord, TDeriv, TReal> >
    {
        using type = TReal;
    };

    template<int N, typename real>
    struct MassType<sofa::gpu::cuda::CudaRigidTypes<N, real> >
    {
        using type = sofa::defaulttype::RigidMass<N, real>;
    };

} // namespace sofa::component::mass


// define block traits for Vec3r1 (see matrix_bloc_traits.h)
// mainly used with CompressedRowSparseMatrix
namespace sofa::linearalgebra
{

template <class T, typename IndexType >
class matrix_bloc_traits < sofa::gpu::cuda::Vec3r1<T>, IndexType > : public matrix_bloc_traits<sofa::type::Vec<3, T>, IndexType>
{
public:
    typedef sofa::gpu::cuda::Vec3r1<T> Block;

};

} // namespace sofa::linearalgebra

#endif
