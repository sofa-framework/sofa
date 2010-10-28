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
#ifndef SOFA_DEFAULTTYPE_VECTYPES_H
#define SOFA_DEFAULTTYPE_VECTYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <iostream>
#include <algorithm>



namespace sofa
{

namespace defaulttype
{

using helper::vector;

template<class TCoord, class TDeriv, class TReal = typename TCoord::value_type>
class StdVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef TReal Real;
    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;
    typedef vector<Real> VecReal;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef Coord CPos;
    static const CPos& getCPos(const Coord& c) { return c; }
    static void setCPos(Coord& c, const CPos& v) { c = v; }
    typedef Deriv DPos;
    static const DPos& getDPos(const Deriv& d) { return d; }
    static void setDPos(Deriv& d, const DPos& v) { d = v; }

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    template<typename T>
    static void set(Coord& c, T x, T y, T z)
    {
        if (c.size() > 0)
            c[0] = (Real)x;
        if (c.size() > 1)
            c[1] = (Real)y;
        if (c.size() > 2)
            c[2] = (Real)z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& c)
    {
        x = (c.size() > 0) ? (T) c[0] : (T) 0.0;
        y = (c.size() > 1) ? (T) c[1] : (T) 0.0;
        z = (c.size() > 2) ? (T) c[2] : (T) 0.0;
    }

    template<typename T>
    static void add(Coord& c, T x, T y, T z)
    {
        if (c.size() > 0)
            c[0] += (Real)x;
        if (c.size() > 1)
            c[1] += (Real)y;
        if (c.size() > 2)
            c[2] += (Real)z;
    }

    static const char* Name();

    static Coord interpolate(const helper::vector< Coord > &ancestors, const helper::vector< Real > &coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            c += ancestors[i] * coefs[i];
        }

        return c;
    }
};

/// Custom vector allocator class allowing data to be allocated at a specific location (such as for transmission through DMA, PCI-Express, Shared Memory, Network)
template<class T>
class ExtVectorAllocator
{
public:
    typedef T              value_type;
    typedef unsigned int   size_type;
    virtual ~ExtVectorAllocator() {}
    virtual void resize(value_type*& data, size_type size, size_type& maxsize, size_type& cursize)=0;
    virtual void close(value_type* data)=0;
};

/// Custom vector class.
///
/// This class allows custom buffer allocation while not having any virtual methods using a bridge pattern with ExtVectorAllocator
template<class T>
class ExtVector
{
public:
    typedef T				value_type;
    typedef unsigned int	size_type;
    typedef T&				reference;
    typedef const T&		const_reference;
    typedef T*				iterator;
    typedef const T*		const_iterator;

protected:
    value_type* data;
    size_type   maxsize;
    size_type   cursize;
    ExtVectorAllocator<T>* allocator;

public:
    explicit ExtVector(ExtVectorAllocator<T>* alloc = NULL) : data(NULL),  maxsize(0), cursize(0), allocator(alloc) {}
    ExtVector(int size, ExtVectorAllocator<T>* alloc) : data(NULL), maxsize(0), cursize(0), allocator(alloc) { resize(size); }
    ~ExtVector() { if (allocator) allocator->close(data); }

    void init() {}

    void setAllocator(ExtVectorAllocator<T>* alloc)
    {
        if (alloc != allocator)
        {
            if (cursize)
            {
                value_type* oldData = data;
                size_type size = cursize;
                data = NULL;
                maxsize = 0;
                cursize = 0;
                alloc->resize(data, size, maxsize, cursize);
                std::copy(oldData, oldData+size, data);
                if (allocator)
                    allocator->close(oldData);
            }
            allocator = alloc;
        }
    }
    void setData(value_type* d, size_type s) { data=d; maxsize=s; cursize=s; }
    T* getData() { return this->data; }
    const T* getData() const { return this->data; }

    value_type& operator[](size_type i) { return data[i]; }
    const value_type& operator[](size_type i) const { return data[i]; }
    size_type size() const { return cursize; }
    bool empty() const { return cursize==0; }
    void reserve(size_type size)
    {
        if (size <= maxsize)
            return;
        size_type temp = cursize;
        if (allocator)
            allocator->resize(data, size, maxsize, temp);
        else
        {
            std::cerr << "Error: invalid reserve request ("<<size<<">"<<maxsize<<") on external vector without allocator.\n";
        }
    }
    void resize(size_type size)
    {
        if (size <= maxsize)
            cursize = size;
        else if (allocator)
            allocator->resize(data, size, maxsize, cursize);
        else
        {
            cursize = maxsize;
            std::cerr << "Error: invalid resize request ("<<size<<">"<<maxsize<<") on external vector without allocator.\n";
        }
    }
    void clear()
    {
        resize(0);
    }
    void push_back(const T& v)
    {
        int i = this->size();
        resize(i+1);
        (*this)[i] = v;
    }
    T* begin() { return getData(); }
    const T* begin() const { return getData(); }
    T* end() { return getData()+size(); }
    const T* end() const { return getData()+size(); }

    ExtVector& operator=(const ExtVector& ev)
    {
        cursize = ev.size();
        maxsize = 0;
        while(maxsize < cursize)
            maxsize *= 2;
        resize(cursize);
        T* oldData = data;
        data = new T[maxsize];
        if (cursize)
            std::copy(ev.begin(), ev.end(), data);
        if (oldData!=NULL) delete[] oldData;

        return *this;
    }

    ExtVector(const ExtVector& ev)
    {
        cursize = ev.size();
        maxsize = 0;
        while(maxsize < cursize)
            maxsize *= 2;
        data = new T[maxsize];
        if (cursize)
            std::copy(ev.begin(), ev.end(), data);
        //Alloc
    }


/// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const ExtVector<T>& vec )
    {
        if( vec.size()>0 )
        {
            for( unsigned int i=0; i<vec.size()-1; ++i ) os<<vec[i]<<" ";
            os<<vec[vec.size()-1];
        }
        return os;
    }

/// Input stream
    inline friend std::istream& operator>> ( std::istream& in, ExtVector<T>& vec )
    {
        T t;
        vec.clear();
        while(in>>t)
        {
            vec.push_back(t);
        }
        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

};

template<class T>
class DefaultAllocator : public ExtVectorAllocator<T>
{
public:
    typedef typename ExtVectorAllocator<T>::value_type value_type;
    typedef typename ExtVectorAllocator<T>::size_type size_type;
    virtual void close(value_type* data)
    {
        if (data!=NULL) delete[] data;
        delete this;
    }
    virtual void resize(value_type*& data, size_type size, size_type& maxsize, size_type& cursize)
    {
        if (size > maxsize)
        {
            T* oldData = data;
            maxsize = (size > 2*maxsize ? size : 2*maxsize);
            data = new T[maxsize];
            if (cursize)
                std::copy(oldData, oldData+cursize, data);
            //for (size_type i = 0 ; i < cursize ; ++i)
            //    data[i] = oldData[i];
            if (oldData!=NULL) delete[] oldData;
        }
        cursize = size;
    }
};

/// Resizable custom vector class using DefaultAllocator
template<class T>
class ResizableExtVector : public ExtVector<T>
{
public:
    typedef typename ExtVector<T>::value_type value_type;
    typedef typename ExtVector<T>::size_type size_type;
    ResizableExtVector()
        : ExtVector<T>(new DefaultAllocator<T>)
    {
    }

    ResizableExtVector(const ResizableExtVector& ev)
        :ExtVector<T>(ev)
    {
        this->allocator = new DefaultAllocator<T>;
    }
};

template<class TCoord, class TDeriv, class TReal = typename TCoord::value_type>
class ExtVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef TReal Real;
    typedef ResizableExtVector<Coord> VecCoord;
    typedef ResizableExtVector<Deriv> VecDeriv;
    typedef ResizableExtVector<Real> VecReal;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef Coord CPos;
    static const CPos& getCPos(const Coord& c) { return c; }
    static void setCPos(Coord& c, const CPos& v) { c = v; }
    typedef Deriv DPos;
    static const DPos& getDPos(const Deriv& d) { return d; }
    static void setDPos(Deriv& d, const DPos& v) { d = v; }

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    template<typename T>
    static void set(Coord& c, T x, T y, T z)
    {

        if (c.size() > 0)
            c[0] = (Real)x;
        if (c.size() > 1)
            c[1] = (Real)y;
        if (c.size() > 2)
            c[2] = (Real)z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& c)
    {
        x = (c.size() > 0) ? (T) c[0] : (T) 0.0;
        y = (c.size() > 1) ? (T) c[1] : (T) 0.0;
        z = (c.size() > 2) ? (T) c[2] : (T) 0.0;
    }

    template<typename T>
    static void add(Coord& c, T x, T y, T z)
    {
        if (c.size() > 0)
            c[0] += (Real)x;
        if (c.size() > 1)
            c[1] += (Real)y;
        if (c.size() > 2)
            c[2] += (Real)z;
    }

    static const char* Name();

    static Coord interpolate(const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            c += ancestors[i] * coefs[i];
        }

        return c;
    }
};

//
// 3D
//

/// 3D DOFs, double precision
typedef StdVectorTypes<Vec3d,Vec3d,double> Vec3dTypes;
/// 3D DOFs, single precision
typedef StdVectorTypes<Vec3f,Vec3f,float> Vec3fTypes;

template<> inline const char* Vec3dTypes::Name() { return "Vec3d"; }
template<> inline const char* Vec3fTypes::Name() { return "Vec3f"; }

/// 3D external DOFs, double precision
typedef ExtVectorTypes<Vec3d,Vec3d,double> ExtVec3dTypes;
/// 3D external DOFs, single precision
typedef ExtVectorTypes<Vec3f,Vec3f,float> ExtVec3fTypes;

template<> inline const char* ExtVec3dTypes::Name() { return "ExtVec3d"; }
template<> inline const char* ExtVec3fTypes::Name() { return "ExtVec3f"; }

//
// 2D
//

/// 2D DOFs, double precision
typedef StdVectorTypes<Vec2d,Vec2d,double> Vec2dTypes;
/// 2D DOFs, single precision
typedef StdVectorTypes<Vec2f,Vec2f,float> Vec2fTypes;

template<> inline const char* Vec2dTypes::Name() { return "Vec2d"; }
template<> inline const char* Vec2fTypes::Name() { return "Vec2f"; }

/// 2D external DOFs, double precision
typedef ExtVectorTypes<Vec2d,Vec2d,double> ExtVec2dTypes;
/// 2D external DOFs, single precision
typedef ExtVectorTypes<Vec2f,Vec2f,float> ExtVec2fTypes;

template<> inline const char* ExtVec2dTypes::Name() { return "ExtVec2d"; }
template<> inline const char* ExtVec2fTypes::Name() { return "ExtVec2f"; }

//
// 1D
//

/// 1D DOFs, double precision
typedef StdVectorTypes<Vec1d,Vec1d,double> Vec1dTypes;
/// 1D DOFs, single precision
typedef StdVectorTypes<Vec1f,Vec1f,float> Vec1fTypes;

template<> inline const char* Vec1dTypes::Name() { return "Vec1d"; }
template<> inline const char* Vec1fTypes::Name() { return "Vec1f"; }

/// 1D external DOFs, double precision
typedef ExtVectorTypes<Vec1d,Vec1d,double> ExtVec1dTypes;
/// 1D external DOFs, single precision
typedef ExtVectorTypes<Vec1f,Vec1f,float> ExtVec1fTypes;

template<> inline const char* ExtVec1dTypes::Name() { return "ExtVec1d"; }
template<> inline const char* ExtVec1fTypes::Name() { return "ExtVec1f"; }

//
// 6D (3 coordinates + 3 angles)
//

/// 6D DOFs, double precision
typedef StdVectorTypes<Vec6d,Vec6d,double> Vec6dTypes;
/// 6D DOFs, single precision
typedef StdVectorTypes<Vec6f,Vec6f,float> Vec6fTypes;

template<> inline const char* Vec6dTypes::Name() { return "Vec6d"; }
template<> inline const char* Vec6fTypes::Name() { return "Vec6f"; }

/// 6D external DOFs, double precision
typedef ExtVectorTypes<Vec6d,Vec6d,double> ExtVec6dTypes;
/// 6D external DOFs, single precision
typedef ExtVectorTypes<Vec6f,Vec6f,float> ExtVec6fTypes;

template<> inline const char* ExtVec6dTypes::Name() { return "ExtVec6d"; }
template<> inline const char* ExtVec6fTypes::Name() { return "ExtVec6f"; }

#ifdef SOFA_FLOAT
/// 6D DOFs, double precision (default)
typedef Vec6fTypes Vec6Types;
/// 3D DOFs, double precision (default)
typedef Vec3fTypes Vec3Types;
/// 2D DOFs, double precision (default)
typedef Vec2fTypes Vec2Types;
/// 1D DOFs, double precision (default)
typedef Vec1fTypes Vec1Types;
/// 6D external DOFs, double precision (default)
typedef ExtVec6fTypes ExtVec6Types;
/// 3D external DOFs, double precision (default)
typedef ExtVec3fTypes ExtVec3Types;
/// 2D external DOFs, double precision (default)
typedef ExtVec2fTypes ExtVec2Types;
/// 1D external DOFs, double precision (default)
typedef ExtVec1fTypes ExtVec1Types;
#else
/// 6D DOFs, double precision (default)
typedef Vec6dTypes Vec6Types;
/// 3D DOFs, double precision (default)
typedef Vec3dTypes Vec3Types;
/// 2D DOFs, double precision (default)
typedef Vec2dTypes Vec2Types;
/// 1D DOFs, double precision (default)
typedef Vec1dTypes Vec1Types;
/// 6D external DOFs, double precision (default)
typedef ExtVec6dTypes ExtVec6Types;
/// 3D external DOFs, double precision (default)
typedef ExtVec3dTypes ExtVec3Types;
/// 2D external DOFs, double precision (default)
typedef ExtVec2dTypes ExtVec2Types;
/// 1D external DOFs, double precision (default)
typedef ExtVec1dTypes ExtVec1Types;
#endif


// Specialization of the defaulttype::DataTypeInfo type traits template

template<class T>
struct DataTypeInfo< sofa::defaulttype::ExtVector<T> > : public VectorTypeInfo<sofa::defaulttype::ExtVector<T> >
{
    // Remove copy-on-write behavior which is normally activated for vectors
    enum { CopyOnWrite     = 0 };

    static std::string name() { std::ostringstream o; o << "ExtVector<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

template<class T>
struct DataTypeInfo< sofa::defaulttype::ResizableExtVector<T> > : public VectorTypeInfo<sofa::defaulttype::ResizableExtVector<T> >
{
    // Remove copy-on-write behavior which is normally activated for vectors
    enum { CopyOnWrite     = 0 };

    static std::string name() { std::ostringstream o; o << "ResizableExtVector<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

} // namespace defaulttype


namespace helper
{

template<class T>
class ReadAccessor< defaulttype::ExtVector<T> > : public ReadAccessorVector< defaulttype::ExtVector<T> >
{
public:
    typedef ReadAccessorVector< defaulttype::ExtVector<T> > Inherit;
    typedef typename Inherit::container_type container_type;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template<class T>
class WriteAccessor< defaulttype::ExtVector<T> > : public WriteAccessorVector< defaulttype::ExtVector<T> >
{
public:
    typedef WriteAccessorVector< defaulttype::ExtVector<T> > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};

template<class T>
class ReadAccessor< defaulttype::ResizableExtVector<T> > : public ReadAccessorVector< defaulttype::ResizableExtVector<T> >
{
public:
    typedef ReadAccessorVector< defaulttype::ResizableExtVector<T> > Inherit;
    typedef typename Inherit::container_type container_type;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template<class T>
class WriteAccessor< defaulttype::ResizableExtVector<T> > : public WriteAccessorVector< defaulttype::ResizableExtVector<T> >
{
public:
    typedef WriteAccessorVector< defaulttype::ResizableExtVector<T> > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};

} // namespace helper


namespace core
{
namespace behavior
{

/** Return the inertia force applied to a body referenced in a moving coordinate system.
\param sv spatial velocity (omega, vorigin) of the coordinate system
\param a acceleration of the origin of the coordinate system
\param m mass of the body
\param x position of the body in the moving coordinate system
\param v velocity of the body in the moving coordinate system
This default implementation returns no inertia.
*/
template<class Coord, class Deriv, class Vec, class M, class SV>
Deriv inertiaForce( const SV& /*sv*/, const Vec& /*a*/, const M& /*m*/, const Coord& /*x*/, const Deriv& /*v*/ );

/// Specialization of the inertia force for 3D particles
template <>
inline defaulttype::Vec<3, double> inertiaForce<
defaulttype::Vec<3, double>,
            defaulttype::Vec<3, double>,
            objectmodel::BaseContext::Vec3,
            double,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const double& m,
                    const defaulttype::Vec<3, double>& x,
                    const defaulttype::Vec<3, double>& v
            )
{
    const objectmodel::BaseContext::Vec3& omega=sv.getAngularVelocity();
    //std::cerr<<"inertiaForce, sv = "<<sv<<", omega ="<<omega<<", a = "<<a<<", m= "<<m<<", x= "<<x<<", v= "<<v<<std::endl;
    return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

/// Specialization of the inertia force for 3D particles
template <>
inline defaulttype::Vec<3, float> inertiaForce<
defaulttype::Vec<3, float>,
            defaulttype::Vec<3, float>,
            objectmodel::BaseContext::Vec3,
            float,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const float& m,
                    const defaulttype::Vec<3, float>& x,
                    const defaulttype::Vec<3, float>& v
            )
{
    const objectmodel::BaseContext::Vec3& omega=sv.getAngularVelocity();
    //std::cerr<<"inertiaForce, sv = "<<sv<<", omega ="<<omega<<", a = "<<a<<", m= "<<m<<", x= "<<x<<", v= "<<v<<std::endl;
    return -( a + omega.cross( omega.cross(x) + v*2 ))*m;
}

/// Specialization of the inertia force for 2D particles
template <>
inline defaulttype::Vec<2, double> inertiaForce<
defaulttype::Vec<2, double>,
            defaulttype::Vec<2, double>,
            objectmodel::BaseContext::Vec3,
            double,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const double& m,
                    const defaulttype::Vec<2, double>& x,
                    const defaulttype::Vec<2, double>& v
            )
{
    double omega=(double)sv.getAngularVelocity()[2]; // z direction
    defaulttype::Vec<2, double> a2( (double)a[0], (double)a[1] );
    return -( a2 -( x*omega + v*2 )*omega )*m;
}

/// Specialization of the inertia force for 2D particles
template <>
inline defaulttype::Vec<2, float> inertiaForce<
defaulttype::Vec<2, float>,
            defaulttype::Vec<2, float>,
            objectmodel::BaseContext::Vec3,
            float,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& sv,
                    const objectmodel::BaseContext::Vec3& a,
                    const float& m,
                    const defaulttype::Vec<2, float>& x,
                    const defaulttype::Vec<2, float>& v
            )
{
    float omega=(float)sv.getAngularVelocity()[2]; // z direction
    defaulttype::Vec<2, float> a2( (float)a[0], (float)a[1] );
    return -( a2 -( x*omega + v*2 )*omega )*m;
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
