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
#ifndef FRAME_DUALQUATTYPES_H
#define FRAME_DUALQUATTYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <iostream>

namespace sofa
{

namespace defaulttype
{

using std::endl;
using sofa::helper::vector;

template<int N, typename real>
class DualQuatCoord;

template<typename real>
class DualQuatCoord<3,real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<3,Real> Pos;
    typedef helper::Quater<Real> Rot;
    typedef Vec<3,Real> Vec3;
    typedef helper::Quater<Real> Quat;

protected:
    Quat center;
    Quat orientation;
public:
    DualQuatCoord (const Quat &qCenter, const Quat &orient)
        : center(qCenter), orientation(orient) {}

    template<typename real2>
    DualQuatCoord(const DualQuatCoord<3,real2>& c)
        : center(c.getCenter()), orientation(c.getOrientation()) {}

    template<typename real2>
    DualQuatCoord(const RigidCoord<3,real2>& c)
        : orientation(c.getOrientation())
    {
        setTranslation(c.getCenter());
    }


    DualQuatCoord () { clear(); }
    void clear() { center.clear(); center[3]=(Real)0.; orientation.clear(); }



    template<typename real2>
    void operator =(const DualQuatCoord<3,real2>& c)
    {
        center = c.getCenter();
        orientation = c.getOrientation();
    }

    template<typename real2>
    void operator =(const RigidCoord<3,real2>& c)
    {
        orientation = c.getOrientation();
        setTranslation(c.getCenter());
    }

    void operator =(const Vec3& p)
    {
        setTranslation(p);
    }

    void setTranslation(const Vec3& p)
    {
        center[0] =  (real)0.5* ( p[0]*orientation[3] + p[1]*orientation[2] - p[2]*orientation[1] );
        center[1] =  (real)0.5* (-p[0]*orientation[2] + p[1]*orientation[3] + p[2]*orientation[0] );
        center[2] =  (real)0.5* ( p[0]*orientation[1] - p[1]*orientation[0] + p[2]*orientation[3] );
        center[3] = -(real)0.5* ( p[0]*orientation[0] + p[1]*orientation[1] + p[2]*orientation[2] );
    }

    Vec3 getTranslation()
    {
        Vec3 t;
        t[0] =  (real)2. * ( -center[3]*orientation[0] + center[0]*orientation[3] - center[1]*orientation[2] + center[2]*orientation[1] );
        t[1] =  (real)2. * ( -center[3]*orientation[1] + center[0]*orientation[2] + center[1]*orientation[3] - center[2]*orientation[0] );
        t[2] =  (real)2. * ( -center[3]*orientation[2] - center[0]*orientation[1] + center[1]*orientation[0] + center[2]*orientation[3] );
        return t;
    }

    void operator +=(const Vec<6,real>& a)
    {
        Vec3 p=getTranslation()+getVCenter(a);
        Quat qDot = orientation.vectQuatMult(getVOrientation(a));
        for (int i = 0; i < 4; i++)
            orientation[i] += qDot[i] * 0.5f;
        orientation.normalize();
        setTranslation(p);
    }

    DualQuatCoord<3,real> operator + (const Vec<6,real>& a) const
    {
        DualQuatCoord c = *this;
        Vec3 p=c.getTranslation()+getVCenter(a);
        Quat qDot = c.orientation.vectQuatMult(getVOrientation(a));
        for (int i = 0; i < 4; i++)
            c.orientation[i] += qDot[i] * 0.5f;
        c.orientation.normalize();
        c.setTranslation(p);
        return c;
    }

    //DualQuatCoord<3,real> operator -(const DualQuatCoord<3,real>& a) const
    //{
    //    return DualQuatCoord<3,real>(this->center - a.getCenter(), a.orientation.inverse() * this->orientation);
    //}

    //DualQuatCoord<3,real> operator *(const DualQuatCoord<3,real>& a) const
    //{
    //    return DualQuatCoord<3,real>(this->center + a.getCenter(), a.orientation * this->orientation);
    //}

    void operator +=(const DualQuatCoord<3,real>& a)
    {
        center += a.getCenter();
        orientation *= a.getOrientation();
    }

    template<typename real2>
    void operator*=(real2 a)
    {
        center *= a;
    }

    template<typename real2>
    void operator/=(real2 a)
    {
        center /= a;
    }

    template<typename real2>
    DualQuatCoord<3,real> operator*(real2 a) const
    {
        DualQuatCoord r = *this;
        r*=a;
        return r;
    }





    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const DualQuatCoord<3,real>& a) const
    {
        return center[0]*a.center[0]+center[1]*a.center[1]+center[2]*a.center[2]+center[3]*a.center[3]
                +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
    }

    /// Squared norm
    real norm2() const
    {
        real r = (this->center).elems[0]*(this->center).elems[0];
        for (int i=1; i<3; i++)
            r += (this->center).elems[i]*(this->center).elems[i];
        return r;
    }

    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt(norm2());
    }


    void normalize()
    {
        real mag = (real) sqrt ( orientation[0]*orientation[0] + orientation[1]*orientation[1] + orientation[2]*orientation[2] + orientation[3]*orientation[3] );
        if( mag != 0)
        {
            for (unsigned int j = 0; j < 4; j++ )
            {
                orientation[j] /= mag;
                center[j] /= mag;
            }
            Real dp = (real) ( orientation[0]*center[0] + orientation[1]*center[1] + orientation[2]*center[2] + orientation[3]*center[3] );
            for ( unsigned int j = 0; j < 4; j++ )
                center[j] -= dp*orientation[j];
        }
    }



    Quat& getCenter () { return center; }
    Quat& getOrientation () { return orientation; }
    const Quat& getCenter () const { return center; }
    const Quat& getOrientation () const { return orientation; }

    static DualQuatCoord<3,real> identity()
    {
        DualQuatCoord c;
        return c;
    }

    Vec3 rotate(const Vec3& v) const
    {
        return orientation.rotate(v);
    }
    Vec3 inverseRotate(const Vec3& v) const
    {
        return orientation.inverseRotate(v);
    }

    void invert()
    {
        for ( unsigned int j = 0; j < 3; j++ )
        {
            orientation[j]=-orientation[j];
            center[j]=-center[j];
        }
    }

    DualQuatCoord<3,real> inverse( )
    {
        DualQuatCoord r;
        for ( unsigned int j = 0; j < 3; j++ )
        {
            r.orientation[j]=-orientation[j];
            r.center[j]=-center[j];
        }
        r.orientation[3]=orientation[3];
        r.center[3]=center[3];
        return r;
    }

    /// Apply a transformation with respect to itself
    void multRight( const DualQuatCoord<3,real>& c )
    {
        center = orientation * c.getCenter() + center * c.getOrientation();
        orientation = orientation * c.getOrientation();
    }

    /// compute the product with another frame on the right
    DualQuatCoord<3,real> mult( const DualQuatCoord<3,real>& c ) const
    {
        DualQuatCoord r;
        r.center = orientation * c.getCenter() + center * c.getOrientation();
        r.orientation = orientation * c.getOrientation();
        return r;
    }

    /// Set from the given matrix
    template<class Mat>
    void fromMatrix(const Mat& m)
    {
        Mat3x3d rot; rot = m;
        orientation.fromMatrix(rot);
        Vec3 p;
        p[0] = m[0][3];
        p[1] = m[1][3];
        p[2] = m[2][3];
        *this=p;
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix( Mat& m) const
    {
        m.identity();
        orientation.toMatrix(m);
        Vec3 p=getTranslation();
        m[0][3] =  (real)p[0];
        m[1][3] =  (real)p[1];
        m[2][3] =  (real)p[2];
    }

    template<class Mat>
    void writeRotationMatrix( Mat& m) const
    {
        orientation.toMatrix(m);
    }

    /// Write the OpenGL transformation matrix
    void writeOpenGlMatrix( float m[16] ) const
    {
        orientation.writeOpenGlMatrix(m);
        Vec3 p=getTranslation();
        m[12] =  (float)p[0];
        m[13] =  (float)p[1];
        m[14] =  (float)p[2];
    }

    /// Project a point from the child frame to the parent frame
    Vec3 pointToParent( const Vec3& v ) const
    {
        return orientation.rotate(v)+getTranslation();
    }

    /// Project a point from the parent frame to the child frame
    Vec3 pointToChild( const Vec3& v ) const
    {
        return orientation.inverseRotate(v-getTranslation());
    }

    /// compute the projection of a vector from the parent frame to the child
    Vec3 vectorToChild( const Vec3& v ) const
    {
        return orientation.inverseRotate(v);
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const DualQuatCoord<3,real>& v )
    {
        out<<v.center<<" "<<v.orientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, DualQuatCoord<3,real>& v )
    {
        in>>v.center>>v.orientation;
        return in;
    }
    static int max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 8 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for DualQuats)
    enum { spatial_dimensions = 3 };

    real* ptr() { return center.ptr(); }
    const real* ptr() const { return center.ptr(); }

    static unsigned int size() {return 8;}

    /// Access to i-th element.
    real& operator[](int i)
    {
        if (i<4)
            return this->center(i);
        else
            return this->orientation[i-4];
    }

    /// Const access to i-th element.
    const real& operator[](int i) const
    {
        if (i<4)
            return this->center(i);
        else
            return this->orientation[i-4];
    }
};




} // namespace defaulttype


} // namespace sofa


#endif
