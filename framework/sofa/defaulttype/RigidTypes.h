/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_RIGIDTYPES_H
#define SOFA_DEFAULTTYPE_RIGIDTYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/helper/vector.h>
#include <iostream>
using std::endl;

namespace sofa
{

namespace defaulttype
{

using sofa::helper::vector;

/** Degrees of freedom of rigid bodies. Orientations are modeled using quaternions.
*/
class RigidTypes
{
public:
    typedef Vec3d Vec3;
    typedef Vec3::value_type Real;

    class Deriv
    {
    private:
        Vec3 vCenter;
        Vec3 vOrientation;
    public:
        friend class Coord;

        Deriv (const Vec3 &velCenter, const Vec3 &velOrient)
            : vCenter(velCenter), vOrientation(velOrient) {}
        Deriv () { clear(); }

        void clear() { vCenter.clear(); vOrientation.clear(); }

        void operator +=(const Deriv& a)
        {
            vCenter += a.vCenter;
            vOrientation += a.vOrientation;
        }

        Deriv operator + (const Deriv& a) const
        {
            Deriv d;
            d.vCenter = vCenter + a.vCenter;
            d.vOrientation = vOrientation + a.vOrientation;
            return d;
        }

        void operator*=(double a)
        {
            vCenter *= a;
            vOrientation *= a;
        }

        Deriv operator*(double a) const
        {
            Deriv r = *this;
            r*=a;
            return r;
        }

        Deriv operator - () const
        {
            return Deriv(-vCenter, -vOrientation);
        }

        /// dot product
        double operator*(const Deriv& a) const
        {
            return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]+vCenter[2]*a.vCenter[2]
                    +vOrientation[0]*a.vOrientation[0]+vOrientation[1]*a.vOrientation[1]
                    +vOrientation[2]*a.vOrientation[2];
        }

        Vec3& getVCenter (void) { return vCenter; }
        Vec3& getVOrientation (void) { return vOrientation; }
        const Vec3& getVCenter (void) const { return vCenter; }
        const Vec3& getVOrientation (void) const { return vOrientation; }
        /*		inline friend std::ostream& operator << (std::ostream& out, const Deriv& v ){
        		    out<<"vCenter = "<<v.getVCenter();
        		    out<<", vOrientation = "<<v.getVOrientation();
        		    return out;
        		}*/
        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Deriv& v )
        {
            out<<v.vCenter<<" "<<v.vOrientation;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Deriv& v )
        {
            in>>v.vCenter>>v.vOrientation;
            return in;
        }
    };

    class Coord
    {
    private:
        Vec3 center;
        Quat orientation;
    public:
        Coord (const Vec3 &posCenter, const Quat &orient)
            : center(posCenter), orientation(orient) {}
        Coord () { clear(); }

        void clear() { center.clear(); orientation.clear(); }

        void operator +=(const Deriv& a)
        {
            center += a.getVCenter();
            orientation.normalize();
            Quat qDot = orientation.vectQuatMult(a.getVOrientation());
            for (int i = 0; i < 4; i++)
                orientation[i] += qDot[i] * 0.5;
            orientation.normalize();
        }

        Coord operator + (const Deriv& a) const
        {
            Coord c = *this;
            c.center += a.getVCenter();
            c.orientation.normalize();
            Quat qDot = c.orientation.vectQuatMult(a.getVOrientation());
            for (int i = 0; i < 4; i++)
                c.orientation[i] += qDot[i] * 0.5;
            c.orientation.normalize();
            return c;
        }

        void operator +=(const Coord& a)
        {
            std::cout << "+="<<std::endl;
            center += a.getCenter();
            //orientation += a.getOrientation();
            //orientation.normalize();
        }

        void operator*=(double a)
        {
            std::cout << "*="<<std::endl;
            center *= a;
            //orientation *= a;
        }

        Coord operator*(double a) const
        {
            Coord r = *this;
            r*=a;
            return r;
        }

        /// dot product (FF: WHAT????  )
        double operator*(const Coord& a) const
        {
            return center[0]*a.center[0]+center[1]*a.center[1]+center[2]*a.center[2]
                    +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                    +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
        }

        Vec3& getCenter () { return center; }
        Quat& getOrientation () { return orientation; }
        const Vec3& getCenter () const { return center; }
        const Quat& getOrientation () const { return orientation; }
        /*                inline friend std::ostream& operator << (std::ostream& out, const Coord& c ){
                            out<<"translation = "<<c.getCenter();
                            out<<", rotation = "<<c.getOrientation();
                            return out;
                        }*/

        static Coord identity()
        {
            Coord c;
            return c;
        }

        /// Apply a transformation with respect to itself
        void multRight( const Coord& c )
        {
            center += orientation.rotate(c.getCenter());
            orientation = orientation * c.getOrientation();
        }

        /// compute the product with another frame on the right
        Coord mult( const Coord& c ) const
        {
            Coord r;
            r.center = center + orientation.rotate( c.center );
            r.orientation = orientation * c.getOrientation();
            return r;
        }

        /// Write the OpenGL transformation matrix
        void writeOpenGlMatrix( float m[16] ) const
        {
            orientation.writeOpenGlMatrix(m);
            m[12] = (float)center[0];
            m[13] = (float)center[1];
            m[14] = (float)center[2];
        }

        /// compute the projection of a vector from the parent frame to the child
        Vec3 vectorToChild( const Vec3& v ) const
        {
            return orientation.inverseRotate(v);
        }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Coord& v )
        {
            out<<v.center<<" "<<v.orientation;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Coord& v )
        {
            in>>v.center>>v.orientation;
            return in;
        }
    };

    template <class T>
    class SparseData
    {
    public:
        SparseData(unsigned int _index, T& _data): index(_index), data(_data) {};
        unsigned int index;
        T data;
    };

    typedef SparseData<Coord> SparseCoord;
    typedef SparseData<Deriv> SparseDeriv;

    typedef vector<SparseCoord> SparseVecCoord;
    typedef vector<SparseDeriv> SparseVecDeriv;

    //! All the Constraints applied to a state Vector
    typedef	vector<SparseVecDeriv> VecConst;

    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;

    static void set(Coord& c, double x, double y, double z)
    {
        c.getCenter()[0] = x;
        c.getCenter()[1] = y;
        c.getCenter()[2] = z;
    }

    static void get(double& x, double& y, double& z, const Coord& c)
    {
        x = c.getCenter()[0];
        y = c.getCenter()[1];
        z = c.getCenter()[2];
    }

    static void add(Coord& c, double x, double y, double z)
    {
        c.getCenter()[0] += x;
        c.getCenter()[1] += y;
        c.getCenter()[2] += z;
    }

    static void set(Deriv& c, double x, double y, double z)
    {
        c.getVCenter()[0] = x;
        c.getVCenter()[1] = y;
        c.getVCenter()[2] = z;
    }

    static void get(double& x, double& y, double& z, const Deriv& c)
    {
        x = c.getVCenter()[0];
        y = c.getVCenter()[1];
        z = c.getVCenter()[2];
    }

    static void add(Deriv& c, double x, double y, double z)
    {
        c.getVCenter()[0] += x;
        c.getVCenter()[1] += y;
        c.getVCenter()[2] += z;
    }

    static const char* Name()
    {
        return "Rigid";
    }
};

class RigidMass
{
public:
    double mass,volume;
    Mat3x3d inertiaMatrix;	      // Inertia matrix of the object
    Mat3x3d inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat3x3d invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat3x3d invInertiaMassMatrix; // inverse of inertiaMassMatrix
    RigidMass(double m=1)
    {
        mass = m;
        volume = 1;
        inertiaMatrix.identity();
        recalc();
    }
    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        invInertiaMatrix.invert(inertiaMatrix);
        invInertiaMassMatrix.invert(inertiaMassMatrix);
    }
    inline friend std::ostream& operator << (std::ostream& out, const RigidMass& m )
    {
        out<<"mass = "<<m.mass;
        out<<", volume = "<<m.volume;
        out<<", inertia = "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, RigidMass& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }
};



/// Specialization for potential energy
// inline RigidTypes::Deriv operator*( const Vec3d& g, const RigidMass& m)
// {
//     return RigidTypes::Deriv( Vec3d(0,0,0), g * m.mass );
// }


inline RigidTypes::Deriv operator*(const RigidTypes::Deriv& d, const RigidMass& m)
{
    RigidTypes::Deriv res;
    res.getVCenter() = d.getVCenter() * m.mass;
    res.getVOrientation() = m.inertiaMassMatrix * d.getVOrientation();
    return res;
}

inline RigidTypes::Deriv operator/(const RigidTypes::Deriv& d, const RigidMass& m)
{
    RigidTypes::Deriv res;
    res.getVCenter() = d.getVCenter() / m.mass;
    res.getVOrientation() = m.invInertiaMassMatrix * d.getVOrientation();
    return res;
}

} // namespace defaulttype


//================================================================================================================
// This is probably useless because the RigidObject actually contains its mass and computes its inertia forces itself:
//================================================================================================================

namespace core
{
namespace componentmodel
{
namespace behavior
{
/// Specialization of the inertia force for defaulttype::RigidTypes
template <>
inline defaulttype::RigidTypes::Deriv inertiaForce<
defaulttype::RigidTypes::Coord,
            defaulttype::RigidTypes::Deriv,
            objectmodel::BaseContext::Vec3,
            defaulttype::RigidMass,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::RigidMass& mass,
                    const defaulttype::RigidTypes::Coord& x,
                    const defaulttype::RigidTypes::Deriv& v
            )
{
    defaulttype::RigidTypes::Vec3 omega( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::RigidTypes::Vec3 origin = x.getCenter(), finertia, zero(0,0,0);

    finertia = -( aframe + omega.cross( omega.cross(origin) + v.getVCenter()*2 ))*mass.mass;
    return defaulttype::RigidTypes::Deriv( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}
} // namespace behavoir

} // namespace componentmodel

} // namespace core

} // namespace sofa


#endif
