#ifndef SOFA_COMPONENTS_COMMON_RIGIDTYPES_H
#define SOFA_COMPONENTS_COMMON_RIGIDTYPES_H

#include "Vec.h"
#include "Mat.h"
#include "Quat.h"
#include <Sofa/Core/Context.h>
#include <Sofa/Core/Mass.h>
#include <vector>
#include <iostream>
using std::endl;

namespace Sofa
{

namespace Components
{

namespace Common
{

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
            d.vOrientation = vCenter + a.vOrientation;
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
        inline friend std::ostream& operator << (std::ostream& out, const Deriv& v )
        {
            out<<"vCenter = "<<v.getVCenter();
            out<<", vOrientation = "<<v.getVOrientation();
            return out;
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
        inline friend std::ostream& operator << (std::ostream& out, const Coord& c )
        {
            out<<"translation = "<<c.getCenter();
            out<<", rotation = "<<c.getOrientation();
            return out;
        }

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
    };

    typedef std::vector<Coord> VecCoord;
    typedef std::vector<Deriv> VecDeriv;

    static void set(Coord& c, double x, double y, double z)
    {
        c.getCenter()[0] = x;
        c.getCenter()[1] = y;
        c.getCenter()[2] = z;
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

    static void add(Deriv& c, double x, double y, double z)
    {
        c.getVCenter()[0] += x;
        c.getVCenter()[1] += y;
        c.getVCenter()[2] += z;
    }
};

class RigidMass
{
public:
    double mass;
    Mat3x3d inertiaMatrix;	      // Inertia matrix of the object
    Mat3x3d inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat3x3d invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat3x3d invInertiaMassMatrix; // inverse of inertiaMassMatrix
    RigidMass(double m=1.0)
    {
        mass = m;
        inertiaMatrix.identity();
        recalc();
    }
    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        invInertiaMatrix.invert(inertiaMatrix);
        invInertiaMassMatrix.invert(inertiaMassMatrix);
    }
};

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

} // namespace Common

} // namespace Components

//================================================================================================================
// This is probably useless because the RigidObject actually contains its mass and computes its inertia forces itself:
//================================================================================================================

namespace Core
{
/// Specialization of the inertia force for Components::Common::RigidTypes
template <>
inline Components::Common::RigidTypes::Deriv inertiaForce<
Components::Common::RigidTypes::Coord,
           Components::Common::RigidTypes::Deriv,
           Context::Vec3,
           Components::Common::RigidMass,
           Context::SpatialVector
           >
           (
                   const Context::SpatialVector& vframe,
                   const Context::Vec3& aframe,
                   const Components::Common::RigidMass& mass,
                   const Components::Common::RigidTypes::Coord& x,
                   const Components::Common::RigidTypes::Deriv& v )
{
    Components::Common::RigidTypes::Vec3 omega( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    Components::Common::RigidTypes::Vec3 origin = x.getCenter(), finertia, zero(0,0,0);

    finertia = -( aframe + omega.cross( omega.cross(origin) + v.getVCenter()*2 ))*mass.mass;
    return Components::Common::RigidTypes::Deriv( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

}

} // namespace Sofa

#endif
