#ifndef SOFA_COMPONENTS_COMMON_LAPAROSCOPICRIGIDTYPES_H
#define SOFA_COMPONENTS_COMMON_LAPAROSCOPICRIGIDTYPES_H

#include "RigidTypes.h"
#include <Sofa/Core/Context.h>
#include <Sofa/Core/Mass.h>
#include <Sofa/Components/Common/vector.h>
#include <iostream>
using std::endl;

namespace Sofa
{

namespace Components
{

namespace Common
{

using Sofa::Components::Common::vector;

class LaparoscopicRigidTypes
{
public:
    typedef Vec3d Vec3;
    typedef Vec3::value_type Real;

    class Deriv
    {
    private:
        Real vTranslation;
        Vec3 vOrientation;
    public:
        friend class Coord;

        Deriv (const Real &velTranslation, const Vec3 &velOrient)
            : vTranslation(velTranslation), vOrientation(velOrient) {}
        Deriv () { clear(); }

        void clear() { vTranslation = 0; vOrientation.clear(); }

        void operator +=(const Deriv& a)
        {
            vTranslation += a.vTranslation;
            vOrientation += a.vOrientation;
        }

        Deriv operator + (const Deriv& a) const
        {
            Deriv d;
            d.vTranslation = vTranslation + a.vTranslation;
            d.vOrientation = vOrientation + a.vOrientation;
            return d;
        }

        void operator*=(double a)
        {
            vTranslation *= a;
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
            return Deriv(-vTranslation, -vOrientation);
        }

        /// dot product
        double operator*(const Deriv& a) const
        {
            return vTranslation*a.vTranslation
                    +vOrientation[0]*a.vOrientation[0]+vOrientation[1]*a.vOrientation[1]
                    +vOrientation[2]*a.vOrientation[2];
        }

        Real& getVTranslation (void) { return vTranslation; }
        Vec3& getVOrientation (void) { return vOrientation; }
        const Real& getVTranslation (void) const { return vTranslation; }
        const Vec3& getVOrientation (void) const { return vOrientation; }
        inline friend std::ostream& operator << (std::ostream& out, const Deriv& v )
        {
            out<<"vTranslation = "<<v.getVTranslation();
            out<<", vOrientation = "<<v.getVOrientation();
            return out;
        }
    };

    class Coord
    {
    private:
        Real translation;
        Quat orientation;
    public:
        Coord (const Real &posTranslation, const Quat &orient)
            : translation(posTranslation), orientation(orient) {}
        Coord () { clear(); }

        void clear() { translation = 0; orientation.clear(); }

        void operator +=(const Deriv& a)
        {
            translation += a.getVTranslation();
            orientation.normalize();
            Quat qDot = orientation.vectQuatMult(a.getVOrientation());
            for (int i = 0; i < 4; i++)
                orientation[i] += qDot[i] * 0.5;
            orientation.normalize();
        }

        Coord operator + (const Deriv& a) const
        {
            Coord c = *this;
            c.translation += a.getVTranslation();
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
            translation += a.getTranslation();
            //orientation += a.getOrientation();
            //orientation.normalize();
        }

        void operator*=(double a)
        {
            std::cout << "*="<<std::endl;
            translation *= a;
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
            return translation*a.translation
                    +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                    +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
        }

        Real& getTranslation () { return translation; }
        Quat& getOrientation () { return orientation; }
        const Real& getTranslation () const { return translation; }
        const Quat& getOrientation () const { return orientation; }
        inline friend std::ostream& operator << (std::ostream& out, const Coord& c )
        {
            out<<"translation = "<<c.getTranslation();
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
            translation += c.getTranslation();
            orientation = orientation * c.getOrientation();
        }

        /// compute the product with another frame on the right
        Coord mult( const Coord& c ) const
        {
            Coord r;
            r.translation = translation + c.translation; //orientation.rotate( c.translation );
            r.orientation = orientation * c.getOrientation();
            return r;
        }
        /// compute the projection of a vector from the parent frame to the child
        Vec3 vectorToChild( const Vec3& v ) const
        {
            return orientation.inverseRotate(v);
        }
    };

    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;

    static void set(Coord& c, double x, double, double)
    {
        c.getTranslation() = x;
        //c.getTranslation()[1] = y;
        //c.getTranslation()[2] = z;
    }

    static void get(double& x, double&, double&, const Coord& c)
    {
        x = c.getTranslation();
        //y = c.getTranslation();
        //z = c.getTranslation()[2];
    }

    static void add(Coord& c, double x, double, double)
    {
        c.getTranslation() += x;
        //c.getTranslation()[1] += y;
        //c.getTranslation()[2] += z;
    }

    static void set(Deriv& c, double x, double, double)
    {
        c.getVTranslation() = x;
        //c.getVTranslation()[1] = y;
        //c.getVTranslation()[2] = z;
    }

    static void get(double& x, double& y, double& z, const Deriv& c)
    {
        x = c.getVTranslation();
        y = 0; //c.getVTranslation()[1];
        z = 0; //c.getVTranslation()[2];
    }

    static void add(Deriv& c, double x, double, double)
    {
        c.getVTranslation() += x;
        //c.getVTranslation()[1] += y;
        //c.getVTranslation()[2] += z;
    }
};

inline LaparoscopicRigidTypes::Deriv operator*(const LaparoscopicRigidTypes::Deriv& d, const RigidMass& m)
{
    LaparoscopicRigidTypes::Deriv res;
    res.getVTranslation() = d.getVTranslation() * m.mass;
    res.getVOrientation() = m.inertiaMassMatrix * d.getVOrientation();
    return res;
}

inline LaparoscopicRigidTypes::Deriv operator/(const LaparoscopicRigidTypes::Deriv& d, const RigidMass& m)
{
    LaparoscopicRigidTypes::Deriv res;
    res.getVTranslation() = d.getVTranslation() / m.mass;
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
inline Components::Common::LaparoscopicRigidTypes::Deriv inertiaForce<
Components::Common::LaparoscopicRigidTypes::Coord,
           Components::Common::LaparoscopicRigidTypes::Deriv,
           Context::Vec3,
           Components::Common::RigidMass,
           Context::SpatialVector
           >
           (
                   const Context::SpatialVector& vframe,
                   const Context::Vec3& aframe,
                   const Components::Common::RigidMass& mass,
                   const Components::Common::LaparoscopicRigidTypes::Coord& x,
                   const Components::Common::LaparoscopicRigidTypes::Deriv& v )
{
    Components::Common::RigidTypes::Vec3 omega( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    Components::Common::RigidTypes::Vec3 origin, finertia, zero(0,0,0);
    origin[0] = x.getTranslation();

    finertia = -( aframe + omega.cross( omega.cross(origin) + Components::Common::RigidTypes::Vec3(v.getVTranslation()*2,0,0) ))*mass.mass;
    return Components::Common::LaparoscopicRigidTypes::Deriv( finertia[0], zero );
    /// \todo replace zero by Jomega.cross(omega)
}

}

} // namespace Sofa

#endif
