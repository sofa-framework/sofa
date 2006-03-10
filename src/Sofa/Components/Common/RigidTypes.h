#ifndef SOFA_COMPONENTS_COMMON_RIGIDTYPES_H
#define SOFA_COMPONENTS_COMMON_RIGIDTYPES_H

#include "Vec.h"
#include "Quat.h"

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

        /// dot product
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
    };

    typedef std::vector<Coord> VecCoord;
    typedef std::vector<Deriv> VecDeriv;

    static void set(Coord& c, double x, double y, double z)
    {
        c.getCenter()[0] = x;
        c.getCenter()[1] = y;
        c.getCenter()[2] = z;
    }
};

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
