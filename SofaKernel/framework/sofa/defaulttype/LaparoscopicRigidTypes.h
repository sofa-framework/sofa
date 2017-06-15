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
#ifndef SOFA_DEFAULTTYPE_LAPAROSCOPICRIGIDTYPES_H
#define SOFA_DEFAULTTYPE_LAPAROSCOPICRIGIDTYPES_H

#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <iostream>

#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace defaulttype
{

/// TODO
/// What this type is for?
/// A little description here ?
class LaparoscopicRigid3Types
{
public:

    typedef SReal Real;

    class Deriv
    {
    private:
        Real vTranslation;
        Vector3 vOrientation;
    public:
        typedef Real value_type;
        typedef int size_type;
        typedef Real Pos;
        typedef Vector3 Rot;
        friend class Coord;

        Deriv (const Real &velTranslation, const Vector3 &velOrient)
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

        void operator*=(Real a)
        {
            vTranslation *= a;
            vOrientation *= a;
        }

        Deriv operator*(Real a) const
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
        Real operator*(const Deriv& a) const
        {
            return vTranslation*a.vTranslation
                    +vOrientation[0]*a.vOrientation[0]+vOrientation[1]*a.vOrientation[1]
                    +vOrientation[2]*a.vOrientation[2];
        }

        Real& getVTranslation (void) { return vTranslation; }
        Vector3& getVOrientation (void) { return vOrientation; }
        const Real& getVTranslation (void) const { return vTranslation; }
        const Vector3& getVOrientation (void) const { return vOrientation; }
        inline friend std::ostream& operator << (std::ostream& out, const Deriv& v )
        {
            out<<v.getVTranslation();
            out<<" "<<v.getVOrientation();
            return out;
        }
        inline friend std::istream& operator >> (std::istream& in, Deriv& v )
        {
            in>>v.vTranslation;
            in>>v.vOrientation;
            return in;
        }

        /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
        enum { total_size = 4 };
        /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
        enum { spatial_dimensions = 3 };

        Real* ptr() { return &vTranslation; }
        const Real* ptr() const { return &vTranslation; }

        static unsigned int size() { return 4; }
        Real& operator[](int i)
        {
            if (i < 1) return this->vTranslation;
            else       return this->vOrientation(i-1);
        }
        const Real& operator[](int i) const
        {
            if (i < 1) return this->vTranslation;
            else       return this->vOrientation(i-1);
        }

        /// @name Comparison operators
        /// @{

        bool operator==(const Deriv& b) const
        {
            return vTranslation == b.vTranslation && vOrientation == b.vOrientation;
        }

        bool operator!=(const Deriv& b) const
        {
            return vTranslation != b.vTranslation || vOrientation != b.vOrientation;
        }

        /// @}

    };

    class Coord
    {

    private:
        Real translation;
        Quat orientation;
    public:
        typedef Real value_type;
        typedef int size_type;
        typedef Real Pos;
        typedef Quat Rot;
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
                orientation[i] += qDot[i] * (SReal)0.5;
            orientation.normalize();
        }

        Coord operator + (const Deriv& a) const
        {
            Coord c = *this;
            c.translation += a.getVTranslation();
            c.orientation.normalize();
            Quat qDot = c.orientation.vectQuatMult(a.getVOrientation());
            for (int i = 0; i < 4; i++)
                c.orientation[i] += qDot[i] * (SReal)0.5;
            c.orientation.normalize();
            return c;
        }

        void operator +=(const Coord& a)
        {
// 			std::cout << "+="<<std::endl;
            translation += a.getTranslation();
            //orientation += a.getOrientation();
            //orientation.normalize();
        }

        void operator*=(Real a)
        {
// 			std::cout << "*="<<std::endl;
            translation *= a;
            //orientation *= a;
        }

        Coord operator*(Real a) const
        {
            Coord r = *this;
            r*=a;
            return r;
        }

        /// dot product (FF: WHAT????  )
        Real operator*(const Coord& a) const
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
            out<<c.getTranslation();
            out<<" "<<c.getOrientation();
            return out;
        }
        inline friend std::istream& operator >> (std::istream& in, Coord& c )
        {
            in>>c.translation;
            in>>c.orientation;
            return in;
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
        Vector3 vectorToChild( const Vector3& v ) const
        {
            return orientation.inverseRotate(v);
        }


        /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
        enum { total_size = 5 };
        /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
        enum { spatial_dimensions = 3 };

        Real* ptr() { return &translation; }
        const Real* ptr() const { return &translation; }

        static unsigned int size() { return 5; }
        Real& operator[](int i)
        {
            if (i < 1) return this->translation;
            else       return this->orientation[i-1];
        }
        const Real& operator[](int i) const
        {
            if (i < 1) return this->translation;
            else       return this->orientation[i-1];
        }
    };

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef Coord::Pos CPos;
    typedef Coord::Rot CRot;
    static const CPos& getCPos(const Coord& c) { return c.getTranslation(); }
    static void setCPos(Coord& c, const CPos& v) { c.getTranslation() = v; }
    static const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef Deriv::Pos DPos;
    typedef Deriv::Rot DRot;
    static const DPos& getDPos(const Deriv& d) { return d.getVTranslation(); }
    static void setDPos(Deriv& d, const DPos& v) { d.getVTranslation() = v; }
    static const DRot& getDRot(const Deriv& d) { return d.getVOrientation(); }
    static void setDRot(Deriv& d, const DRot& v) { d.getVOrientation() = v; }

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    typedef helper::vector<Coord> VecCoord;
    typedef helper::vector<Deriv> VecDeriv;
    typedef helper::vector<Real> VecReal;

    template<typename T>
    static void set(Coord& c, T x, T, T)
    {
        c.getTranslation() = (Real)x;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& c)
    {
        x = (T)c.getTranslation();
        y = (T)0;
        z = (T)0;
    }

    template<typename T>
    static void add(Coord& c, T x, T, T)
    {
        c.getTranslation() += (Real)x;
    }

    template<typename T>
    static void set(Deriv& d, T x, T, T)
    {
        d.getVTranslation() = (Real)x;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Deriv& d)
    {
        x = (T)d.getVTranslation();
        y = (T)0;
        z = (T)0;
    }

    template<typename T>
    static void add(Deriv& d, T x, T, T)
    {
        d.getVTranslation() += (T)x;
    }
    static const char* Name()
    {
        return "LaparoscopicRigid3";
    }

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

    static Deriv interpolate(const helper::vector< Deriv > &ancestors, const helper::vector< Real > &coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }
};

inline LaparoscopicRigid3Types::Deriv operator*(const LaparoscopicRigid3Types::Deriv& d, const Rigid3Mass& m)
{
    LaparoscopicRigid3Types::Deriv res;
    res.getVTranslation() = d.getVTranslation() * m.mass;
    res.getVOrientation() = m.inertiaMassMatrix * d.getVOrientation();
    return res;
}

inline LaparoscopicRigid3Types::Deriv operator/(const LaparoscopicRigid3Types::Deriv& d, const Rigid3Mass& m)
{
    LaparoscopicRigid3Types::Deriv res;
    res.getVTranslation() = d.getVTranslation() / m.mass;
    res.getVOrientation() = m.invInertiaMassMatrix * d.getVOrientation();
    return res;
}

typedef LaparoscopicRigid3Types LaparoscopicRigidTypes; ///< Alias

// Specialization of the defaulttype::DataTypeInfo type traits template

template<>
struct DataTypeInfo< sofa::defaulttype::LaparoscopicRigid3Types::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::LaparoscopicRigid3Types::Deriv, sofa::defaulttype::LaparoscopicRigid3Types::Deriv::total_size >
{
    static const char* name() { return "LaparoscopicRigid3Types::Deriv"; }
};

template<>
struct DataTypeInfo< sofa::defaulttype::LaparoscopicRigid3Types::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::LaparoscopicRigid3Types::Coord, sofa::defaulttype::LaparoscopicRigid3Types::Coord::total_size >
{
    static const char* name() { return "LaparoscopicRigid3Types::Coord"; }
};

} // namespace defaulttype

} // namespace sofa

#endif
