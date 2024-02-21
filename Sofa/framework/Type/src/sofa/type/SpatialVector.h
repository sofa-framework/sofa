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
#pragma once

#include <sofa/type/Mat.h>
#include <sofa/type/config.h>

namespace sofa::type
{

/**
 * A spatial vector.
 * When representing a velocity, lineVec is the angular velocity and freeVec is the linear velocity.
 * When representing a spatial force, lineVec is the force and freeVec is the torque.
 * */
template<class TReal>
class SpatialVector
{
public:
    using Real = TReal;
    using Vec = sofa::type::Vec<3, TReal>;
    using Mat66 = sofa::type::Mat<6, 6, TReal>;

    Vec lineVec{ type::NOINIT };
    Vec freeVec{ type::NOINIT };

    void clear();
    SpatialVector() = default;
    /**
    \param l The line vector: angular velocity, or force
    \param f The free vector: linear velocity, or torque
    */
    SpatialVector( const Vec& l, const Vec& f );


    SpatialVector& operator+= (const SpatialVector& v);

    //template<class Real2>
    SpatialVector operator* ( Real a ) const
    {
        return SpatialVector( lineVec *a, freeVec * a);
    }

    SpatialVector& operator*= ( Real a )
    {
        lineVec *=a;
        freeVec *= a;
        return *this;
    }

    SpatialVector operator + ( const SpatialVector& v ) const;
    SpatialVector operator - ( const SpatialVector& v ) const;
    SpatialVector operator - ( ) const;
    /// Spatial dot product (cross terms)
    Real operator* ( const SpatialVector& v ) const;
    /// Spatial cross product
    SpatialVector cross( const SpatialVector& v ) const;
    /// product with a dense matrix
    SpatialVector operator* (const Mat66&) const;

    /// write to an output stream
    inline friend std::ostream& operator << (std::ostream& out, const SpatialVector& t )
    {
        out << t.lineVec << " " << t.freeVec;
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, SpatialVector& t )
    {
        in >> t.lineVec >> t.freeVec;
        return in;
    }

    /// If the SpatialVector models a spatial velocity, then the linear velocity is the freeVec.
    /// Otherwise, the SpatialVector models a spatial force, and this method returns a torque.
    Vec& getLinearVelocity()
    {
        return freeVec;
    }
    const Vec& getLinearVelocity() const
    {
        return freeVec;
    }
    void setLinearVelocity(const Vec& v)
    {
        freeVec = v;
    }
    /// If the SpatialVector models a spatial velocity, then the angular velocity is the lineVec.
    /// Otherwise, the SpatialVector models a spatial force, and this method returns a force.
    Vec& getAngularVelocity()
    {
        return lineVec;
    }
    const Vec& getAngularVelocity() const
    {
        return lineVec;
    }
    void setAngularVelocity(const Vec& v)
    {
        lineVec = v;
    }

    /// If the SpatialVector models a spatial force, then the torque is the freeVec.
    /// Otherwise, the SpatialVector models a spatial velocity, and this method returns a linear velocity.
    Vec& getTorque()
    {
        return freeVec;
    }
    const Vec& getTorque() const
    {
        return freeVec;
    }
    void setTorque(const Vec& v)
    {
        freeVec = v;
    }
    /// If the SpatialVector models a spatial force, then the torque is the lineVec.
    /// Otherwise, the SpatialVector models a spatial velocity, and this method returns an angular velocity.
    Vec& getForce()
    {
        return lineVec;
    }
    const Vec& getForce() const
    {
        return lineVec;
    }
    void setForce(const Vec& v)
    {
        lineVec = v;
    }
};

#if !defined(SOFA_TYPE_SPATIALVECTOR_CPP)
extern template class SOFA_TYPE_API SpatialVector<double>;
extern template class SOFA_TYPE_API SpatialVector<float>;
#endif

}
