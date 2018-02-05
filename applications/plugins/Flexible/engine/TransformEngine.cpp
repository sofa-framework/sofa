/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_FLEXIBLE_TRANSFORMENGINE_CPP
#include <Flexible/config.h>
#include <SofaGeneralEngine/TransformEngine.inl>
#include <sofa/core/ObjectFactory.h>
#include "../types/AffineTypes.h"

namespace sofa
{

namespace component
{

namespace engine
{


template <>
struct Translation <defaulttype::Affine3Types> : public TransformOperation<defaulttype::Affine3Types>
{
    typedef defaulttype::Affine3Types DataTypes;
    typedef DataTypes::Real Real;
    typedef DataTypes::SpatialCoord SpatialCoord;

    Translation() { t.clear(); }
    void execute(DataTypes::Coord &p) const
    {
        p.getCenter()+=t;
    }
    void configure(const defaulttype::Vector3 &_t, bool inverse)
    {
        if (inverse) { for(unsigned int i=0;i<DataTypes::spatial_dimensions;i++) t[i]=(Real)-_t[i]; }
        else { for(unsigned int i=0;i<DataTypes::spatial_dimensions;i++)  t[i]=(Real)_t[i]; }
    }
private:
    SpatialCoord t;
};



template <>
struct Scale<defaulttype::Affine3Types> : public TransformOperation<defaulttype::Affine3Types>
{
    typedef defaulttype::Affine3Types DataTypes;
    typedef DataTypes::Real Real;
    typedef DataTypes::SpatialCoord SpatialCoord;
    typedef DataTypes::Frame Frame;

    Scale() { for(unsigned int i=0;i<DataTypes::spatial_dimensions;i++)  s[i]=1.; }

    void execute(DataTypes::Coord &p) const
    {
        SpatialCoord c = p.getCenter();
        Frame affine = p.getAffine();
        for(unsigned int i=0;i<DataTypes::spatial_dimensions;i++)   { affine[i][i]*=s[i];  c[i]*=s[i]; }
        p.getCenter() = c;
        p.getAffine() = affine;
    }

    void configure(const defaulttype::Vector3 &_s, bool inverse)
    {
        if (inverse) { for(unsigned int i=0;i<DataTypes::spatial_dimensions;i++) s[i]=(Real)(1.0/_s[i]); }
        else { for(unsigned int i=0;i<DataTypes::spatial_dimensions;i++)  s[i]=_s[i]; }
    }
private:
    SpatialCoord s;
};


template<>
struct RotationSpecialized<defaulttype::Affine3Types, 3, false> : public TransformOperation<defaulttype::Affine3Types>
{
    typedef defaulttype::Affine3Types DataTypes;
    typedef DataTypes::Real Real;
    typedef DataTypes::SpatialCoord SpatialCoord;
    typedef DataTypes::Frame Frame;

    void execute(DataTypes::Coord &p) const
    {
        p.getCenter() = q.rotate(p.getCenter());
        Frame R; q.toMatrix(R);
        Frame affine = R*p.getAffine();
        p.getAffine() = affine;
    }

    void configure(const defaulttype::Vector3 &r, bool inverse)
    {
        q=helper::Quater<Real>::createQuaterFromEuler( r*(M_PI/180.0));
        if (inverse) q = q.inverse();
    }

    void configure(const defaulttype::Quaternion &qi, bool inverse, sofa::core::objectmodel::Base*)
    {
        q=qi;
        if (inverse) q = q.inverse();
    }
private:
    defaulttype::Quaternion q;
};


SOFA_DECL_CLASS(TransformAffineEngine)

int TransformAffineEngineClass = core::RegisterObject("Transform position of dofs")
        .add< TransformEngine<defaulttype::Affine3Types> >()
        ;

template class SOFA_Flexible_API TransformEngine<defaulttype::Affine3Types>;


} // namespace constraint

} // namespace component

} // namespace sofa

