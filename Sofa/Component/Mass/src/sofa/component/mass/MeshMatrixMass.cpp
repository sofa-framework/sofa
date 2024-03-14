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
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_CPP

#include <sofa/component/mass/MeshMatrixMass.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mass
{

using namespace sofa::type;
using namespace sofa::defaulttype;

template <>
Vec6 MeshMatrixMass<Vec3Types>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    const auto &vertexMass= d_vertexMass.getValue();
    const auto &edgeMass= d_edgeMass.getValue();

    const helper::ReadAccessor< DataVecCoord > x = vx;
    const helper::ReadAccessor< DataVecDeriv > v = vv;

    Vec6 momentum;
    for( unsigned int i=0 ; i<v.size() ; i++ )
    {
        Deriv linearMomentum = v[i] * vertexMass[i];
        for( Index j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( Index j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    const auto& edges = l_topology->getEdges();
    for(size_t i=0 ; i<l_topology->getNbEdges() ; ++i )
    {
        const sofa::Index v0 = edges[i][0];
        const sofa::Index v1 = edges[i][1];

        // is it correct to share the edge mass between the 2 vertices?
        const MassType m = edgeMass[i] * static_cast<MassType>(0.5);

        Deriv linearMomentum = v[v0] * m;
        for( Index j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        Deriv angularMomentum = cross( x[v0], linearMomentum );
        for( Index j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];

        linearMomentum = v[v1] * m;
        for( Index j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];
        angularMomentum = cross( x[v1], linearMomentum );
        for( Index j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}


// Register in the Factory
int MeshMatrixMassClass = core::RegisterObject("Define a specific mass for each particle")
        .add< MeshMatrixMass<Vec3Types> >()
        .add< MeshMatrixMass<Vec2Types> >()
        .add< MeshMatrixMass<Vec2Types, Vec3Types> >()
        .add< MeshMatrixMass<Vec1Types> >()
        .add< MeshMatrixMass<Vec1Types, Vec2Types> >()
        .add< MeshMatrixMass<Vec1Types, Vec3Types> >()
;

template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec3Types>;
template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec2Types>;
template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec2Types, defaulttype::Vec3Types>;
template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec1Types>;
template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec1Types, defaulttype::Vec2Types>;
template class SOFA_COMPONENT_MASS_API MeshMatrixMass<defaulttype::Vec1Types, defaulttype::Vec3Types>;



} // namespace sofa::component::mass
