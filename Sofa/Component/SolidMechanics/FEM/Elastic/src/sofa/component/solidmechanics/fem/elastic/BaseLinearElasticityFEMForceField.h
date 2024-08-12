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
#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>


namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
class BaseLinearElasticityFEMForceField : virtual public core::behavior::ForceField<DataTypes>
{
public:
    using Coord = typename DataTypes::Coord;
    using VecReal = typename DataTypes::VecReal;
    using Real = typename DataTypes::Real;

    SOFA_CLASS(SOFA_TEMPLATE(BaseLinearElasticityFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    Data<Real> d_poissonRatio; ///< FEM Poisson Ratio in Hooke's law [0,0.5[
    Data<VecReal > d_youngModulus; ///< FEM Young's Modulus in Hooke's law

    /// Link to be set to the topology container in the component graph.
    SingleLink<BaseLinearElasticityFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_topology;

    static VecReal GetDefaultYoungModulusValue();

    BaseLinearElasticityFEMForceField();
    void init() override;

    void setPoissonRatio(Real val);
    void setYoungModulus(Real val);

    Real getYoungModulusInElement(sofa::Size elementId);
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_BASELINEARELASTICITYFEMFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseLinearElasticityFEMForceField<defaulttype::Vec3Types>;
#endif

}
