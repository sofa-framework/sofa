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
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/ForceField.inl>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes>
BaseLinearElasticityFEMForceField<DataTypes>::BaseLinearElasticityFEMForceField()
    : d_poissonRatio(initData(&d_poissonRatio, { defaultPoissonRatioValue }, "poissonRatio", "FEM Poisson Ratio in Hooke's law [0,0.5["))
    , d_youngModulus(initData(&d_youngModulus, { defaultYoungModulusValue }, "youngModulus", "FEM Young's Modulus in Hooke's law"))
    , l_topology(initLink("topology", "link to the topology container"))
{
    d_poissonRatio.setRequired(true);
    d_youngModulus.setRequired(true);

    this->addUpdateCallback("checkPoissonRatio", {&d_poissonRatio}, [this](const core::DataTracker& )
    {
        checkPoissonRatio();
        return this->getComponentState();
    }, {});

    this->addUpdateCallback("checkPositiveYoungModulus", {&d_youngModulus}, [this](const core::DataTracker& )
    {
        checkYoungModulus();
        return this->getComponentState();
    }, {});
}

template <class DataTypes>
void BaseLinearElasticityFEMForceField<DataTypes>::checkPoissonRatio()
{
    auto poissonRatio = sofa::helper::getWriteAccessor(d_poissonRatio);
    for (auto& p : poissonRatio)
    {
        if (p < 0 || p >= 0.5)
        {
            msg_warning() << "Poisson's ratio must be in the range [0, 0.5), "
                    "but an out-of-bounds value has been provided (" <<
                    p << "). It is set to " << defaultPoissonRatioValue <<
                    " to ensure the correct behavior";
            p = defaultPoissonRatioValue;
        }
    }
}

template <class DataTypes>
void BaseLinearElasticityFEMForceField<DataTypes>::checkYoungModulus()
{
    auto youngModulus = sofa::helper::getWriteAccessor(d_youngModulus);
    for (auto& y : youngModulus)
    {
        if (y < 0)
        {
            msg_warning() << "Young's modulus must be positive, but "
                    "a negative value has been provided (" << y <<
                    "). It is set to " << defaultYoungModulusValue <<
                    " to ensure the correct behavior";
            y = defaultYoungModulusValue;
        }
    }
}

template <class DataTypes>
void BaseLinearElasticityFEMForceField<DataTypes>::init()
{
    core::behavior::ForceField<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (l_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << this->l_topology.getLinkedPath()
            << ", nor in current context: " << this->getContext()->name << ". Object must have a BaseMeshTopology. "
            << "The list of available BaseMeshTopology components is: "
            << core::ObjectFactory::getInstance()->listClassesDerivedFrom<core::topology::BaseMeshTopology>();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "Topology path used: '" << this->l_topology.getLinkedPath() << "'";

    checkYoungModulus();
    checkPoissonRatio();
}

template <class DataTypes>
void BaseLinearElasticityFEMForceField<DataTypes>::setPoissonRatio(Real val)
{
    d_poissonRatio.setValue({val});
}

template <class DataTypes>
void BaseLinearElasticityFEMForceField<DataTypes>::setYoungModulus(Real val)
{
    d_youngModulus.setValue({val});
}

template <class DataTypes>
auto BaseLinearElasticityFEMForceField<DataTypes>::getVecRealInElement(sofa::Size elementId, const Data<VecReal>& data, Real defaultValue) const
-> Real
{
    const auto& dataValue = data.getValue();
    if (dataValue.size() > elementId)
    {
        return dataValue[elementId];
    }
    if (!dataValue.empty())
    {
        return dataValue[0];
    }

    msg_warning() << "'" << data.getName() << "' Data field is empty. Using default value " << defaultValue;
    return defaultValue;
}


template <class DataTypes>
auto BaseLinearElasticityFEMForceField<DataTypes>::getYoungModulusInElement(sofa::Size elementId) const
-> Real
{
    return getVecRealInElement(elementId, d_youngModulus, defaultYoungModulusValue);
}

template <class DataTypes>
auto BaseLinearElasticityFEMForceField<DataTypes>::getPoissonRatioInElement(sofa::Size elementId) const
-> Real
{
    return getVecRealInElement(elementId, d_poissonRatio, defaultPoissonRatioValue);
}

template <class DataTypes>
auto BaseLinearElasticityFEMForceField<DataTypes>::toLameParameters(
    const _2DMaterials elementType,
    const Real youngModulus,
    const Real poissonRatio) -> std::pair<Real, Real>
{
    SOFA_UNUSED(elementType);

    //Lamé's first parameter
    const Real lambda = youngModulus * poissonRatio / (1 - poissonRatio * poissonRatio);

    //Lamé's second parameter (or shear modulus)
    const Real mu = youngModulus / (2 * (1 + poissonRatio));

    return {lambda, mu};
}

template <class DataTypes>
auto BaseLinearElasticityFEMForceField<DataTypes>::toLameParameters(
    const _3DMaterials elementType,
    const Real youngModulus,
    const Real poissonRatio) -> std::pair<Real, Real>
{
    SOFA_UNUSED(elementType);

    //Lamé's first parameter
    const Real lambda = youngModulus * poissonRatio / ((1 - 2 * poissonRatio) * (1 + poissonRatio));

    //Lamé's second parameter (or shear modulus)
    const Real mu = youngModulus / (2 * (1 + poissonRatio));

    return {lambda, mu};
}

}
