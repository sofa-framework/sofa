#pragma once
#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/LinearMechanicalParametersComponent.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/LameParameters.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
LinearMechanicalParametersComponent<DataTypes>::LinearMechanicalParametersComponent()
: d_poissonRatio(initData(&d_poissonRatio, static_cast<Real>(0.45), "poissonRatio",
    "Poisson's ratio: represents the material's ability to undergo deformation in directions orthogonal to the applied stress"))
, d_youngModulus(initData(&d_youngModulus, static_cast<Real>(1e6), "youngModulus",
    "Young's modulus: describes the material's stiffness"))
{
    this->addUpdateCallback("toLameCoefficients", {&this->d_youngModulus, &this->d_poissonRatio},
    [this](const sofa::core::DataTracker& )
    {
        setLameCoefficients();
        return this->getComponentState();
    }, {});
    setLameCoefficients();
}

template <class DataTypes>
void LinearMechanicalParametersComponent<DataTypes>::setLameCoefficients()
{
    sofa::component::solidmechanics::fem::elastic::LameLambda<Real> lambdaStrong { 0 };
    sofa::component::solidmechanics::fem::elastic::LameMu<Real> muStrong { 0 };

    sofa::component::solidmechanics::fem::elastic::toLameParameters<DataTypes::spatial_dimensions, Real>(
        sofa::component::solidmechanics::fem::elastic::YoungModulus<Real>(this->d_youngModulus.getValue()),
        sofa::component::solidmechanics::fem::elastic::PoissonRatio<Real>(this->d_poissonRatio.getValue()),
        lambdaStrong, muStrong);

    m_lambda = lambdaStrong.get();
    m_mu = muStrong.get();
}

}  // namespace elasticity
