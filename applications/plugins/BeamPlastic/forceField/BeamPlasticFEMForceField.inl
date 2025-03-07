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
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/core/visual/VisualParams.h>

#include "BeamPlasticFEMForceField.h"
#include <BeamPlastic/constitutiveLaw/RambergOsgood.h>


namespace sofa::plugin::beamplastic::component::forcefield::_beamplasticfemforcefield_
{

using core::objectmodel::BaseContext;
using sofa::plugin::beamplastic::component::constitutivelaw::RambergOsgood;

template<class DataTypes>
BeamPlasticFEMForceField<DataTypes>::BeamPlasticFEMForceField()
    : m_beamsData(initData(&m_beamsData, "beamsData", "Internal element data"))
    , d_usePrecomputedStiffness(initData(&d_usePrecomputedStiffness, true, "usePrecomputedStiffness",
                                         "indicates if a precomputed elastic stiffness matrix is used, instead of being computed by reduced integration"))
    , d_useConsistentTangentOperator(initData(&d_useConsistentTangentOperator, false, "useConsistentTangentOperator",
                                              "indicates wether to use a consistent tangent operator in the computation of the plastic stiffness matrix"))
    , d_isPerfectlyPlastic(initData(&d_isPerfectlyPlastic, false, "isPerfectlyPlastic", "indicates wether the behaviour model is perfectly plastic"))
    , d_modelName(initData(&d_modelName, std::string("RambergOsgood"), "modelName", "the name of the 1D contitutive law model to be used in plastic deformation"))
    , m_indexedElements(nullptr)
    , d_poissonRatio(initData(&d_poissonRatio,(Real)0.3f,"poissonRatio","Potion Ratio"))
    , d_youngModulus(initData(&d_youngModulus, (Real)5000, "youngModulus", "Young Modulus"))
    , d_initialYieldStress(initData(&d_initialYieldStress,(Real)6.0e8,"initialYieldStress","yield stress"))
    , d_zSection(initData(&d_zSection, (Real)0.2, "zSection", "length of the section in the z direction for rectangular beams"))
    , d_ySection(initData(&d_ySection, (Real)0.2, "ySection", "length of the section in the y direction for rectangular beams"))
    , d_useSymmetricAssembly(initData(&d_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , d_isTimoshenko(initData(&d_isTimoshenko,false,"isTimoshenko","implements a Timoshenko beam model"))
    , d_sectionShape(initData(&d_sectionShape,"rectangular","sectionShape","Geometry of the section shape (rectangular or circular)"))
{
    d_poissonRatio.setRequired(true);
    d_youngModulus.setReadOnly(true);
}

template<class DataTypes>
BeamPlasticFEMForceField<DataTypes>::BeamPlasticFEMForceField(Real poissonRatio, Real youngModulus, Real yieldStress, Real zSection,
                                                    Real ySection, bool useVD, bool isPlasticMuller, bool isTimoshenko,
                                                    bool isPlasticKrabbenhoft, bool isPerfectlyPlastic,
                                                    type::vector<Quat<SReal>> localOrientations)
    : m_beamsData(initData(&m_beamsData, "beamsData", "Internal element data"))
    , d_usePrecomputedStiffness(initData(&d_usePrecomputedStiffness, true, "usePrecomputedStiffness",
                                         "indicates if a precomputed elastic stiffness matrix is used, instead of being computed by reduced integration"))
    , d_useConsistentTangentOperator(initData(&d_useConsistentTangentOperator, false, "useConsistentTangentOperator",
                                              "indicates wether to use a consistent tangent operator in the computation of the plastic stiffness matrix"))
    , d_isPerfectlyPlastic(initData(&d_isPerfectlyPlastic, false, "isPerfectlyPlastic", "indicates wether the behaviour model is perfectly plastic"))
    , d_modelName(initData(&d_modelName, std::string("RambergOsgood"), "modelName", "the name of the 1D contitutive law model to be used in plastic deformation"))
    , m_indexedElements(nullptr)
    , d_poissonRatio(initData(&d_poissonRatio,(Real)poissonRatio,"poissonRatio","Potion Ratio"))
    , d_youngModulus(initData(&d_youngModulus,(Real)youngModulus,"youngModulus","Young Modulus"))
    , d_initialYieldStress(initData(&d_initialYieldStress, (Real)yieldStress, "initialYieldStress", "yield stress"))
    , d_zSection(initData(&d_zSection, (Real)zSection, "zSection", "length of the section in the z direction for rectangular beams"))
    , d_ySection(initData(&d_ySection, (Real)ySection, "ySection", "length of the section in the y direction for rectangular beams"))
    , d_useSymmetricAssembly(initData(&d_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , d_isTimoshenko(initData(&d_isTimoshenko, isTimoshenko, "isTimoshenko", "implements a Timoshenko beam model"))
    , d_sectionShape(initData(&d_sectionShape, "rectangular", "sectionShape", "Geometry of the section shape (rectangular or circular)"))
{
    d_poissonRatio.setRequired(true);
    d_youngModulus.setReadOnly(true);
}

template<class DataTypes>
BeamPlasticFEMForceField<DataTypes>::~BeamPlasticFEMForceField()
{

}

/*****************************************************************************/
/*                           INITIALISATION                                  */
/*****************************************************************************/

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::bwdInit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);
    m_lastUpdatedStep=-1.0;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::init()
{
    Inherit1::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name << ". Object must have a BaseMeshTopology (i.e. EdgeSetTopology or MeshTopology)";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (m_topology->getNbEdges() == 0)
    {
        msg_error() << "Topology is empty.";
        return;
    }
    m_indexedElements = &m_topology->getEdges();

    // Retrieving the 1D plastic constitutive law model
    std::string constitutiveModel = d_modelName.getValue();
    if (constitutiveModel == "RambergOsgood")
    {
        Real youngModulus = d_youngModulus.getValue();
        Real yieldStress = d_initialYieldStress.getValue();

        //Initialisation of the comparison threshold for stress tensor norms to 0.
        // Plasticity computation requires to basically compare stress tensor norms to 0.
        // As stress norm values can vary of several orders of magnitude, depending on the
        // considered materials and/or applied forces, this comparison has to be carried out
        // carefully.
        // The idea here is to use the initialYieldStress of the material, and the
        // available precision limit (e.g. std::numeric_limits<double>::epsilon()).
        // We rely on the value of the initial Yield stress, as we can expect plastic
        // deformation to occur inside a relatively small intervl of stresses around this value.
        const int orderOfMagnitude = d_initialYieldStress.getValue(); //Should use std::abs, but d_initialYieldStress > 0
        m_stressComparisonThreshold = std::numeric_limits<double>::epsilon() * orderOfMagnitude;

        m_ConstitutiveLaw = std::unique_ptr<RambergOsgood<DataTypes>>(new RambergOsgood<DataTypes>(youngModulus, yieldStress));
        if (this->f_printLog.getValue())
            msg_info() << "The model is " << constitutiveModel;
    }
    else
    {
        msg_error() << "constitutive law model name " << constitutiveModel << " is not valid (should be RambergOsgood)";
    }

    m_beamsData.createTopologyHandler(m_topology);

    reinit();
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::reinit()
{
    const size_t n = m_indexedElements->size();

    //Initialises the lastPos field with the rest position
    m_lastPos = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    m_prevStresses.resize(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 27; j++)
            m_prevStresses[i][j] = VoigtTensor2();

    if (d_useConsistentTangentOperator.getValue())
    {
        // No need to store elastic predictors at each iteration if the consistent
        // tangent operator is not used.
        m_elasticPredictors.resize(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 27; j++)
                m_elasticPredictors[i][j] = VoigtTensor2();
    }

    initBeams( n );
    for (unsigned int i=0; i<n; ++i)
        reinitBeam(i);
    msg_info() << "reinit OK, "<<n<<" elements." ;
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::initBeams(size_t size)
{
    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    bd.resize(size);
    m_beamsData.endEdit();
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::reinitBeam(unsigned int i)
{
    double stiffness, yieldStress, length, poisson, zSection, ySection;
    Index a = (*m_indexedElements)[i][0];
    Index b = (*m_indexedElements)[i][1];

    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    stiffness =  d_youngModulus.getValue();

    yieldStress = d_initialYieldStress.getValue();
    length = (x0[a].getCenter()-x0[b].getCenter()).norm() ;

    zSection = d_zSection.getValue();
    ySection = d_ySection.getValue();
    poisson = d_poissonRatio.getValue();

    setBeam(i, stiffness, yieldStress, length, poisson, zSection, ySection);

    computeMaterialBehaviour(i, a, b);

    // Initialisation of the elastic stiffness matrix
    if (d_usePrecomputedStiffness.getValue())
        computeStiffness(i, a, b);
    else
        computeVDStiffness(i, a, b);
    // Initialisation of the tangent stiffness matrix
    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    Matrix12x12& Kt_loc = bd[i]._Kt_loc;
    Kt_loc.clear();
    m_beamsData.endEdit();

    // Initialisation of the beam element orientation
    //TO DO: is necessary ?
    beamQuat(i) = x0[a].getOrientation();
    beamQuat(i).normalize();
    m_beamsData.endEdit(); // consecutive to beamQuat

}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::setBeam(unsigned int i, double E, double yS, double L, double nu, double zSection, double ySection)
{
    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    bd[i].init(E, yS, L, nu, zSection, ySection, d_isTimoshenko.getValue());
    m_beamsData.endEdit();
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::BeamInfo::init(double E, double yS, double L, double nu, double zSection, double ySection, bool isTimoshenko)
{
    _E = E;
    _nu = nu;
    _L = L;

    _zDim = zSection;
    _yDim = ySection;

    _G = _E / (2.0*(1.0 + _nu));
    _Iz = ySection*ySection*ySection*zSection / 12.0;

    _Iy = zSection*zSection*zSection*ySection / 12.0;
    _J = _Iz + _Iy;
    _A = zSection*ySection;

    double phiY, phiZ;
    double L2 = L*L;
    double kappaY = 1.0;
    double kappaZ = 1.0;

    if (_A == 0)
    {
        phiY = 0.0;
        phiZ = 0.0;
    }
    else
    {
        phiY = (12.0*_E*_Iy / (kappaZ*_G*_A*L2));
        phiZ = (12.0*_E*_Iz / (kappaY*_G*_A*L2));
    }

    double phiYInv = (1 / (1 + phiY));
    double phiZInv = (1 / (1 + phiZ));

    _integrationInterval = ozp::quadrature::make_interval(0, -ySection / 2, -zSection / 2, L, ySection / 2, zSection / 2);

    //Computation of the Be matrix for this beam element, based on the integration points.

    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    int i = 0; //Gauss Point iterator

    //Euler-Bernoulli beam theory
    LambdaType initBeMatrixEulerB = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        // Step 1: total strain computation
        double xi = u1 / _L;
        double eta = u2 / _L;
        double zeta = u3 / _L;

        _BeMatrices[i](0, 0) = -1 / _L;
        _BeMatrices[i](1, 0) = _BeMatrices[i](2, 0) = _BeMatrices[i](3, 0) = _BeMatrices[i](4, 0) = _BeMatrices[i](5, 0) = 0.0;

        _BeMatrices[i](0, 1) = (6 * eta*(1 - 2 * xi)) / _L;
        _BeMatrices[i](1, 1) = _BeMatrices[i](2, 1) = _BeMatrices[i](3, 1) = _BeMatrices[i](4, 1) = _BeMatrices[i](5, 1) = 0.0;

        _BeMatrices[i](0, 2) = (6 * zeta*(1 - 2 * xi)) / _L;
        _BeMatrices[i](1, 2) = _BeMatrices[i](2, 2) = _BeMatrices[i](3, 2) = _BeMatrices[i](4, 2) = _BeMatrices[i](5, 2) = 0.0;

        _BeMatrices[i](0, 3) = _BeMatrices[i](1, 3) = _BeMatrices[i](2, 3) = 0.0;
        _BeMatrices[i](3, 3) = 0;
        _BeMatrices[i](4, 3) = -eta / 2;
        _BeMatrices[i](5, 3) = zeta / 2;

        _BeMatrices[i](0, 4) = zeta * (6 * xi - 4);
        _BeMatrices[i](1, 4) = _BeMatrices[i](2, 4) = _BeMatrices[i](3, 4) = _BeMatrices[i](4, 4) = _BeMatrices[i](5, 4) = 0.0;

        _BeMatrices[i](0, 5) = eta * (4 - 6 * xi);
        _BeMatrices[i](1, 5) = _BeMatrices[i](2, 5) = _BeMatrices[i](3, 5) = _BeMatrices[i](4, 5) = _BeMatrices[i](5, 5) = 0.0;

//        _BeMatrices[i].block<6, 1>(0, 6) = -_BeMatrices[i].block<6, 1>(0, 0);

//        _BeMatrices[i].block<6, 1>(0, 7) = -_BeMatrices[i].block<6, 1>(0, 1);

//        _BeMatrices[i].block<6, 1>(0, 8) = -_BeMatrices[i].block<6, 1>(0, 2);

        Mat<6, 1, Real> subMat;
        _BeMatrices[i].getsub(0, 0, subMat);
        _BeMatrices[i].setsub(0, 6, -subMat);
        _BeMatrices[i].getsub(0, 1, subMat);
        _BeMatrices[i].setsub(0, 7, -subMat);
        _BeMatrices[i].getsub(0, 2, subMat);
        _BeMatrices[i].setsub(0, 8, -subMat);

        _BeMatrices[i](0, 9) = _BeMatrices[i](1, 9) = _BeMatrices[i](2, 9) = 0.0;
        _BeMatrices[i](3, 9) = 0;
        _BeMatrices[i](4, 9) = eta / 2;
        _BeMatrices[i](5, 9) = -zeta / 2;

        _BeMatrices[i](0, 10) = zeta * (6 * xi - 2);
        _BeMatrices[i](1, 10) = _BeMatrices[i](2, 10) = _BeMatrices[i](3, 10) = _BeMatrices[i](4, 10) = _BeMatrices[i](5, 10) = 0.0;

        _BeMatrices[i](0, 11) = eta * (2 - 6 * xi);
        _BeMatrices[i](1, 11) = _BeMatrices[i](2, 11) = _BeMatrices[i](3, 11) = _BeMatrices[i](4, 11) = _BeMatrices[i](5, 11) = 0.0;

        i++;
    };

    //Timoshenko beam theory
    LambdaType initBeMatrixTimo = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        // Step 1: total strain computation
        double xi = u1 / _L;
        double eta = u2 / _L;
        double zeta = u3 / _L;

        //row 0
        _BeMatrices[i](0, 0) = -1 / _L;
        _BeMatrices[i](0, 1) = (phiZInv * 6 * eta * (1 - 2 * xi)) / _L;
        _BeMatrices[i](0, 2) = (phiYInv * 6 * zeta * (1 - 2 * xi)) / _L;
        _BeMatrices[i](0, 3) = 0;
        _BeMatrices[i](0, 4) = phiYInv * zeta * (6 * xi - 4 - phiY);
        _BeMatrices[i](0, 5) = phiZInv * eta * (4 - 6 * xi + phiZ);
        _BeMatrices[i](0, 6) = 1 / _L;
        _BeMatrices[i](0, 7) = (phiZInv * 6 * eta * (2 * xi - 1)) / _L;
        _BeMatrices[i](0, 8) = (phiYInv * 6 * zeta * (2 * xi - 1)) / _L;
        _BeMatrices[i](0, 9) = 0;
        _BeMatrices[i](0, 10) = phiYInv * zeta * (6 * xi - 2 + phiY);
        _BeMatrices[i](0, 11) = phiZInv * eta * (2 - 6 * xi - phiZ);

        //rows 1, 2, 3
        _BeMatrices[i](1, 0) = _BeMatrices[i](1, 1) = _BeMatrices[i](1, 2) = _BeMatrices[i](1, 3) = 0.0;
        _BeMatrices[i](1, 4) = _BeMatrices[i](1, 5) = _BeMatrices[i](1, 6) = _BeMatrices[i](1, 7) = 0.0;
        _BeMatrices[i](1, 8) = _BeMatrices[i](1, 9) = _BeMatrices[i](1, 10) = _BeMatrices[i](1, 11) = 0.0;

        _BeMatrices[i](2, 0) = _BeMatrices[i](2, 1) = _BeMatrices[i](2, 2) = _BeMatrices[i](2, 3) = 0.0;
        _BeMatrices[i](2, 4) = _BeMatrices[i](2, 5) = _BeMatrices[i](2, 6) = _BeMatrices[i](2, 7) = 0.0;
        _BeMatrices[i](2, 8) = _BeMatrices[i](2, 9) = _BeMatrices[i](2, 10) = _BeMatrices[i](2, 11) = 0.0;

        _BeMatrices[i](3, 0) = _BeMatrices[i](3, 1) = _BeMatrices[i](3, 2) = _BeMatrices[i](3, 3) = 0.0;
        _BeMatrices[i](3, 4) = _BeMatrices[i](3, 5) = _BeMatrices[i](3, 6) = _BeMatrices[i](3, 7) = 0.0;
        _BeMatrices[i](3, 8) = _BeMatrices[i](3, 9) = _BeMatrices[i](3, 10) = _BeMatrices[i](3, 11) = 0.0;

        //row 4
        _BeMatrices[i](4, 2) = -(phiYInv * phiY) / (2 * _L);
        _BeMatrices[i](4, 3) = -eta / 2;
        _BeMatrices[i](4, 4) = (phiYInv * phiY) / 4;
        _BeMatrices[i](4, 8) = (phiYInv * phiY) / (2 * _L);
        _BeMatrices[i](4, 9) = eta / 2;
        _BeMatrices[i](4, 10) = (phiYInv * phiY) / 4;
        _BeMatrices[i](4, 0) = _BeMatrices[i](4, 1) = _BeMatrices[i](4, 5) = 0.0;
        _BeMatrices[i](4, 6) = _BeMatrices[i](4, 7) = _BeMatrices[i](4, 11) = 0.0;

        //row5
        _BeMatrices[i](5, 1) = -(phiZInv * phiZ) / (2 * _L);
        _BeMatrices[i](5, 3) = zeta / 2;
        _BeMatrices[i](5, 5) = -(phiZInv * phiZ) / 4;
        _BeMatrices[i](5, 7) = (phiZInv * phiZ) / (2 * _L);
        _BeMatrices[i](5, 9) = -zeta / 2;
        _BeMatrices[i](5, 11) = -(phiZInv * phiZ) / 4;
        _BeMatrices[i](5, 0) = _BeMatrices[i](5, 2) = _BeMatrices[i](5, 4) = 0.0;
        _BeMatrices[i](5, 6) = _BeMatrices[i](5, 8) = _BeMatrices[i](5, 10) = 0.0;

        i++;
    };

    if (isTimoshenko)
        ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(_integrationInterval, initBeMatrixTimo);
    else
        ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(_integrationInterval, initBeMatrixEulerB);


    int gaussPointIt = 0; //Gauss Point iterator

    // Euler-Bernoulli beam model
    LambdaType initialiseEBShapeFunctions = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        // Step 1: total strain computation
        double xi = u1 / _L;
        double eta = u2 / _L;
        double zeta = u3 / _L;

        double xi2 = xi*xi;
        double xi3 = xi*xi*xi;

        _N[gaussPointIt](0, 0) = 1 - xi;
        _N[gaussPointIt](0, 1) = 6 * (xi - xi2)*eta;
        _N[gaussPointIt](0, 2) = 6 * (xi - xi2)*zeta;
        _N[gaussPointIt](0, 3) = 0;
        _N[gaussPointIt](0, 4) = (1 - 4 * xi + 3 * xi2)*_L*zeta;
        _N[gaussPointIt](0, 5) = (-1 + 4 * xi - 3 * xi2)*_L*eta;
        _N[gaussPointIt](0, 6) = xi;
        _N[gaussPointIt](0, 7) = 6 * (-xi + xi2)*eta;
        _N[gaussPointIt](0, 8) = 6 * (-xi + xi2)*zeta;
        _N[gaussPointIt](0, 9) = 0;
        _N[gaussPointIt](0, 10) = (-2 * xi + 3 * xi2)*_L*zeta;
        _N[gaussPointIt](0, 11) = (2 * xi - 3 * xi2)*_L*eta;

        _N[gaussPointIt](1, 0) = 0;
        _N[gaussPointIt](1, 1) = 1 - 3 * xi2 + 2 * xi3;
        _N[gaussPointIt](1, 2) = 0;
        _N[gaussPointIt](1, 3) = (xi - 1)*_L*zeta;
        _N[gaussPointIt](1, 4) = 0;
        _N[gaussPointIt](1, 5) = (xi - 2 * xi2 + xi3)*_L;
        _N[gaussPointIt](1, 6) = 0;
        _N[gaussPointIt](1, 7) = 3 * xi2 - 2 * xi3;
        _N[gaussPointIt](1, 8) = 0;
        _N[gaussPointIt](1, 9) = -_L*xi*zeta;
        _N[gaussPointIt](1, 10) = 0;
        _N[gaussPointIt](1, 11) = (-xi2 + xi3)*_L;

        _N[gaussPointIt](2, 0) = 0;
        _N[gaussPointIt](2, 1) = 0;
        _N[gaussPointIt](2, 2) = 1 - 3 * xi2 + 2 * xi3;
        _N[gaussPointIt](2, 3) = (1 - xi)*_L*eta;
        _N[gaussPointIt](2, 4) = (-xi + 2 * xi2 - xi3)*_L;
        _N[gaussPointIt](2, 5) = 0;
        _N[gaussPointIt](2, 6) = 0;
        _N[gaussPointIt](2, 7) = 0;
        _N[gaussPointIt](2, 8) = 3 * xi2 - 2 * xi3;
        _N[gaussPointIt](2, 9) = _L*xi*eta;
        _N[gaussPointIt](2, 10) = (xi2 - xi3)*_L;
        _N[gaussPointIt](2, 11) = 0;

        gaussPointIt++;
    };

    // Timoshenko beam model
    LambdaType initialiseTShapeFunctions = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        double xi = u1 / _L;
        double eta = u2 / _L;
        double zeta = u3 / _L;

        double xi2 = xi*xi;
        double xi3 = xi*xi*xi;

        _N[gaussPointIt](0, 0) = 1 - xi;
        _N[gaussPointIt](0, 1) = 6 * phiZInv * (xi - xi2)*eta;
        _N[gaussPointIt](0, 2) = 6 * phiYInv * (xi - xi2)*zeta;
        _N[gaussPointIt](0, 3) = 0;
        _N[gaussPointIt](0, 4) = _L * phiYInv * (1 - 4 * xi + 3 * xi2 + phiY*(1 - xi))*zeta;
        _N[gaussPointIt](0, 5) = -_L * phiZInv * (1 - 4 * xi + 3 * xi2 + phiZ*(1 - xi))*eta;
        _N[gaussPointIt](0, 6) = xi;
        _N[gaussPointIt](0, 7) = 6 * phiZInv * (-xi + xi2)*eta;
        _N[gaussPointIt](0, 8) = 6 * phiYInv * (-xi + xi2)*zeta;
        _N[gaussPointIt](0, 9) = 0;
        _N[gaussPointIt](0, 10) = _L * phiYInv * (-2 * xi + 3 * xi2 + phiY*xi)*zeta;
        _N[gaussPointIt](0, 11) = -_L * phiZInv * (-2 * xi + 3 * xi2 + phiZ*xi)*eta;

        _N[gaussPointIt](1, 0) = 0;
        _N[gaussPointIt](1, 1) = phiZInv * (1 - 3 * xi2 + 2 * xi3 + phiZ*(1 - xi));
        _N[gaussPointIt](1, 2) = 0;
        _N[gaussPointIt](1, 3) = (xi - 1)*_L*zeta;
        _N[gaussPointIt](1, 4) = 0;
        _N[gaussPointIt](1, 5) = _L * phiZInv * (xi - 2 * xi2 + xi3 + (phiZ / 2)*(xi - xi2));
        _N[gaussPointIt](1, 6) = 0;
        _N[gaussPointIt](1, 7) = phiZInv * (3 * xi2 - 2 * xi3 + phiZ*xi);
        _N[gaussPointIt](1, 8) = 0;
        _N[gaussPointIt](1, 9) = -_L * xi  *zeta;
        _N[gaussPointIt](1, 10) = 0;
        _N[gaussPointIt](1, 11) = _L * phiZInv *(-xi2 + xi3 - (phiZ / 2)*(xi - xi2));

        _N[gaussPointIt](2, 0) = 0;
        _N[gaussPointIt](2, 1) = 0;
        _N[gaussPointIt](2, 2) = phiYInv * (1 - 3 * xi2 + 2 * xi3 + phiY*(1 - xi));
        _N[gaussPointIt](2, 3) = (1 - xi) * _L * eta;
        _N[gaussPointIt](2, 4) = -_L * phiYInv * (xi - 2 * xi2 + xi3 + (phiY / 2)*(xi - xi2));
        _N[gaussPointIt](2, 5) = 0;
        _N[gaussPointIt](2, 6) = 0;
        _N[gaussPointIt](2, 7) = 0;
        _N[gaussPointIt](2, 8) = phiYInv * (3 * xi2 - 2 * xi3 + phiY*xi);
        _N[gaussPointIt](2, 9) = _L * xi * eta;
        _N[gaussPointIt](2, 10) = -_L * phiYInv * (-xi2 + xi3 - (phiY / 2)*(xi - xi2));
        _N[gaussPointIt](2, 11) = 0;

        gaussPointIt++;
    };

    if (isTimoshenko)
        ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(_integrationInterval, initialiseTShapeFunctions);
    else
        ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(_integrationInterval, initialiseEBShapeFunctions);

    //Intialises the drawing points shape functions

    if (!isTimoshenko)
    {
        for (int i = 1; i < _nbCentrelineSeg; i++)
        {
            double xi = i*(_L / _nbCentrelineSeg) / _L;
            double xi2 = xi*xi;
            double xi3 = xi*xi*xi;

            //NB :
            //double eta = 0;
            //double zeta = 0;

            _drawN[i - 1](0, 0) = 1 - xi;
            _drawN[i - 1](0, 1) = 0;
            _drawN[i - 1](0, 2) = 0;
            _drawN[i - 1](0, 3) = 0;
            _drawN[i - 1](0, 4) = 0;
            _drawN[i - 1](0, 5) = 0;
            _drawN[i - 1](0, 6) = xi;
            _drawN[i - 1](0, 7) = 0;
            _drawN[i - 1](0, 8) = 0;
            _drawN[i - 1](0, 9) = 0;
            _drawN[i - 1](0, 10) = 0;
            _drawN[i - 1](0, 11) = 0;

            _drawN[i - 1](1, 0) = 0;
            _drawN[i - 1](1, 1) = 1 - 3 * xi2 + 2 * xi3;
            _drawN[i - 1](1, 2) = 0;
            _drawN[i - 1](1, 3) = 0;
            _drawN[i - 1](1, 4) = 0;
            _drawN[i - 1](1, 5) = (xi - 2 * xi2 + xi3)*_L;
            _drawN[i - 1](1, 6) = 0;
            _drawN[i - 1](1, 7) = 3 * xi2 - 2 * xi3;
            _drawN[i - 1](1, 8) = 0;
            _drawN[i - 1](1, 9) = 0;
            _drawN[i - 1](1, 10) = 0;
            _drawN[i - 1](1, 11) = (-xi2 + xi3)*_L;

            _drawN[i - 1](2, 0) = 0;
            _drawN[i - 1](2, 1) = 0;
            _drawN[i - 1](2, 2) = 1 - 3 * xi2 + 2 * xi3;
            _drawN[i - 1](2, 3) = 0;
            _drawN[i - 1](2, 4) = (-xi + 2 * xi2 - xi3)*_L;
            _drawN[i - 1](2, 5) = 0;
            _drawN[i - 1](2, 6) = 0;
            _drawN[i - 1](2, 7) = 0;
            _drawN[i - 1](2, 8) = 3 * xi2 - 2 * xi3;
            _drawN[i - 1](2, 9) = 0;
            _drawN[i - 1](2, 10) = (xi2 - xi3)*_L;
            _drawN[i - 1](2, 11) = 0;
        }
    }
    else
    {
        for (int i = 1; i < _nbCentrelineSeg; i++)
        {
            double xi = i*(_L / _nbCentrelineSeg) / _L;
            double xi2 = xi*xi;
            double xi3 = xi*xi*xi;

            //NB :
            //double eta = 0;
            //double zeta = 0;
            _drawN[i - 1](0, 0) = 1 - xi;
            _drawN[i - 1](0, 1) = 0;
            _drawN[i - 1](0, 2) = 0;
            _drawN[i - 1](0, 3) = 0;
            _drawN[i - 1](0, 4) = 0;
            _drawN[i - 1](0, 5) = 0;
            _drawN[i - 1](0, 6) = xi;
            _drawN[i - 1](0, 7) = 0;
            _drawN[i - 1](0, 8) = 0;
            _drawN[i - 1](0, 9) = 0;
            _drawN[i - 1](0, 10) = 0;
            _drawN[i - 1](0, 11) = 0;

            _drawN[i - 1](1, 0) = 0;
            _drawN[i - 1](1, 1) = phiZInv * (1 - 3 * xi2 + 2 * xi3 + phiZ*(1 - xi));
            _drawN[i - 1](1, 2) = 0;
            _drawN[i - 1](1, 3) = 0;
            _drawN[i - 1](1, 4) = 0;
            _drawN[i - 1](1, 5) = _L * phiZInv * (xi - 2 * xi2 + xi3 + (phiZ / 2)*(xi - xi2));
            _drawN[i - 1](1, 6) = 0;
            _drawN[i - 1](1, 7) = phiZInv * (3 * xi2 - 2 * xi3 + phiZ*xi);
            _drawN[i - 1](1, 8) = 0;
            _drawN[i - 1](1, 9) = 0;
            _drawN[i - 1](1, 10) = 0;
            _drawN[i - 1](1, 11) = _L * phiZInv *(-xi2 + xi3 - (phiZ / 2)*(xi - xi2));

            _drawN[i - 1](2, 0) = 0;
            _drawN[i - 1](2, 1) = 0;
            _drawN[i - 1](2, 2) = phiYInv * (1 - 3 * xi2 + 2 * xi3 + phiY*(1 - xi));
            _drawN[i - 1](2, 3) = 0;
            _drawN[i - 1](2, 4) = -_L * phiYInv * (xi - 2 * xi2 + xi3 + (phiY / 2)*(xi - xi2));
            _drawN[i - 1](2, 5) = 0;
            _drawN[i - 1](2, 6) = 0;
            _drawN[i - 1](2, 7) = 0;
            _drawN[i - 1](2, 8) = phiYInv * (3 * xi2 - 2 * xi3 + phiY*xi);
            _drawN[i - 1](2, 9) = 0;
            _drawN[i - 1](2, 10) = -_L * phiYInv * (-xi2 + xi3 - (phiY / 2)*(xi - xi2));
            _drawN[i - 1](2, 11) = 0;
        }
    }

    // Initialises the plastic indicators
    // NB: each vector contains 27 components,
    // associated with the 27 Gauss points used for reduced integration
    _pointMechanicalState.assign(MechanicalState::ELASTIC);
    _beamMechanicalState = MechanicalState::ELASTIC;
    
    _localYieldStresses.assign(yS);
    _backStresses.assign(VoigtTensor2()); // TO DO: check if zero is correct
    _effectivePlasticStrains.assign(0.0);


    //**********************************//
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::reset()
{
    for (unsigned i = 0; i < m_prevStresses.size(); ++i)
        for (unsigned j = 0; j < 27; ++j)
            m_prevStresses[i][j] = VoigtTensor2();

    if (d_useConsistentTangentOperator.getValue())
    {
        for (unsigned i = 0; i < m_elasticPredictors.size(); ++i)
            for (unsigned j = 0; j < 27; ++j)
                m_elasticPredictors[i][j] = VoigtTensor2();

    }

    // TO DO: call to init?
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeStiffness(int i, Index, Index)
{
    Real   phiy, phiz;
    Real _L = m_beamsData.getValue()[i]._L;
    Real _A = m_beamsData.getValue()[i]._A;
    Real _nu = m_beamsData.getValue()[i]._nu;
    Real _E = m_beamsData.getValue()[i]._E;
    Real _Iy = m_beamsData.getValue()[i]._Iy;
    Real _Iz = m_beamsData.getValue()[i]._Iz;
    Real _G = m_beamsData.getValue()[i]._G;
    Real _J = m_beamsData.getValue()[i]._J;
    Real L2 = (_L * _L);
    Real L3 = (L2 * _L);
    Real EIy = (_E * _Iy);
    Real EIz = (_E * _Iz);

    // Find shear-deformation parameters
    if (_A == 0)
    {
        phiy = 0.0;
        phiz = 0.0;
    }
    else
    {
        phiy = (24.0 * (1.0 + _nu) * _Iz / (_A * L2));
        phiz = (24.0 * (1.0 + _nu) * _Iy / (_A * L2));
    }

    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    Matrix12x12& k_loc = bd[i]._k_loc;

    // Define stiffness matrix 'k' in local coordinates
    k_loc.clear();
    k_loc[6][6] = k_loc[0][0] = _E*_A / _L;
    k_loc[7][7] = k_loc[1][1] = (12.0*EIz / (L3*(1.0 + phiy)));
    k_loc[8][8] = k_loc[2][2] = (12.0*EIy / (L3*(1.0 + phiz)));
    k_loc[9][9] = k_loc[3][3] = _G*_J / _L;
    k_loc[10][10] = k_loc[4][4] = ((4.0 + phiz)*EIy / (_L*(1.0 + phiz)));
    k_loc[11][11] = k_loc[5][5] = ((4.0 + phiy)*EIz / (_L*(1.0 + phiy)));

    k_loc[4][2] = (-6.0*EIy / (L2*(1.0 + phiz)));
    k_loc[5][1] = (6.0*EIz / (L2*(1.0 + phiy)));
    k_loc[6][0] = -k_loc[0][0];
    k_loc[7][1] = -k_loc[1][1];
    k_loc[7][5] = -k_loc[5][1];
    k_loc[8][2] = -k_loc[2][2];
    k_loc[8][4] = -k_loc[4][2];
    k_loc[9][3] = -k_loc[3][3];
    k_loc[10][2] = k_loc[4][2];
    k_loc[10][4] = ((2.0 - phiz)*EIy / (_L*(1.0 + phiz)));
    k_loc[10][8] = -k_loc[4][2];
    k_loc[11][1] = k_loc[5][1];
    k_loc[11][5] = ((2.0 - phiy)*EIz / (_L*(1.0 + phiy)));
    k_loc[11][7] = -k_loc[5][1];

    for (int i = 0; i <= 10; i++)
        for (int j = i + 1; j<12; j++)
            k_loc[i][j] = k_loc[j][i];

    m_beamsData.endEdit();
}

inline type::Quat<SReal> qDiff(type::Quat<SReal> a, const type::Quat<SReal>& b)
{
    if (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]<0)
    {
        a[0] = -a[0];
        a[1] = -a[1];
        a[2] = -a[2];
        a[3] = -a[3];
    }
    type::Quat<SReal> q = b.inverse() * a;
    return q;
}

/********************************************************************************/
/*                             VISITOR METHODS                                  */
/********************************************************************************/


template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & /*dataV*/ )
{
    VecDeriv& f = *(dataF.beginEdit());
    const VecCoord& p=dataX.getValue();
    f.resize(p.size());

    typename VecElement::const_iterator it;
    unsigned int i;

    for (it = m_indexedElements->begin(), i = 0; it != m_indexedElements->end(); ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];

        // The choice of computational method (elastic, plastic, or post-plastic)
        // is made in accumulateNonLinearForce
        accumulateNonLinearForce(f, p, i, a, b);
    }

    // Save the current positions as a record for the next time step.
    // This has to be done after the call to accumulateNonLinearForce
    // (otherwise the current position will be used instead in the 
    // computation)
    //TO DO: check if this is copy operator
    m_lastPos = p;

    dataF.endEdit();
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams *mparams, DataVecDeriv& datadF , const DataVecDeriv& datadX)
{
    VecDeriv& df = *(datadF.beginEdit());
    const VecDeriv& dx=datadX.getValue();
    Real kFactor = sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    df.resize(dx.size());

    typename VecElement::const_iterator it;
    unsigned int i = 0;
    for(it = m_indexedElements->begin() ; it != m_indexedElements->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];

        // The choice of the computational method (elastic, plastic, or post-plastic)
        // is made in applyNonLinearStiffness
        applyNonLinearStiffness(df, dx, i, a, b, kFactor);
    }

    datadF.endEdit();
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real k = sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    linearalgebra::BaseMatrix* mat = r.matrix;

    if (r)
    {
        unsigned int i=0;
        unsigned int &offset = r.offset;

        typename VecElement::const_iterator it;
        for(it = m_indexedElements->begin() ; it != m_indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];

            const MechanicalState beamMechanicalState = m_beamsData.getValue()[i]._beamMechanicalState;

            Quat<SReal>& q = beamQuat(i); //x[a].getOrientation();
            q.normalize();
            Mat<3, 3, Real> R,Rt;
            q.toMatrix(R);
            Rt.transpose(R);
            Matrix12x12 K;
            bool exploitSymmetry = d_useSymmetricAssembly.getValue();

            Matrix12x12 K0;
            if (beamMechanicalState == MechanicalState::PLASTIC)
                K0 = m_beamsData.getValue()[i]._Kt_loc;
            else
            {
                if (d_usePrecomputedStiffness.getValue())
                    K0 = m_beamsData.getValue()[i]._k_loc;
                else
                    K0 = m_beamsData.getValue()[i]._Ke_loc;
            }
               
            if (exploitSymmetry) {
                for (int x1 = 0; x1<12; x1 += 3) {
                    for (int y1 = x1; y1<12; y1 += 3)
                    {
                        Mat<3, 3, Real> m;
                        K0.getsub(x1, y1, m);
                        m = R*m*Rt;

                        for (int i = 0; i<3; i++)
                            for (int j = 0; j<3; j++)
                            {
                                K.elems[i + x1][j + y1] += m[i][j];
                                K.elems[j + y1][i + x1] += m[i][j];
                            }
                        if (x1 == y1)
                            for (int i = 0; i<3; i++)
                                for (int j = 0; j<3; j++)
                                    K.elems[i + x1][j + y1] *= double(0.5);
                    }
                }
            } // end if (exploitSymmetry)
            else
            {
                for (int x1 = 0; x1<12; x1 += 3) {
                    for (int y1 = 0; y1<12; y1 += 3)
                    {
                        Mat<3, 3, Real> m;
                        K0.getsub(x1, y1, m);
                        m = R*m*Rt;
                        K.setsub(x1, y1, m);
                    }
                }
            }

            int index[12];
            for (int x1 = 0; x1<6; x1++)
                index[x1] = offset + a * 6 + x1;
            for (int x1 = 0; x1<6; x1++)
                index[6 + x1] = offset + b * 6 + x1;
            for (int x1 = 0; x1<12; ++x1)
                for (int y1 = 0; y1<12; ++y1)
                    mat->add(index[x1], index[y1], -K(x1, y1)*k);

            //TO DO: m_beamsData.endEdit(); consecutive to the call to beamQuat

        } // end for m_indexedElements
    } // end if (r)
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    std::vector<Vec3> centrelinePoints;
    std::vector<Vec3> gaussPoints;
    std::vector<RGBAColor> colours;

    for (unsigned int i=0; i<m_indexedElements->size(); ++i)
        drawElement(i, gaussPoints, centrelinePoints, colours, x);

    vparams->drawTool()->setPolygonMode(2, true);
    vparams->drawTool()->setLightingEnabled(true);
    vparams->drawTool()->drawPoints(gaussPoints, 3, colours);
    vparams->drawTool()->drawLines(centrelinePoints, 1.0, RGBAColor(0.24f, 0.72f, 0.96f, 1.0f));
    vparams->drawTool()->setLightingEnabled(false);
    vparams->drawTool()->setPolygonMode(0, false);
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::drawElement(int i, std::vector< Vec3 > &gaussPoints,
                                                 std::vector< Vec3 > &centrelinePoints,
                                                 std::vector<RGBAColor> &colours,
                                                 const VecCoord& x)
{
    Index a = (*m_indexedElements)[i][0];
    Index b = (*m_indexedElements)[i][1];

    Vec3 pa, pb;
    pa = x[a].getCenter();
    pb = x[b].getCenter();

    const Quat<SReal>& q = beamQuat(i);

    //***** Gauss points *****//

    //Compute current displacement

    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    beamQuat(i) = x[a].getOrientation();
    beamQuat(i).normalize();

    m_beamsData.endEdit();

    Vec3 u, P1P2, P1P2_0;
    // local displacement
    Matrix12x1 disp;

    // translations //
    P1P2_0 = x0[b].getCenter() - x0[a].getCenter();
    P1P2_0 = x0[a].getOrientation().inverseRotate(P1P2_0);
    P1P2 = x[b].getCenter() - x[a].getCenter();
    P1P2 = x[a].getOrientation().inverseRotate(P1P2);
    u = P1P2 - P1P2_0;

    disp[0] = 0.0; 	disp[1] = 0.0; 	disp[2] = 0.0;
    disp[6] = u[0]; disp[7] = u[1]; disp[8] = u[2];

    // rotations //
    type::Quat<SReal> dQ0, dQ;

    dQ0 = qDiff(x0[b].getOrientation(), x0[a].getOrientation());
    dQ = qDiff(x[b].getOrientation(), x[a].getOrientation());

    dQ0.normalize();
    dQ.normalize();

    type::Quat<SReal> tmpQ = qDiff(dQ, dQ0);
    tmpQ.normalize();

    u = tmpQ.quatToRotationVector();

    disp[3] = 0.0; 	disp[4] = 0.0; 	disp[5] = 0.0;
    disp[9] = u[0]; disp[10] = u[1]; disp[11] = u[2];

    //Compute the positions of the Gauss points
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    Matrix3x12 N;
    const Vec<27, MechanicalState>& pointMechanicalState = m_beamsData.getValue()[i]._pointMechanicalState;
    int gaussPointIt = 0; //incremented in the lambda function to iterate over Gauss points

    LambdaType computeGaussCoordinates = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        //Shape function
        N = m_beamsData.getValue()[i]._N[gaussPointIt];
        Mat<3, 1, Real> u = N*disp;

        type::Vec3d beamVec = {u[0][0]+u1, u[1][0]+u2, u[2][0]+u3};
        type::Vec3d gp = pa + q.rotate(beamVec);
        gaussPoints.push_back(gp);

        if (pointMechanicalState[gaussPointIt] == MechanicalState::ELASTIC)
            colours.push_back({1.0f,0.015f,0.015f,1.0f}); //RED
        else if (pointMechanicalState[gaussPointIt] == MechanicalState::PLASTIC)
            colours.push_back({0.051f,0.15f,0.64f,1.0f}); //BLUE
        else
            colours.push_back({0.078f,0.41f,0.078f,1.0f}); //GREEN

        gaussPointIt++; //next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeGaussCoordinates);

    //****** Centreline ******//
    int nbSeg = m_beamsData.getValue()[i]._nbCentrelineSeg; //number of segments descretising the centreline

    centrelinePoints.push_back(pa);

    Matrix3x12 drawN;
    const double L = m_beamsData.getValue()[i]._L;
    for (int drawPointIt = 0; drawPointIt < nbSeg - 1; drawPointIt++)
    {
        //Shape function of the centreline point
        drawN = m_beamsData.getValue()[i]._drawN[drawPointIt];
        Mat<3, 1, Real> u = drawN*disp;

        type::Vec3d beamVec = {u[0][0] + (drawPointIt +1)*(L/nbSeg), u[1][0], u[2][0]};
        type::Vec3d clp = pa + q.rotate(beamVec);
        centrelinePoints.push_back(clp); //First time as the end of the former segment
        centrelinePoints.push_back(clp); //Second time as the beginning of the next segment
    }

    centrelinePoints.push_back(pb);
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    if (!onlyVisible) return;


    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::min();
    Real maxBBox[3] = { min_real,min_real,min_real };
    Real minBBox[3] = { max_real,max_real,max_real };


    const size_t npoints = this->mstate->getSize();
    const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    for (size_t i = 0; i<npoints; i++)
    {
        const Vec3 &pt = p[i].getCenter();

        for (int c = 0; c<3; c++)
        {
            if (pt[c] > maxBBox[c]) maxBBox[c] = pt[c];
            if (pt[c] < minBBox[c]) minBBox[c] = pt[c];
        }
    }

    this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox, maxBBox));

}


/*****************************************************************************/
/*                        PLASTIC IMPLEMENTATION                             */
/*****************************************************************************/


/***************************** Virtual Displacement **************************/

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeVDStiffness(int i, Index, Index)
{
    Real _L = m_beamsData.getValue()[i]._L;
    Real _yDim = m_beamsData.getValue()[i]._yDim;
    Real _zDim = m_beamsData.getValue()[i]._zDim;

    const double E = m_beamsData.getValue()[i]._E;
    const double nu = m_beamsData.getValue()[i]._nu;

    const Matrix6x6& C = m_beamsData.getValue()[i]._materialBehaviour;
    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    Matrix12x12& Ke_loc = bd[i]._Ke_loc;
    Ke_loc.clear();

    // Reduced integration
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    // Setting variables for the reduced intergation process defined in quadrature.h

    Matrix6x12 Be;
    // Stress matrix, to be integrated
    Matrix12x12 stiffness = Matrix12x12();

    int gaussPointIterator = 0; //incremented in the lambda function to iterate over Gauss points

    LambdaType computeStressMatrix = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        Be = m_beamsData.getValue()[i]._BeMatrices[gaussPointIterator];

        stiffness += (w1*w2*w3)*beTCBeMult(Be.transposed(), C, nu, E);

        gaussPointIterator++; //next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeStressMatrix);

    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
        {
            Ke_loc[i][j] = stiffness(i, j);
        }

    m_beamsData.endEdit();
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeMaterialBehaviour(int i, Index a, Index b)
{

    Real E = m_beamsData.getValue()[i]._E; // Young's modulus
    Real nu = m_beamsData.getValue()[i]._nu; // Poisson ratio

    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());

    Matrix6x6& C = bd[i]._materialBehaviour;
    // Material behaviour matrix, here: Hooke's law
    //TO DO: handle incompressible materials (with nu = 0.5)
    C(0, 0) = C(1, 1) = C(2, 2) = 1 - nu;
    C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = nu;
    C(0, 3) = C(0, 4) = C(0, 5) = 0;
    C(1, 3) = C(1, 4) = C(1, 5) = 0;
    C(2, 3) = C(2, 4) = C(2, 5) = 0;
    C(3, 0) = C(3, 1) = C(3, 2) = C(3, 4) = C(3, 5) = 0;
    C(4, 0) = C(4, 1) = C(4, 2) = C(4, 3) = C(4, 5) = 0;
    C(5, 0) = C(5, 1) = C(5, 2) = C(5, 3) = C(5, 4) = 0;
    C(3, 3) = C(4, 4) = C(5, 5) = 1 - 2*nu;
    C *= E / ( (1 + nu) * (1 - 2*nu) );

    m_beamsData.endEdit();
}

template< class DataTypes>
bool BeamPlasticFEMForceField<DataTypes>::goToPlastic(const VoigtTensor2 &stressTensor,
                                                 const double yieldStress,
                                                 const bool verbose /*=FALSE*/)
{
    if (verbose)
    {
        std::cout.precision(17);
        std::cout << vonMisesYield(stressTensor, yieldStress) << std::scientific << " "; //DEBUG
    }
    // Plasticity occurs if Von Mises function is >= 0
    return vonMisesYield(stressTensor, yieldStress) > m_stressComparisonThreshold;
}

template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::deviatoricStress(const VoigtTensor2 &stressTensor) -> VoigtTensor2
{
    // Returns the deviatoric stress from a given stress tensor in Voigt notation

    VoigtTensor2 deviatoricStress = stressTensor;
    double mean = (stressTensor[0][0] + stressTensor[1][0] + stressTensor[2][0]) / 3.0;
    for (int i = 0; i < 3; i++)
        deviatoricStress[i][0] -= mean;

    return deviatoricStress;
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::equivalentStress(const VoigtTensor2 &stressTensor)
{
    double res = 0.0;
    double sigmaX = stressTensor[0][0];
    double sigmaY = stressTensor[1][0];
    double sigmaZ = stressTensor[2][0];
    double sigmaYZ = stressTensor[3][0];
    double sigmaZX = stressTensor[4][0];
    double sigmaXY = stressTensor[5][0];

    double aux1 = 0.5*((sigmaX - sigmaY)*(sigmaX - sigmaY) + (sigmaY - sigmaZ)*(sigmaY - sigmaZ) + (sigmaZ - sigmaX)*(sigmaZ - sigmaX));
    double aux2 = 3.0*(sigmaYZ*sigmaYZ + sigmaZX*sigmaZX + sigmaXY*sigmaXY);

    res = helper::rsqrt(aux1 + aux2);
    return res;
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::vonMisesYield(const VoigtTensor2 &stressTensor,
                                                          const double yieldStress)
{
    double eqStress = equivalentStress(stressTensor);
    return eqStress - yieldStress;
}

template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::vonMisesGradient(const VoigtTensor2 &stressTensor) -> VoigtTensor2
{
    // NB: this gradient represent the normal to the yield surface
    // in case the Von Mises yield criterion is used.
    // /!\ the Norm of the gradient is sqrt(3/2): it has to be multiplied
    // by sqrt(2/3) to give the unit normal to the yield surface

    VoigtTensor2 gradient = VoigtTensor2();

    if (equalsZero(sofa::type::scalarProduct(stressTensor, stressTensor)))
        return gradient; //TO DO: is that correct ?

    double sigmaX = stressTensor[0][0];
    double sigmaY = stressTensor[1][0];
    double sigmaZ = stressTensor[2][0];
    double sigmaYZ = stressTensor[3][0];
    double sigmaZX = stressTensor[4][0];
    double sigmaXY = stressTensor[5][0];

    gradient[0] = 2 * sigmaX - sigmaY - sigmaZ;
    gradient[1] = 2 * sigmaY - sigmaZ - sigmaX;
    gradient[2] = 2 * sigmaZ - sigmaX - sigmaY;
    gradient[3] = 3 * sigmaYZ;
    gradient[4] = 3 * sigmaZX;
    gradient[5] = 3 * sigmaXY;

    double sigmaEq = equivalentStress(stressTensor);
    gradient *= 1 / (2 * sigmaEq);

    return gradient;
}

template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::vonMisesHessian(const VoigtTensor2 &stressTensor,
                                                          const double yieldStress) -> VectTensor4
{
    VectTensor4 hessian = VectTensor4();

    if (equalsZero(sofa::type::scalarProduct(stressTensor, stressTensor)))
        return hessian; //TO DO: is that correct ?

    //Order 1 terms
    double sigmaXX = stressTensor[0][0];
    double sigmaYY = stressTensor[1][0];
    double sigmaZZ = stressTensor[2][0];
    double sigmaYZ = stressTensor[3][0];
    double sigmaZX = stressTensor[4][0];
    double sigmaXY = stressTensor[5][0];

    double auxX = 2 * sigmaXX - sigmaYY - sigmaZZ;
    double auxY = 2 * sigmaYY - sigmaZZ - sigmaXX;
    double auxZ = 2 * sigmaZZ - sigmaXX - sigmaYY;

    //Order 2 terms
    double sX2 = sigmaXX*sigmaXX;
    double sY2 = sigmaYY*sigmaYY;
    double sZ2 = sigmaZZ*sigmaZZ;
    double sYsZ = sigmaYY*sigmaZZ;
    double sZsX = sigmaZZ*sigmaXX;
    double sXsY = sigmaXX*sigmaYY;

    double sYZ2 = sigmaYZ*sigmaYZ;
    double sZX2 = sigmaZX*sigmaZX;
    double sXY2 = sigmaXY*sigmaXY;

    //Others
    double sigmaE = vonMisesYield(stressTensor, yieldStress) + yieldStress;
    double sigmaE3 = sigmaE*sigmaE*sigmaE;
    double invSigmaE = 1 / sigmaE;

    //1st row
    hessian(0, 0) = invSigmaE - (auxX*auxX / (4 * sigmaE3));
    hessian(0, 1) = -( 3*sigmaXY*auxX / (4*sigmaE3) );
    hessian(0, 2) = -( 3*sigmaZX*auxX / (4*sigmaE3) );
    hessian(0, 3) = -( 3*sigmaXY*auxX / (4*sigmaE3) );
    hessian(0, 4) = -0.5*invSigmaE - ( (-2*sX2 - 2*sY2 + sZ2 - sYsZ - sZsX + 5*sXsY) / (4*sigmaE3) );
    hessian(0, 5) = -( 3*sigmaYZ*auxX / (4*sigmaE3) );
    hessian(0, 6) = -( 3*sigmaZX*auxX / (4*sigmaE3) );
    hessian(0, 7) = -( 3*sigmaYZ*auxX / (4*sigmaE3) );
    hessian(0, 8) = -0.5*invSigmaE - ( (-2*sX2 + sY2 - 2*sZ2 - sYsZ + 5*sZsX - sXsY) / (4*sigmaE3) );
    
    //2nd row
    hessian(1, 0) = -( 3*sigmaXY*auxX / (4*sigmaE3) );
    hessian(1, 1) = (3.0 / (2.0*sigmaE)) - (9.0*sXY2 / (4.0*sigmaE3));
    hessian(1, 2) = -( 9.0*sigmaXY*sigmaZX / (4.0*sigmaE3) );
    hessian(1, 3) = -( 9.0*sigmaXY*sigmaXY / (4.0*sigmaE3) );
    hessian(1, 4) = -( 3*sigmaXY*auxY / (4*sigmaE3) );
    hessian(1, 5) = -( 9.0*sigmaXY*sigmaYZ / (4.0*sigmaE3) );
    hessian(1, 6) = -( 9.0*sigmaXY*sigmaZX / (4.0*sigmaE3) );
    hessian(1, 7) = -( 9.0*sigmaXY*sigmaYZ / (4.0*sigmaE3) );
    hessian(1, 8) = -( 3*sigmaXY*auxZ / (4*sigmaE3) );

    //3rd row
    hessian(2, 0) = -( 3*sigmaZX*auxX / (4*sigmaE3) );
    hessian(2, 1) = -( 9.0*sigmaZX*sigmaXY / (4.0*sigmaE3) );
    hessian(2, 2) = (3.0 / (2.0*sigmaE)) - (9.0*sZX2 / (4.0*sigmaE3));
    hessian(2, 3) = -( 9.0*sigmaZX*sigmaXY / (4.0*sigmaE3) );
    hessian(2, 4) = -( 3*sigmaZX*auxY / (4*sigmaE3) );
    hessian(2, 5) = -( 9.0*sigmaZX*sigmaYZ / (4.0*sigmaE3) );
    hessian(2, 6) = -( 9.0*sigmaZX*sigmaZX / (4.0*sigmaE3) );
    hessian(2, 7) = -( 9.0*sigmaZX*sigmaYZ / (4.0*sigmaE3) );
    hessian(2, 8) = -( 3*sigmaZX*auxZ / (4*sigmaE3) );

    //4th row
    hessian(3, 0) = -( 3*sigmaXY*auxX / (4*sigmaE3) );
    hessian(3, 1) = -( 9.0*sigmaXY*sigmaXY / (4.0*sigmaE3) );
    hessian(3, 2) = -( 9.0*sigmaXY*sigmaZX / (4.0*sigmaE3) );
    hessian(3, 3) = (3.0 / (2.0*sigmaE)) - (9.0*sXY2 / (4.0*sigmaE3));
    hessian(3, 4) = -( 3*sigmaXY*auxY / (4*sigmaE3) );
    hessian(3, 5) = -( 9.0*sigmaXY*sigmaYZ / (4.0*sigmaE3) );
    hessian(3, 6) = -( 9.0*sigmaXY*sigmaZX / (4.0*sigmaE3) );
    hessian(3, 7) = -( 9.0*sigmaXY*sigmaYZ / (4.0*sigmaE3) );
    hessian(3, 8) = -( 3*sigmaXY*auxZ / (4*sigmaE3) );

    //5th row
    hessian(4, 0) = hessian(0, 4);
    hessian(4, 1) = -( 3*sigmaXY*auxY / (4*sigmaE3) );
    hessian(4, 2) = -( 3*sigmaZX*auxY / (4*sigmaE3) );
    hessian(4, 3) = -( 3*sigmaXY*auxY / (4*sigmaE3) );
    hessian(4, 4) = invSigmaE - (auxY*auxY / (4*sigmaE3));
    hessian(4, 5) = -( 3*sigmaYZ*auxY / (4*sigmaE3) );
    hessian(4, 6) = -( 3*sigmaZX*auxY / (4*sigmaE3) );
    hessian(4, 7) = -( 3*sigmaYZ*auxY / (4*sigmaE3) );
    hessian(4, 8) = -0.5*invSigmaE - ( (sX2 - 2*sY2 - 2*sZ2 + 5*sYsZ - sZsX - sXsY) / (4*sigmaE3) );

    //6th row
    hessian(5, 0) = -( 3*sigmaYZ*auxX / (4*sigmaE3) );
    hessian(5, 1) = -( 9.0*sigmaYZ*sigmaXY / (4.0*sigmaE3) );
    hessian(5, 2) = -( 9.0*sigmaYZ*sigmaZX / (4.0*sigmaE3) );
    hessian(5, 3) = -( 9.0*sigmaYZ*sigmaXY / (4.0*sigmaE3) );
    hessian(5, 4) = -( 3*sigmaYZ*auxY / (4*sigmaE3) );
    hessian(5, 5) = (3.0 / (2.0*sigmaE)) - (9.0*sYZ2 / (4.0*sigmaE3));
    hessian(5, 6) = -( 9.0*sigmaYZ*sigmaZX / (4.0*sigmaE3) );
    hessian(5, 7) = -( 9.0*sigmaYZ*sigmaYZ / (4.0*sigmaE3) );
    hessian(5, 8) = -( 3*sigmaYZ*auxZ / (4*sigmaE3) );

    //7th row
    hessian(6, 0) = -( 3*sigmaZX*auxX / (4*sigmaE3) );
    hessian(6, 1) = -( 9.0*sigmaZX*sigmaXY / (4.0*sigmaE3) );
    hessian(6, 2) = -( 9.0*sigmaZX*sigmaZX / (4.0*sigmaE3) );
    hessian(6, 3) = -( 9.0*sigmaZX*sigmaXY / (4.0*sigmaE3) );
    hessian(6, 4) = -( 3*sigmaZX*auxY / (4*sigmaE3) );
    hessian(6, 5) = -( 9.0*sigmaZX*sigmaYZ / (4.0*sigmaE3) );
    hessian(6, 6) = (3.0 / (2.0*sigmaE)) - (9.0*sZX2 / (4.0*sigmaE3));
    hessian(6, 7) = -( 9.0*sigmaZX*sigmaYZ / (4.0*sigmaE3) );
    hessian(6, 8) = -( 3*sigmaZX*auxZ / (4*sigmaE3) );

    //8th row
    hessian(7, 0) = -( 3*sigmaYZ*auxX / (4*sigmaE3) );
    hessian(7, 1) = -( 9.0*sigmaYZ*sigmaXY / (4.0*sigmaE3) );
    hessian(7, 2) = -( 9.0*sigmaYZ*sigmaZX / (4.0*sigmaE3) );
    hessian(7, 3) = -( 9.0*sigmaYZ*sigmaXY / (4.0*sigmaE3) );
    hessian(7, 4) = -( 3*sigmaYZ*auxY / (4*sigmaE3) );
    hessian(7, 5) = -( 9.0*sigmaYZ*sigmaYZ / (4.0*sigmaE3) );
    hessian(7, 6) = -( 9.0*sigmaYZ*sigmaZX / (4.0*sigmaE3) );
    hessian(7, 7) = (3.0 / (2.0*sigmaE)) - (9.0*sYZ2 / (4.0*sigmaE3));
    hessian(7, 8) = -( 3*sigmaYZ*auxZ / (4*sigmaE3) );

    //9th row
    hessian(8, 0) = -0.5*invSigmaE - ( (-2*sX2 + sY2 - 2*sZ2 - sYsZ + 5*sZsX - sXsY) / (4*sigmaE3) );
    hessian(8, 1) = -( 3*sigmaXY*auxZ / (4*sigmaE3) );
    hessian(8, 2) = -( 3*sigmaZX*auxZ / (4*sigmaE3) );
    hessian(8, 3) = -( 3*sigmaZX*auxZ / (4*sigmaE3) );
    hessian(8, 4) = -0.5*invSigmaE - ( (sX2 - 2*sY2 - 2*sZ2 + 5*sYsZ - sZsX - sXsY) / (4*sigmaE3) );
    hessian(8, 5) = -( 3*sigmaYZ*auxZ / (4*sigmaE3) );
    hessian(8, 6) = -( 3*sigmaZX*auxZ / (4*sigmaE3) );
    hessian(8, 7) = -( 3*sigmaYZ*auxZ / (4*sigmaE3) );
    hessian(8, 8) = invSigmaE - (auxZ*auxZ / (4*sigmaE3));

    return hessian;
}



/***************************** Alternative methods for DEBUG **************************/

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::vectEquivalentStress(const VectTensor2 &stressTensor)
{
    // Compute the equivalent stress using a vector notation

    double eqStress = 0.0;
    double sigmaXX = stressTensor[0][0];
    double sigmaXY = stressTensor[1][0];
    double sigmaXZ = stressTensor[2][0];
    double sigmaYX = stressTensor[3][0];
    double sigmaYY = stressTensor[4][0];
    double sigmaYZ = stressTensor[5][0];
    double sigmaZX = stressTensor[6][0];
    double sigmaZY = stressTensor[7][0];
    double sigmaZZ = stressTensor[8][0];

    double aux1 = 0.5*((sigmaXX - sigmaYY)*(sigmaXX - sigmaYY) + (sigmaYY - sigmaZZ)*(sigmaYY - sigmaZZ) + (sigmaZZ - sigmaXX)*(sigmaZZ - sigmaXX));
    double aux2 = (3.0 / 2.0)*(sigmaXY*sigmaXY + sigmaYX*sigmaYX + sigmaXZ*sigmaXZ + sigmaZX*sigmaZX + sigmaYZ*sigmaYZ + sigmaZY*sigmaZY);

    eqStress = helper::rsqrt(aux1 + aux2);
    return eqStress;
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::devEquivalentStress(const VoigtTensor2 &stressTensor)
{
    // Compute the equivalent stress from the expression
    // of the deviatoric stress tensor

    VoigtTensor2 devStress = deviatoricStress(stressTensor);
    VectTensor2 vectDevStress = voigtToVect2(devStress);

    Mat<1, 1, Real> squaredNormMat = vectDevStress.transposed()*vectDevStress;
    return helper::rsqrt(3.0 * squaredNormMat[0][0] / 2.0);
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::devVonMisesYield(const VoigtTensor2 &stressTensor,
                                                        const double yieldStress)
{
    double devEqStress = devEquivalentStress(stressTensor);
    return devEqStress - yieldStress;
}


template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::vectVonMisesYield(const VectTensor2 &stressTensor,
                                                         const double yieldStress)
{
    double eqStress = vectEquivalentStress(stressTensor);
    return eqStress - yieldStress;
}


template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::vectVonMisesGradient(const VectTensor2 &stressTensor) -> VectTensor2
{
    // Computation of Von Mises yield function gradient,
    // in vector notation

    VectTensor2 gradient = VectTensor2();

    if (equalsZero(sofa::type::scalarProduct(stressTensor, stressTensor)))
        return gradient; //TO DO: is that correct ?

    double sigmaXX = stressTensor[0][0];
    double sigmaXY = stressTensor[1][0];
    double sigmaXZ = stressTensor[2][0];
    double sigmaYX = stressTensor[3][0];
    double sigmaYY = stressTensor[4][0];
    double sigmaYZ = stressTensor[5][0];
    double sigmaZX = stressTensor[6][0];
    double sigmaZY = stressTensor[7][0];
    double sigmaZZ = stressTensor[8][0];

    gradient[0][0] = 2 * sigmaXX - sigmaYY - sigmaZZ;
    gradient[1][0] = 3 * sigmaXY;
    gradient[2][0] = 3 * sigmaXZ;
    gradient[3][0] = 3 * sigmaYX;
    gradient[4][0] = 2 * sigmaYY - sigmaZZ - sigmaXX;
    gradient[5][0] = 3 * sigmaYZ;
    gradient[6][0] = 3 * sigmaZX;
    gradient[7][0] = 3 * sigmaZY;
    gradient[8][0] = 2 * sigmaZZ - sigmaXX - sigmaYY;

    double sigmaEq = vectEquivalentStress(stressTensor);
    gradient *= 1 / (2 * sigmaEq);

    return gradient;
}


template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::devVonMisesGradient(const VoigtTensor2 &stressTensor) -> VoigtTensor2
{
    // Computation of the gradient of the Von Mises function, at stressTensor,
    // using the expression of the deviatoric stress tensor

    VoigtTensor2 gradient = VoigtTensor2();

    if (equalsZero(sofa::type::scalarProduct(stressTensor, stressTensor)))
        return gradient; //TO DO: is that correct ?

    VoigtTensor2 devStress = deviatoricStress(stressTensor);
    double devEqStress = devEquivalentStress(stressTensor);

    gradient = (3.0 / (2.0*devEqStress))*devStress;

    return gradient;
}



/*************************** Voigt notation correction ***********************/

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::voigtDotProduct(const VoigtTensor2 &t1, const VoigtTensor2 &t2)
{
    // This method provides a correct implementation of the dot product for 2nd-order tensors represented
    // with Voigt notation. As the tensors are symmetric, then can be represented with only 6 elements,
    // but all non-diagonal elements have to be taken into account for a dot product.

    double res = 0.0;
    res += t1[0] * t2[0] + t1[1] * t2[1] + t1[2] * t2[2];      //diagonal elements
    res += 2 * (t1[3] * t2[3] + t1[4] * t2[4] + t1[5] * t2[5]);  //non-diagonal elements
    return res;
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::voigtTensorNorm(const VoigtTensor2 &t)
{
    // This method provides a correct implementation of the norm for 2nd-order tensors represented
    // with Voigt notation. The unrepresented elements are taken into account in the
    // voigtDotProduct method

    return helper::rsqrt(voigtDotProduct(t, t));
}

template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::beTTensor2Mult(const Matrix12x6 &BeT,
                                                         const VoigtTensor2 &T) -> Matrix12x1
{
    // In Voigt notation, 3 rows in Be (i.e. 3 columns in Be^T) are missing.
    // These rows correspond to the 3 symmetrical non-diagonal elements of the
    // tensors, which are not expressed in Voigt notation.
    // We have to add the contribution of these rows in the computation BeT*Tensor.

    Matrix12x1 res = Matrix12x1();

    res += BeT*T; // contribution of the 6 first columns
                  // We compute the contribution of the 3 missing columns.
                  // This can be achieved with block computation.

    Matrix12x3 additionalColumns = Matrix12x3();
//    additionalColumns.block<12, 1>(0, 0) = BeT.block<12, 1>(0, 3); // T_yz
//    additionalColumns.block<12, 1>(0, 1) = BeT.block<12, 1>(0, 4); // T_zx
//    additionalColumns.block<12, 1>(0, 2) = BeT.block<12, 1>(0, 5); // T_xy
    Matrix12x1 subMat;
    BeT.getsub(0, 3, subMat);
    additionalColumns.setsub(0, 0, subMat); // T_yz
    BeT.getsub(0, 4, subMat);
    additionalColumns.setsub(0, 1, subMat); // T_zx
    BeT.getsub(0, 5, subMat);
    additionalColumns.setsub(0, 2, subMat); // T_xy



    Mat<3, 1, Real> additionalTensorElements = Mat<3, 1, Real>();
    additionalTensorElements[0] = T[3]; // T_yz
    additionalTensorElements[1] = T[4]; // T_zx
    additionalTensorElements[2] = T[5]; // T_xy

    res += additionalColumns*additionalTensorElements;
    return res;
}

template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::beTCBeMult(const Matrix12x6 &BeT,
                                                     const VoigtTensor4 &C,
                                                     const double nu, const double E) -> Matrix12x12
{
    // In Voigt notation, 3 rows in Be (i.e. 3 columns in Be^T) are missing.
    // These rows correspond to the 3 symmetrical non-diagonal elements of the
    // tensors, which are not expressed in Voigt notation.
    // We have to add the contribution of these rows in the computation BeT*C*Be.

    // First part of the computation in Voigt Notation
    Matrix12x12 res = Matrix12x12();
    res += BeT*C*(BeT.transposed()); // contribution of the 6 first columns
                                     // Second part : contribution of the missing rows in Be
    Matrix12x3 leftTerm = Matrix12x3();
//    leftTerm.block<12, 1>(0, 0) = BeT.block<12, 1>(0, 3); // S_3N^T
//    leftTerm.block<12, 1>(0, 1) = BeT.block<12, 1>(0, 4); // S_4N^T
//    leftTerm.block<12, 1>(0, 2) = BeT.block<12, 1>(0, 5); // S_5N^T
    Matrix12x1 subMat;
    BeT.getsub(0, 3, subMat);
    leftTerm.setsub(0, 0, subMat); // S_3N^T
    BeT.getsub(0, 4, subMat);
    leftTerm.setsub(0, 1, subMat); // S_4N^T
    BeT.getsub(0, 5, subMat);
    leftTerm.setsub(0, 2, subMat); // S_5N^T

    Matrix3x12 rightTerm = leftTerm.transposed();
    rightTerm *= E / (1 + nu);

    res += leftTerm*rightTerm;

    return res;
}



/********************* Stress computation - general methods ******************/

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::accumulateNonLinearForce(VecDeriv& f,
                                                              const VecCoord& x,
                                                              int i,
                                                              Index a, Index b)
{
    //Concrete implementation of addForce
    //Computes f += Kx, assuming that this component is linear
    //All non-linearity has to be handled here (including plasticity)

    Matrix12x1 fint = Matrix12x1();

    if (d_isPerfectlyPlastic.getValue())
        computeForceWithPerfectPlasticity(fint, x, i, a, b);
    else
        computeForceWithHardening(fint, x, i, a, b);


    //Passes the contribution to the global system
    Vec12 force;

    for (int i = 0; i < 12; i++)
        force[i] = fint[i][0];

    Vec3 fa1 = x[a].getOrientation().rotate(Vec3(force[0], force[1], force[2]));
    Vec3 fa2 = x[a].getOrientation().rotate(Vec3(force[3], force[4], force[5]));

    Vec3 fb1 = x[a].getOrientation().rotate(Vec3(force[6], force[7], force[8]));
    Vec3 fb2 = x[a].getOrientation().rotate(Vec3(force[9], force[10], force[11]));

    f[a] += Deriv(-fa1, -fa2);
    f[b] += Deriv(-fb1, -fb2);
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::applyNonLinearStiffness(VecDeriv& df,
                                                             const VecDeriv& dx,
                                                             int i,
                                                             Index a, Index b, double fact)
{
    //Concrete implementation of addDForce
    //Computes df += Kdx, the expression of K depending on the mechanical state
    //of the beam element

    //Computes displacement increment, from last system solution
    Vec12 local_depl;
    Vec3 u;
    Quat<SReal>& q = beamQuat(i); //x[a].getOrientation();
    q.normalize();

    u = q.inverseRotate(getVCenter(dx[a]));
    local_depl[0] = u[0];
    local_depl[1] = u[1];
    local_depl[2] = u[2];

    u = q.inverseRotate(getVOrientation(dx[a]));
    local_depl[3] = u[0];
    local_depl[4] = u[1];
    local_depl[5] = u[2];

    u = q.inverseRotate(getVCenter(dx[b]));
    local_depl[6] = u[0];
    local_depl[7] = u[1];
    local_depl[8] = u[2];

    u = q.inverseRotate(getVOrientation(dx[b]));
    local_depl[9] = u[0];
    local_depl[10] = u[1];
    local_depl[11] = u[2];

    m_beamsData.endEdit(); // consecutive to the call to beamQuat

    const MechanicalState beamMechanicalState = m_beamsData.getValue()[i]._beamMechanicalState;
    Vec12 local_dforce;

    // The stiffness matrix we use depends on the mechanical state of the beam element

    if (beamMechanicalState == MechanicalState::PLASTIC)
        local_dforce = m_beamsData.getValue()[i]._Kt_loc * local_depl;
    else
    {
        if (d_usePrecomputedStiffness.getValue())
            // this computation can be optimised: (we know that half of "depl" is null)
            local_dforce = m_beamsData.getValue()[i]._k_loc * local_depl;
        else
            local_dforce = m_beamsData.getValue()[i]._Ke_loc * local_depl;
    }

    Vec3 fa1 = q.rotate(Vec3(local_dforce[0], local_dforce[1], local_dforce[2]));
    Vec3 fa2 = q.rotate(Vec3(local_dforce[3], local_dforce[4], local_dforce[5]));
    Vec3 fb1 = q.rotate(Vec3(local_dforce[6], local_dforce[7], local_dforce[8]));
    Vec3 fb2 = q.rotate(Vec3(local_dforce[9], local_dforce[10], local_dforce[11]));

    df[a] += Deriv(-fa1, -fa2) * fact;
    df[b] += Deriv(-fb1, -fb2) * fact;
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::updateTangentStiffness(int i,
                                                            Index a,
                                                            Index b)
{
    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    Matrix12x12& Kt_loc = bd[i]._Kt_loc;
    const Matrix6x6& C = bd[i]._materialBehaviour;
    const double E = bd[i]._E;
    const double nu = bd[i]._nu;
    Vec<27, MechanicalState>& pointMechanicalState = bd[i]._pointMechanicalState;

    // Reduced integration
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    // Setting variables for the reduced intergation process defined in quadrature.h
    Matrix6x12 Be;
    VoigtTensor4 Cep = VoigtTensor4(); //plastic behaviour tensor
    VoigtTensor2 gradient;

    //Auxiliary matrices
    VoigtTensor2 Cgrad;
    Mat<1, 6, Real> gradTC;

    //Result matrix
    Matrix12x12 tangentStiffness = Matrix12x12();

    VoigtTensor2 currentStressPoint;
    int gaussPointIt = 0;

    // Stress matrix, to be integrated
    LambdaType computeTangentStiffness = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        currentStressPoint = m_prevStresses[i][gaussPointIt];

        // Plastic modulus
        double plasticModulus = computeConstPlasticModulus();

        // Be
        Be = bd[i]._BeMatrices[gaussPointIt];

        // Cep
        gradient = vonMisesGradient(currentStressPoint);

        if (!d_useConsistentTangentOperator.getValue())
        {
            if (equalsZero(sofa::type::scalarProduct(gradient, gradient)) || pointMechanicalState[gaussPointIt] != MechanicalState::PLASTIC)
                Cep = C; //TO DO: is that correct ?
            else
            {
                if (!d_isPerfectlyPlastic.getValue())
                {
                    VoigtTensor2 normal = helper::rsqrt(2.0 / 3.0)*gradient;
                    VectTensor2 vectNormal = voigtToVect2(normal);
                    VectTensor4 vectC = voigtToVect4(C);
                    VectTensor4 vectCep = VectTensor4();

                    VectTensor2 CN = vectC*vectNormal;
                    // NtC = (NC)t because of C symmetry
                    Mat<1, 1, Real> scalarMatrix = vectNormal.transposed()*CN;
                    vectCep = vectC - ( CN*CN.transposed() ) / (scalarMatrix[0][0] + (2.0 / 3.0)*plasticModulus);

                    Cep = vectToVoigt4(vectCep);
                }
                else
                {
                    VoigtTensor2 normal = helper::rsqrt(2.0 / 3.0)*gradient;
                    VectTensor2 vectNormal = voigtToVect2(normal);
                    VectTensor4 vectC = voigtToVect4(C);
                    VectTensor4 vectCep = VectTensor4();

                    VectTensor2 CN = vectC*vectNormal;
                    Mat<1, 1, Real> scalarMatrix = vectNormal.transposed()*CN;
                    vectCep = vectC - ( CN*CN.transposed() ) / scalarMatrix[0][0];

                    Cep = vectToVoigt4(vectCep);
                }
            }
        }
        else // d_useConsistentTangentOperator = true
        {
            if (pointMechanicalState[gaussPointIt] != MechanicalState::PLASTIC)
                Cep = C; //TO DO: is that correct ?
            else
            {
                if (!d_isPerfectlyPlastic.getValue())
                {
                    //Computation of matrix H as in Studies in anisotropic plasticity with reference to the Hill criterion, De Borst and Feenstra, 1990
                    VectTensor4 H = VectTensor4();
                    VectTensor4 I = VectTensor4::Identity();
                    VoigtTensor2 elasticPredictor = m_elasticPredictors[i][gaussPointIt];

                    VectTensor2 vectGradient = voigtToVect2(gradient);
                    VectTensor4 vectC = voigtToVect4(C);
                    double yieldStress = m_beamsData.getValue()[i]._localYieldStresses[gaussPointIt];
                    Mat<1, 1, Real> scalarMatrix = vectGradient.transposed()*vectC*vectGradient;
                    double DeltaLambda = vonMisesYield(elasticPredictor, yieldStress) / scalarMatrix[0][0];
                    VectTensor4 vectHessian = vonMisesHessian(elasticPredictor, yieldStress);

                    VectTensor4 M = (I + DeltaLambda*vectC*vectHessian);
                    // M is symmetric positive definite, we perform Cholesky decomposition to invert it
                    // M is symmetric positive definite, we perform Cholesky decomposition to invert it
                    // For this, we convert the matrix using Eigen
                    // TO DO: is there a more efficient way?
                    Eigen::Matrix<double, 9, 9> eigenM = Eigen::Matrix<double, 9, 9>::Zero();
                    for (int i=0; i < 9; i++)
                        for (int j=0; j < 9; j++)
                            eigenM(i, j) = M[i][j];
                    Eigen::Matrix<double, 9, 9> eigenI = Eigen::Matrix<double, 9, 9>::Identity();
                    Eigen::Matrix<double, 9, 9> invertedEigenM = eigenM.llt().solve(eigenI);
                    VectTensor4 invertedM = VectTensor4();
                    for (int i=0; i < 9; i++)
                        for (int j=0; j < 9; j++)
                            invertedM[i][j] = invertedEigenM(i, j);
                    H = invertedM*vectC;

                    //Computation of Cep
                    if (equalsZero(sofa::type::scalarProduct(gradient, gradient)))
                        Cep = vectToVoigt4(H);
                    else
                    {
                        VectTensor4 consistentCep = VectTensor4();
                        Mat<1, 9, Real> gradTH = vectGradient.transposed()*H;
                        Mat<1, 1, Real> scalarMatrix = gradTH*vectGradient;
                        consistentCep = H - (H*vectGradient*gradTH) / (scalarMatrix[0][0] + plasticModulus);
                        Cep = vectToVoigt4(consistentCep);
                    }
                }
                else
                {
                    //Computation of matrix H as in Studies in anisotropic plasticity with reference to the Hill criterion, De Borst and Feenstra, 1990
                    VectTensor4 H = VectTensor4();
                    VectTensor4 I = VectTensor4::Identity();
                    VoigtTensor2 elasticPredictor = m_elasticPredictors[i][gaussPointIt];

                    VectTensor2 vectGradient = voigtToVect2(gradient);
                    VectTensor4 vectC = voigtToVect4(C);
                    double yieldStress = m_beamsData.getValue()[i]._localYieldStresses[gaussPointIt];
                    // NB: the gradient is the same between the elastic predictor and the new stress
                    Mat<1, 1, Real> scalarMatrix = vectGradient.transposed()*vectC*vectGradient;
                    double DeltaLambda = vonMisesYield(elasticPredictor, yieldStress) / scalarMatrix[0][0];
                    VectTensor4 vectHessian = vonMisesHessian(elasticPredictor, yieldStress);

                    VectTensor4 M = (I + DeltaLambda*vectC*vectHessian);
                    // M is symmetric positive definite, we perform Cholesky decomposition to invert it
                    // For this, we convert the matrix using Eigen
                    // TO DO: is there a more efficient way?
                    Eigen::Matrix<double, 9, 9> eigenM = Eigen::Matrix<double, 9, 9>::Zero();
                    for (int i=0; i < 9; i++)
                        for (int j=0; j < 9; j++)
                            eigenM(i, j) = M[i][j];
                    Eigen::Matrix<double, 9, 9> eigenI = Eigen::Matrix<double, 9, 9>::Identity();
                    Eigen::Matrix<double, 9, 9> invertedEigenM = eigenM.llt().solve(eigenI);
                    VectTensor4 invertedM = VectTensor4();
                    for (int i=0; i < 9; i++)
                        for (int j=0; j < 9; j++)
                            invertedM[i][j] = invertedEigenM(i, j);
                    H = invertedM*vectC;

                    //Computation of Cep
                    if (equalsZero(sofa::type::scalarProduct(gradient, gradient)))
                        Cep = vectToVoigt4(H);
                    else
                    {
                        VectTensor4 consistentCep = VectTensor4();
                        Mat<1, 9, Real> gradTH = vectGradient.transposed()*H;
                        Mat<1, 1, Real> scalarMatrix = gradTH*vectGradient;
                        consistentCep = H - (H*vectGradient*gradTH) / scalarMatrix[0][0];
                        Cep = vectToVoigt4(consistentCep);
                    }
                } // end if d_isPerfectlyPlastic = true
            } // end if pointMechanicalState[gaussPointIt] == MechanicalState::PLASTIC
        } // end if d_useConsistentTangentOperator = true

        tangentStiffness += (w1*w2*w3)*beTCBeMult(Be.transposed(), Cep, nu, E);

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = bd[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeTangentStiffness);

    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            Kt_loc[i][j] = tangentStiffness(i, j);

    m_beamsData.endEdit();
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeLocalDisplacement(const VecCoord& x, Vec12 &localDisp,
                                                              int i, Index a, Index b)
{
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    beamQuat(i) = x[a].getOrientation();
    beamQuat(i).normalize();

    m_beamsData.endEdit();

    Vec3 u, P1P2, P1P2_0;

    // translations //
    P1P2_0 = x0[b].getCenter() - x0[a].getCenter();
    P1P2_0 = x0[a].getOrientation().inverseRotate(P1P2_0);
    P1P2 = x[b].getCenter() - x[a].getCenter();
    P1P2 = x[a].getOrientation().inverseRotate(P1P2);
    u = P1P2 - P1P2_0;

    localDisp[0] = 0.0; localDisp[1] = 0.0; localDisp[2] = 0.0;
    localDisp[6] = u[0]; localDisp[7] = u[1]; localDisp[8] = u[2];

    // rotations //
    Quat<SReal> dQ0, dQ;

    // dQ = QA.i * QB ou dQ = QB * QA.i() ??
    dQ0 = qDiff(x0[b].getOrientation(), x0[a].getOrientation()); // x0[a].getOrientation().inverse() * x0[b].getOrientation();
    dQ = qDiff(x[b].getOrientation(), x[a].getOrientation()); // x[a].getOrientation().inverse() * x[b].getOrientation();
                                                              //u = dQ.toEulerVector() - dQ0.toEulerVector(); // Consider to use quatToRotationVector instead of toEulerVector to have the rotation vector

    dQ0.normalize();
    dQ.normalize();

    Quat<SReal> tmpQ = qDiff(dQ, dQ0);
    tmpQ.normalize();

    u = tmpQ.quatToRotationVector(); //dQ.quatToRotationVector() - dQ0.quatToRotationVector();  // Use of quatToRotationVector instead of toEulerVector:
                                     // this is done to keep the old behavior (before the
                                     // correction of the toEulerVector  function). If the
                                     // purpose was to obtain the Eulerian vector and not the
                                     // rotation vector please use the following line instead
                                     //u = tmpQ.toEulerVector(); //dQ.toEulerVector() - dQ0.toEulerVector();

    localDisp[3] = 0.0; localDisp[4] = 0.0; localDisp[5] = 0.0;
    localDisp[9] = u[0]; localDisp[10] = u[1]; localDisp[11] = u[2];
}


template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeDisplacementIncrement(const VecCoord& pos, const VecCoord& lastPos, Vec12 &currentDisp,
                                                                  Vec12 &lastDisp, Vec12 &dispIncrement, int i, Index a, Index b)
{
    // ***** Displacement for current position *****//

    computeLocalDisplacement(pos, currentDisp, i, a, b);

    // ***** Displacement for last position *****//

    computeLocalDisplacement(lastPos, lastDisp, i, a, b);

    // ***** Displacement increment *****//

    dispIncrement = currentDisp - lastDisp;
}



/********************* Stress computation - auxiliary methods ******************/


template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::computePlasticModulusFromStress(const VoigtTensor2 &stressState)
{
    const double eqStress = equivalentStress(stressState);
    double plasticModulus = m_ConstitutiveLaw->getTangentModulusFromStress(eqStress); //TO DO: check for definition of H' in Hugues 1984
    return plasticModulus;
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::computePlasticModulusFromStrain(int index, int gaussPointId)
{
    const Real effPlasticStrain = m_beamsData.getValue()[index]._effectivePlasticStrains[gaussPointId];
    double plasticModulus = m_ConstitutiveLaw->getTangentModulusFromStrain(effPlasticStrain); //TO DO: check for definition of H' in Hugues 1984
    return plasticModulus;
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::computeConstPlasticModulus()
{
    return 34628588874.0; // TO DO: look for proper constant, from Hugues 1984 definition of H'
}

template<class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::voigtToVect2(const VoigtTensor2 &voigtTensor) -> VectTensor2
{
    // This function aims at vectorising a second-order tensor taking into account
    // all 9 elements of the tensor. The result is thus a 9x1 vector (contrarily
    // to the Voigt vector representation which is only 6x1).
    // By default, we order the elements line by line, so that:
    // res[0] = T_11, res[1] = T_12, res[2] = T_13
    // res[3] = T_12, res[4] = T_22, res[5] = T_23
    // res[6] = T_13, res[7] = T_23, res[8] = T_33
    // where T is the second-order (9 element) tensor in matrix form.

    VectTensor2 res = VectTensor2();

    res[0] = voigtTensor[0];
    res[4] = voigtTensor[1];
    res[8] = voigtTensor[2];
    res[5] = res[7] = voigtTensor[3];
    res[2] = res[6] = voigtTensor[4];
    res[1] = res[3] = voigtTensor[5];

    return res;
}


template<class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::voigtToVect4(const VoigtTensor4 &voigtTensor) -> VectTensor4
{
    // This function aims at vectorising a fourth-order tensor taking into account
    // all 81 elements of the tensor. The result is thus a 9x9 matrix (contrarily
    // to the Voigt vector representation which is only 6x6).
    // By default, we order the elements according to the convention we used for
    // second-order tensors, so that:
    // res(0, *) = (T_1111, T_1112, T_1113, T_1112, T_1122, T_1123, T_1113, T_1123, T_1133)
    // res(1, *) = (T_1211, T_1212, T_1213, T_1212, T_1222, T_1223, T_1213, T_1223, T_1233)
    // etc.
    // where T is the fourth-order (81 element) tensor in matrix form.

    VectTensor4 res = VectTensor4();

    // Row 0
    res(0, 0) = voigtTensor(0, 0);
    res(0, 4) = voigtTensor(0, 1);
    res(0, 8) = voigtTensor(0, 2);
    res(0, 1) = res(0, 3) = voigtTensor(0, 5) / 2;
    res(0, 2) = res(0, 6) = voigtTensor(0, 4) / 2;
    res(0, 5) = res(0, 7) = voigtTensor(0, 3) / 2;

    // Row 4
    res(4, 0) = voigtTensor(1, 0);
    res(4, 4) = voigtTensor(1, 1);
    res(4, 8) = voigtTensor(1, 2);
    res(4, 1) = res(4, 3) = voigtTensor(1, 5) / 2;
    res(4, 2) = res(4, 6) = voigtTensor(1, 4) / 2;
    res(4, 5) = res(4, 7) = voigtTensor(1, 3) / 2;

    // Row 8
    res(8, 0) = voigtTensor(2, 0);
    res(8, 4) = voigtTensor(2, 1);
    res(8, 8) = voigtTensor(2, 2);
    res(8, 1) = res(8, 3) = voigtTensor(2, 5) / 2;
    res(8, 2) = res(8, 6) = voigtTensor(2, 4) / 2;
    res(8, 5) = res(8, 7) = voigtTensor(2, 3) / 2;

    // Rows 1 and 3
    res(3, 0) = res(1, 0) = voigtTensor(5, 0);
    res(3, 4) = res(1, 4) = voigtTensor(5, 1);
    res(3, 8) = res(1, 8) = voigtTensor(5, 2);
    res(3, 1) = res(3, 3) = res(1, 1) = res(1, 3) = voigtTensor(5, 5) / 2;
    res(3, 2) = res(3, 6) = res(1, 2) = res(1, 6) = voigtTensor(5, 4) / 2;
    res(3, 5) = res(3, 7) = res(1, 5) = res(1, 7) = voigtTensor(5, 3) / 2;

    // Rows 2 and 6
    res(6, 0) = res(2, 0) = voigtTensor(4, 0);
    res(6, 4) = res(2, 4) = voigtTensor(4, 1);
    res(6, 8) = res(2, 8) = voigtTensor(4, 2);
    res(6, 1) = res(6, 3) = res(2, 1) = res(2, 3) = voigtTensor(4, 5) / 2;
    res(6, 2) = res(6, 6) = res(2, 2) = res(2, 6) = voigtTensor(4, 4) / 2;
    res(6, 5) = res(6, 7) = res(2, 5) = res(2, 7) = voigtTensor(4, 3) / 2;

    // Rows 5 and 7
    res(5, 0) = res(7, 0) = voigtTensor(3, 0);
    res(5, 4) = res(7, 4) = voigtTensor(3, 1);
    res(5, 8) = res(7, 8) = voigtTensor(3, 2);
    res(5, 1) = res(5, 3) = res(7, 1) = res(7, 3) = voigtTensor(3, 5) / 2;
    res(5, 2) = res(5, 6) = res(7, 2) = res(7, 6) = voigtTensor(3, 4) / 2;
    res(5, 5) = res(5, 7) = res(7, 5) = res(7, 7) = voigtTensor(3, 3) / 2;

    return res;
}


template<class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::vectToVoigt2(const VectTensor2 &vectTensor) -> VoigtTensor2
{
    // This function aims at reducing the expression of a second-order tensor
    // using Voigt notation. The tensor is initially expressed in vector form
    // as:
    // res[0] = T_11, res[1] = T_12, res[2] = T_13
    // res[3] = T_12, res[4] = T_22, res[5] = T_23
    // res[6] = T_13, res[7] = T_23, res[8] = T_33
    // where T is the second-order (9 element) tensor in matrix form.

    VoigtTensor2 res = VoigtTensor2();

    res[0] = vectTensor[0];
    res[1] = vectTensor[4];
    res[2] = vectTensor[8];
    res[3] = vectTensor[5];
    res[4] = vectTensor[2];
    res[5] = vectTensor[1];

    return res;
}


template<class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::vectToVoigt4(const VectTensor4 &vectTensor) -> Matrix6x6
{
    // This function aims at reducing the expression of a fourth-order tensor
    // using Voigt notation. The tensor is initially expressed in vector form
    // as:
    // res(0, *) = (T_1111, T_1112, T_1113, T_1112, T_1122, T_1123, T_1113, T_1123, T_1133)
    // res(1, *) = (T_1211, T_1212, T_1213, T_1212, T_1222, T_1223, T_1213, T_1223, T_1233)
    // etc.
    // where T is the fourth-order (81 element) tensor in matrix form.

    VoigtTensor4 res = VoigtTensor4();

    // 1st row
    res(0, 0) = vectTensor(0, 0);
    res(0, 1) = vectTensor(0, 4);
    res(0, 2) = vectTensor(0, 8);
    res(0, 3) = vectTensor(0, 5);
    res(0, 4) = vectTensor(0, 2);
    res(0, 5) = vectTensor(0, 1);

    // 2nd row
    res(1, 0) = vectTensor(4, 0);
    res(1, 1) = vectTensor(4, 4);
    res(1, 2) = vectTensor(4, 8);
    res(1, 3) = vectTensor(4, 5);
    res(1, 4) = vectTensor(4, 2);
    res(1, 5) = vectTensor(4, 1);

    // 3rd row
    res(2, 0) = vectTensor(8, 0);
    res(2, 1) = vectTensor(8, 4);
    res(2, 2) = vectTensor(8, 8);
    res(2, 3) = vectTensor(8, 5);
    res(2, 4) = vectTensor(8, 2);
    res(2, 5) = vectTensor(8, 1);

    // 4th row
    res(3, 0) = vectTensor(5, 0);
    res(3, 1) = vectTensor(5, 4);
    res(3, 2) = vectTensor(5, 8);
    res(3, 3) = vectTensor(5, 5);
    res(3, 4) = vectTensor(5, 2);
    res(3, 5) = vectTensor(5, 1);

    // 5th row
    res(4, 0) = vectTensor(2, 0);
    res(4, 1) = vectTensor(2, 4);
    res(4, 2) = vectTensor(2, 8);
    res(4, 3) = vectTensor(2, 5);
    res(4, 4) = vectTensor(2, 2);

    // 6th row
    res(5, 0) = vectTensor(1, 0);
    res(5, 1) = vectTensor(1, 4);
    res(5, 2) = vectTensor(1, 8);
    res(5, 3) = vectTensor(1, 5);
    res(5, 4) = vectTensor(1, 2);
    res(5, 5) = vectTensor(1, 1);

    return res;
}


//---------- Incremental force computation for perfect plasticity ----------//

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeForceWithPerfectPlasticity(Matrix12x1& internalForces,
                                                                            const VecCoord& x, int index, Index a, Index b)
{
    // Computes displacement increment, from last system solution
    Vec12 currentDisp;
    Vec12 lastDisp;
    Vec12 dispIncrement;
    computeDisplacementIncrement(x, m_lastPos, currentDisp, lastDisp, dispIncrement, index, a, b);

    // Converts to Matrix data structure
    Matrix12x1 displacementIncrement;
    for (int k = 0; k < 12; k++)
        displacementIncrement(k) = dispIncrement[k];

    //All the rest of the force computation is made inside of the lambda function
    //as the stress and strain are computed for each Gauss point

    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    const Matrix6x6& C = m_beamsData.getValue()[index]._materialBehaviour;
    Matrix6x12 Be;

    VoigtTensor2 initialStressPoint = VoigtTensor2();
    VoigtTensor2 strainIncrement = VoigtTensor2();
    VoigtTensor2 newStressPoint = VoigtTensor2();

    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    Vec<27, MechanicalState>& pointMechanicalState = bd[index]._pointMechanicalState;
    bool isPlasticBeam = false;
    int gaussPointIt = 0;

    // Computation of the new stress point, through material point iterations as in Krabbenhoft lecture notes

    // This function is to be called if the last stress point corresponded to elastic deformation
    LambdaType computeStress = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];
        MechanicalState &mechanicalState = pointMechanicalState[gaussPointIt];

        //Strain
        strainIncrement = Be*displacementIncrement;

        //Stress
        initialStressPoint = m_prevStresses[index][gaussPointIt];
        computePerfectPlasticStressIncrement(index, gaussPointIt, initialStressPoint, newStressPoint,
            strainIncrement, mechanicalState);

        isPlasticBeam = isPlasticBeam || (mechanicalState == MechanicalState::PLASTIC);

        m_prevStresses[index][gaussPointIt] = newStressPoint;

        internalForces += (w1*w2*w3)*beTTensor2Mult(Be.transposed(), newStressPoint);

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[index]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeStress);

    // Updates the beam mechanical state information
    if (!isPlasticBeam)
    {
        MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
        beamMechanicalState = MechanicalState::POSTPLASTIC;
    }
    m_beamsData.endEdit();

    //Update the tangent stiffness matrix with the new computed stresses
    //This matrix will then be used in addDForce and addKToMatrix methods
    updateTangentStiffness(index, a, b);
}


template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computePerfectPlasticStressIncrement(int index,
                                                                               int gaussPointIt,
                                                                               const VoigtTensor2& lastStress,
                                                                               VoigtTensor2& newStressPoint,
                                                                               const VoigtTensor2& strainIncrement,
                                                                               MechanicalState& pointMechanicalState)
{
    /** Material point iterations **/
    //NB: we consider that the yield function and the plastic flow are equal (f=g)
    //    This corresponds to an associative flow rule (for plasticity)

    const Matrix6x6& C = m_beamsData.getValue()[index]._materialBehaviour; //Matrix D in Krabbenhoft's

    /***************************************************/
    /*  Radial return in perfect plasticity - Hugues   */
    /***************************************************/
    {
        //First we compute the trial stress, taking into account the back stress
        // (i.e. the centre of the yield surface)

        VoigtTensor2 elasticIncrement = C * strainIncrement;
        VoigtTensor2 trialStress = lastStress + elasticIncrement;

        if (d_useConsistentTangentOperator.getValue())
            m_elasticPredictors[index][gaussPointIt] = trialStress;

        type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());

        Vec<27, Real>& localYieldStresses = bd[index]._localYieldStresses;
        Real& yieldStress = localYieldStresses[gaussPointIt];

        VoigtTensor2 devTrialStress = deviatoricStress(trialStress);

        double A = voigtDotProduct(devTrialStress, devTrialStress);
        double R = helper::rsqrt(2.0 / 3) * yieldStress;
        const double R2 = R * R;

        if (A <= R2) //TO DO: proper comparison
        {
            // The Gauss point is in elastic state: the back stress and yield stress
            // remain constant, and the new stress is equal to the trial stress.
            newStressPoint = trialStress;

            // If the Gauss point was initially plastic, we update its mechanical state
            if (pointMechanicalState == MechanicalState::PLASTIC)
                pointMechanicalState = MechanicalState::POSTPLASTIC;
        }
        else
        {
            // If the Gauss point was initially elastic, we update its mechanical state
            if (pointMechanicalState == MechanicalState::POSTPLASTIC || pointMechanicalState == MechanicalState::ELASTIC)
                pointMechanicalState = MechanicalState::PLASTIC;

            // We then compute the new stress

            /**** Litterature implementation ****/
            // Ref: Theoretical foundation for large scale computations for nonlinear
            // material behaviour, Hugues (et al) 1984

            double meanStress = (1.0 / 3) * (trialStress[0][0] + trialStress[1][0] + trialStress[2][0]);

            // Computing the new stress
            VoigtTensor2 voigtIdentityTensor = VoigtTensor2();
            voigtIdentityTensor[0] = 1;
            voigtIdentityTensor[1] = 1;
            voigtIdentityTensor[2] = 1;

            newStressPoint = (R / helper::rsqrt(A)) * devTrialStress + meanStress * voigtIdentityTensor;

            // Updating the plastic strain
            VoigtTensor2 yieldNormal = helper::rsqrt(3.0 / 2) * (1.0 / equivalentStress(trialStress)) * devTrialStress;

            double lambda = voigtDotProduct(yieldNormal, strainIncrement);

            VoigtTensor2 plasticStrainIncrement = lambda * yieldNormal;
            type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
            Vec<27, VoigtTensor2>& plasticStrainHistory = bd[index]._plasticStrainHistory;
            plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;
            m_beamsData.endEdit();
        }

    }
}


//---------- Incremental force computation for linear mixed (isotropic and kinematic) hardening ----------//


template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeForceWithHardening(Matrix12x1 &internalForces,
                                                                    const VecCoord& x, int index, Index a, Index b)
{
    // Computes displacement increment, from last system solution
    Vec12 currentDisp;
    Vec12 lastDisp;
    Vec12 dispIncrement;
    computeDisplacementIncrement(x, m_lastPos, currentDisp, lastDisp, dispIncrement, index, a, b);

    // Converts to Matrix data structure
    Matrix12x1 displacementIncrement;
    for (int k = 0; k < 12; k++)
        displacementIncrement(k) = dispIncrement[k];

    //All the rest of the force computation is made inside of the lambda function
    //as the stress and strain are computed for each Gauss point

    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    const Matrix6x6& C = m_beamsData.getValue()[index]._materialBehaviour;
    Matrix6x12 Be;

    VoigtTensor2 initialStressPoint = VoigtTensor2();
    VoigtTensor2 strainIncrement = VoigtTensor2();
    VoigtTensor2 newStressPoint = VoigtTensor2();

    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    Vec<27, MechanicalState>& pointMechanicalState = bd[index]._pointMechanicalState;
    bool isPlasticBeam = false;
    int gaussPointIt = 0;

    // Computation of the new stress point, through material point iterations as in Krabbenhoft lecture notes

    // This function is to be called if the last stress point corresponded to elastic deformation
    LambdaType computeStress = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];
        MechanicalState &mechanicalState = pointMechanicalState[gaussPointIt];

        //Strain
        strainIncrement = Be*displacementIncrement;

        //Stress
        initialStressPoint = m_prevStresses[index][gaussPointIt];
        computeHardeningStressIncrement(index, gaussPointIt, initialStressPoint, newStressPoint,
            strainIncrement, mechanicalState);

        isPlasticBeam = isPlasticBeam || (mechanicalState == MechanicalState::PLASTIC);

        m_prevStresses[index][gaussPointIt] = newStressPoint;

        internalForces += (w1*w2*w3)*beTTensor2Mult(Be.transposed(), newStressPoint);

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[index]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeStress);

    // Updates the beam mechanical state information
    if (!isPlasticBeam)
    {
        MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
        beamMechanicalState = MechanicalState::POSTPLASTIC;
    }
    m_beamsData.endEdit();

    //Update the tangent stiffness matrix with the new computed stresses
    //This matrix will then be used in addDForce and addKToMatrix methods
    if (isPlasticBeam)
        updateTangentStiffness(index, a, b);
}


template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeHardeningStressIncrement(int index,
                                                                     int gaussPointIt,
                                                                     const VoigtTensor2 &lastStress,
                                                                     VoigtTensor2 &newStressPoint,
                                                                     const VoigtTensor2 &strainIncrement,
                                                                     MechanicalState &pointMechanicalState)
{
    /** Material point iterations **/
    //NB: we consider that the yield function and the plastic flow are equal (f=g)
    //    This corresponds to an associative flow rule (for plasticity)

    const Matrix6x6& C = m_beamsData.getValue()[index]._materialBehaviour; //Matrix D in Krabbenhoft's

    /***************************************************/
    /*      Radial return with hardening - Hugues      */
    /***************************************************/

    //First we compute the trial stress, taking into account the back stress
    // (i.e. the centre of the yield surface)

    VoigtTensor2 elasticIncrement = C*strainIncrement;
    VoigtTensor2 trialStress = lastStress + elasticIncrement;

    if (d_useConsistentTangentOperator.getValue())
        m_elasticPredictors[index][gaussPointIt] = trialStress;

    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());

    Vec<27, VoigtTensor2> &backStresses = bd[index]._backStresses;
    VoigtTensor2 &backStress = backStresses[gaussPointIt];

    Vec<27, Real> &localYieldStresses = bd[index]._localYieldStresses;
    Real &yieldStress = localYieldStresses[gaussPointIt];

    if (!goToPlastic(trialStress - backStress, yieldStress))
    {
        // The Gauss point is in elastic state: the back stress and yield stress
        // remain constant, and the new stress is equal to the trial stress.
        newStressPoint = trialStress;

        // If the Gauss point was initially plastic, we update its mechanical state
        if (pointMechanicalState == MechanicalState::PLASTIC)
            pointMechanicalState = MechanicalState::POSTPLASTIC;
    }
    else
    {
        // If the Gauss point was initially elastic, we update its mechanical state
        if (pointMechanicalState == MechanicalState::POSTPLASTIC || pointMechanicalState == MechanicalState::ELASTIC)
            pointMechanicalState = MechanicalState::PLASTIC;

        VoigtTensor2 shiftedTrialStress = trialStress - backStress;
        VoigtTensor2 xiTrial = deviatoricStress(shiftedTrialStress);

        // Normal at the end of the time step
        double xiTrialNorm = voigtTensorNorm(xiTrial);
        VoigtTensor2 finalN = xiTrial / xiTrialNorm;

        const double beta = 0.5; // Indicates the proportion of Kinematic vs isotropic hardening. beta=0 <=> kinematic, beta=1 <=> isotropic

        const double E = m_beamsData.getValue()[index]._E;
        const double nu = m_beamsData.getValue()[index]._nu;
        const double mu = E / (2 * (1 + nu)); // Lame coefficient

        double H = computeConstPlasticModulus();

        // Computation of the plastic multiplier
        double plasticMultiplier = (xiTrialNorm - helper::rsqrt(2.0 / 3.0)*yieldStress) / ( mu*helper::rsqrt(6.0) * (1 + H / (3 * mu)));

        // Updating plastic variables
        newStressPoint = trialStress - helper::rsqrt(6.0)*mu*plasticMultiplier*finalN;

        yieldStress += beta*H*plasticMultiplier;

        backStress += helper::rsqrt(2.0 / 3.0)*(1 - beta)*H*plasticMultiplier*finalN;

        //type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit()); //Done in the beginning to modify the yield and back stresses
        Vec<27, VoigtTensor2> &plasticStrainHistory = bd[index]._plasticStrainHistory;
        VoigtTensor2 plasticStrainIncrement = helper::rsqrt(3.0/2.0)*plasticMultiplier*finalN;
        plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;
        //m_beamsData.endEdit();

        Vec<27, Real> &effectivePlasticStrain = bd[index]._effectivePlasticStrains;
        effectivePlasticStrain[gaussPointIt] += plasticMultiplier;

        m_beamsData.endEdit(); //end edit _backStresses, _localYieldStresses, _plasticStrainHistory, and _effectivePlasticStrains
    }
}


//---------- Gaussian quadrature integration methods ----------//

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::initialiseInterval(int beam, type::vector<Interval3>& integrationIntervals)
{
    if (d_sectionShape.getValue() == "rectangular")
    {
        Real L = m_beamsData.getValue()[beam]._L;
        Real Ly = d_ySection.getValue();
        Real Lz = d_zSection.getValue();

        // Integration interval definition for a local frame at the centre of the beam
        integrationIntervals.push_back(Interval3(-L / 2, L / 2, -Ly / 2, Ly / 2, -Lz / 2, Lz / 2));
    }
    else if (d_sectionShape.getValue() == "circular")
    {
        //TO DO: implement quadrature method for a disc and a hollow-disc cross section
        msg_error() << "Quadrature method for " << d_sectionShape.getValue()
            << " shape cross section has not been implemented yet. Methods for rectangular cross sections are available";
    }
    else
    {
        msg_error() << "Quadrature method for " << d_sectionShape.getValue()
            << " shape cross section has not been implemented yet. Methods for rectangular cross sections are available";
    }
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::initialiseGaussPoints(int beam, type::vector<beamGaussPoints>& gaussPoints, const Interval3& integrationInterval)
{
    //Gaussian nodes coordinates and weights for a 1D integration on [-1,1]
    const double sqrt3_5 = helper::rsqrt(3.0 / 5);
    Vec3 canonical3NodesCoordinates = { -sqrt3_5, 0, sqrt3_5 };
    Vec3 canonical3NodesWeights = { 5.0 / 9, 8.0 / 9, 5.0 / 9 };

    Real L = m_beamsData.getValue()[beam]._L;
    Real A = m_beamsData.getValue()[beam]._A;
    Real Iy = m_beamsData.getValue()[beam]._Iy;
    Real Iz = m_beamsData.getValue()[beam]._Iz;

    Real nu = m_beamsData.getValue()[beam]._nu;
    Real E = m_beamsData.getValue()[beam]._E;

    //Compute actual Gauss points coordinates and weights, with a 3D integration
    //NB: 3 loops because integration is in 3D, 3 iterations per loop because it's a 3 point integration
    unsigned int gaussPointIt = 0;
    for (unsigned int i = 0; i < 3; i++)
    {
        double x = canonical3NodesCoordinates[i];
        double w1 = canonical3NodesWeights[i];
        // Changing first coordinate and weight to adapt to the integration interval
        double a1 = integrationInterval.geta1();
        double b1 = integrationInterval.getb1();
        double xChanged = changeCoordinate(x, a1, b1);
        double w1Changed = changeWeight(w1, a1, b1);

        for (unsigned int j = 0; j < 3; j++)
        {
            double y = canonical3NodesCoordinates[j];
            double w2 = canonical3NodesWeights[j];
            // Changing second coordinate and weight to adapt to the integration interval
            double a2 = integrationInterval.geta2();
            double b2 = integrationInterval.getb2();
            double yChanged = changeCoordinate(y, a2, b2);
            double w2Changed = changeWeight(w2, a2, b2);

            for (unsigned int k = 0; k < 3; k++)
            {
                double z = canonical3NodesCoordinates[k];
                double w3 = canonical3NodesWeights[k];
                // Changing third coordinate and weight to adapt to the integration interval
                double a3 = integrationInterval.geta3();
                double b3 = integrationInterval.getb3();
                double zChanged = changeCoordinate(z, a3, b3);
                double w3Changed = changeWeight(w3, a3, b3);

                GaussPoint3 newGaussPoint = GaussPoint3(xChanged, yChanged, zChanged, w1Changed, w2Changed, w3Changed);
                newGaussPoint.setGradN(computeGradN(xChanged, yChanged, zChanged, L, A, Iy, Iz, E, nu));
                newGaussPoint.setNx(computeNx(xChanged, yChanged, zChanged, L, A, Iy, Iz, E, nu));
                newGaussPoint.setYieldStress(d_initialYieldStress.getValue());
                gaussPoints[beam][gaussPointIt] = newGaussPoint;
                gaussPointIt++;
            }
        }
    }
}

template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::computeNx(Real x, Real y, Real z, Real L, Real A, Real Iy, Real Iz,
                                                    Real E, Real nu, Real kappaY, Real kappaZ)->Matrix3x12
{
    Matrix3x12 Nx = Matrix3x12(); // Sets each element to 0
    Real xi = x / L;
    Real eta = y / L;
    Real zeta = z / L;

    Real xi2 = xi * xi;
    Real xi3 = xi * xi * xi;

    Real L2 = L * L;
    Real G = E / (2.0 * (1.0 + nu));

    Real phiY, phiZ;
    if (A == 0)
    {
        phiY = 0.0;
        phiZ = 0.0;
    }
    else
    {
        phiY = (12.0 * E * Iy / (kappaZ * G * A * L2));
        phiZ = (12.0 * E * Iz / (kappaY * G * A * L2));
    }
    Real phiYInv = ( 1 / (phiY - 1) );
    Real phiZInv = ( 1 / (1 + phiZ) );

    Nx(0, 0) = 1 - xi;
    Nx(0, 1) = 6 * phiZInv * (xi - xi2) * eta;
    Nx(0, 2) = 6 * phiYInv * (xi - xi2) * zeta;
    //Nx(0, 3) = 0;
    Nx(0, 4) = L * phiYInv * (1 - 4 * xi + 3 * xi2 + phiY * (1 - xi)) * zeta;
    Nx(0, 5) = - L * phiZInv * (1 - 4 * xi + 3 * xi2 + phiZ * (1 - xi)) * eta;
    Nx(0, 6) = xi;
    Nx(0, 7) = 6 * phiZInv * (-xi + xi2) * eta;
    Nx(0, 8) = 6 * phiYInv * (-xi + xi2) * zeta;
    //Nx(0, 9) = 0;
    Nx(0, 10) = L * phiYInv * (-2 * xi + 3 * xi2 + phiY * xi) * zeta;
    Nx(0, 11) = - L * phiZInv * (-2 * xi + 3 * xi2 + phiZ * xi) * eta;

    //Nx(1, 0) = 0;
    Nx(1, 1) = phiZInv * (1 - 3 * xi2 + 2 * xi3 + phiZ * (1 - xi));
    //Nx(1, 2) = 0;
    Nx(1, 3) = (xi - 1) * L * zeta;
    //Nx(1, 4) = 0;
    Nx(1, 5) = L * phiZInv * (xi - 2 * xi2 + xi3 + (phiZ / 2) * (xi - xi2));
    //Nx(1, 6) = 0;
    Nx(1, 7) = phiZInv * (3 * xi2 - 2 * xi3 + phiZ * xi);
    //Nx(1, 8) = 0;
    Nx(1, 9) = - L * xi * zeta;
    //Nx(1, 10) = 0;
    Nx(1, 11) = L * phiZInv * (-xi2 + xi3 - (phiZ / 2) * (xi - xi2));

    //Nx(2, 0) = 0;
    //Nx(2, 1) = 0;
    Nx(2, 2) = phiYInv * (1 - 3 * xi2 + 2 * xi3 + phiY * (1 - xi));
    Nx(2, 3) = (1 - xi) * L * eta;
    Nx(2, 4) = - L * phiYInv * (xi - 2 * xi2 + xi3 + (phiY / 2) * (xi - xi2));
    //Nx(2, 5) = 0;
    //Nx(2, 6) = 0;
    //Nx(2, 7) = 0;
    Nx(2, 8) = phiYInv * (3 * xi2 - 2 * xi3 + phiY * xi);
    Nx(2, 9) = L * xi * eta;
    Nx(2, 10) = - L * phiYInv * (-xi2 + xi3 - (phiY / 2) * (xi - xi2));
    //Nx(2, 11) = 0;

    return Nx;
}

template< class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::computeGradN(Real x, Real y, Real z, Real L, Real A, Real Iy, Real Iz,
                                                       Real E, Real nu, Real kappaY, Real kappaZ)->Matrix9x12
{
    Matrix9x12 gradN = Matrix9x12(); // Sets each element to 0
    Real xi = x / L;
    Real eta = y / L;
    Real zeta = z / L;

    Real L2 = L * L;
    Real G = E / (2.0 * (1.0 + nu));

    Real phiY, phiZ;
    if (A == 0)
    {
        phiY = 0.0;
        phiZ = 0.0;
    }
    else
    {
        phiY = (12.0 * E * Iy / (kappaZ * G * A * L2));
        phiZ = (12.0 * E * Iz / (kappaY * G * A * L2));
    }
    Real phiYInv = ( 1 / (phiY - 1) );
    Real phiZInv = ( 1 / (1 + phiZ) );

    //Row 0
    gradN(0, 0) = - 1 / L;
    gradN(0, 1) = - ( 12*phiZInv * xi * eta ) / L;
    gradN(0, 2) = ( 12*phiYInv * xi * zeta ) / L;
    // gradN(0, 3) = 0;
    gradN(0, 4) = - ( 1 + 6*phiYInv * xi ) * zeta;
    gradN(0, 5) = ( 1 - 6*phiZInv * xi) * eta;
    gradN(0, 6) = 1 / L;
    gradN(0, 7) = - gradN(0, 1);
    gradN(0, 8) = - gradN(0, 2);
    // gradN(0, 9) = 0;
    gradN(0, 10) = gradN(0, 4) + 2 * zeta;
    gradN(0, 11) = gradN(0, 5) - 2 * eta;

    //Rows 1 and 3
    gradN(1, 3) = zeta / 2;
    gradN(1, 9) = -zeta / 2;

    gradN(3, 3) = zeta / 2;
    gradN(3, 9) = -zeta / 2;

    //Rows 2 and 6
    gradN(2, 3) = -eta / 2;
    gradN(2, 9) = eta / 2;

    gradN(6, 3) = -eta / 2;
    gradN(6, 9) = eta / 2;

    //Rows 4, 5, 7, 8 are null

    return gradN;
}

template <class DataTypes>
template <typename LambdaType>
void BeamPlasticFEMForceField<DataTypes>::integrateBeam(beamGaussPoints& gaussPoints, LambdaType integrationFun)
{
    //Apply a generic (lambda) integration function to each Gauss point of a beam element
    for (unsigned int gp = 0; gp < gaussPoints.size(); gp++)
    {
        integrationFun(gaussPoints[gp]);
    }
}


/*****************************************************************************/
/*                              MISCELLANEOUS                                */
/*****************************************************************************/

template<class DataTypes>
Quat<SReal>& BeamPlasticFEMForceField<DataTypes>::beamQuat(int i)
{
    type::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    return bd[i].quat;
}

/**************************************************************************/


/*****************************************************************************/
/*                               GaussPoint3                                 */
/*****************************************************************************/

template <class DataTypes>
BeamPlasticFEMForceField<DataTypes>::GaussPoint3::GaussPoint3(Real x, Real y, Real z, Real w1, Real w2, Real w3)
{
    m_coordinates = { x, y, z };
    m_weights = { w1, w2, w3 };
    m_mechanicalState = MechanicalState::ELASTIC; //By default, before any deformation occurs
    m_prevStress = Vec9(); //By default, no deformation => 0 stress tensor
    m_backStress = Vec9(); //By default, no plastic deformation => back stress is 0
    m_yieldStress = 0; //Changed by initialiseGaussPoints, depends on the material
    m_plasticStrain = Vec9(); //By default, no plastic deformation => no history
    m_effectivePlasticStrain = 0; //By default, no plastic deformation => no history
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getNx() const -> const Matrix3x12&
{
    return m_Nx;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setNx(Matrix3x12 Nx)
{
    m_Nx = Nx;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getGradN() const -> const Matrix9x12&
{
    return m_gradN;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setGradN(Matrix9x12 gradN)
{
    m_gradN = gradN;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getMechanicalState() const -> const MechanicalState
{
    return m_mechanicalState;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setMechanicalState(MechanicalState newState)
{
    m_mechanicalState = newState;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getPrevStress() const -> const Vec9&
{
    return m_prevStress;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setPrevStress(Vec9 newStress)
{
    m_prevStress = newStress;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getCoord() const -> const Vec3&
{
    return m_coordinates;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setCoord(Vec3 coord)
{
    m_coordinates = coord;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getWeights() const -> const Vec3&
{
    return m_weights;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setWeights(Vec3 weights)
{
    m_weights = weights;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getBackStress() const -> const Vec9&
{
    return m_backStress;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setBackStress(Vec9 backStress)
{
    m_backStress = backStress;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getYieldStress() const ->const Real
{
    return m_yieldStress;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setYieldStress(Real yieldStress)
{
    m_yieldStress = yieldStress;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getPlasticStrain() const -> const Vec9&
{
    return m_plasticStrain;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setPlasticStrain(Vec9 plasticStrain)
{
    m_plasticStrain = plasticStrain;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::GaussPoint3::getEffectivePlasticStrain() const ->const Real
{
    return m_effectivePlasticStrain;
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::GaussPoint3::setEffectivePlasticStrain(Real effectivePlasticStrain)
{
    m_effectivePlasticStrain = effectivePlasticStrain;
}


/*****************************************************************************/
/*                                Interval3                                  */
/*****************************************************************************/

template <class DataTypes>
BeamPlasticFEMForceField<DataTypes>::Interval3::Interval3()
{
    //By default, integration is considered over [-1,1]*[-1,1]*[-1,1].
    m_a1 = m_a2 = m_a3 = -1;
    m_b1 = m_b2 = m_b3 = 1;
}

template <class DataTypes>
BeamPlasticFEMForceField<DataTypes>::Interval3::Interval3(Real a1, Real b1, Real a2, Real b2, Real a3, Real b3)
{
    m_a1 = a1;
    m_b1 = b1;
    m_a2 = a2;
    m_b2 = b2;
    m_a3 = a3;
    m_b3 = b3;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::Interval3::geta1() const -> Real
{
    return m_a1;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::Interval3::getb1() const -> Real
{
    return m_b1;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::Interval3::geta2() const -> Real
{
    return m_a2;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::Interval3::getb2() const -> Real
{
    return m_b2;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::Interval3::geta3() const -> Real
{
    return m_a3;
}

template <class DataTypes>
auto BeamPlasticFEMForceField<DataTypes>::Interval3::getb3() const -> Real
{
    return m_b3;
}

} // namespace sofa::plugin::beamplastic::component::forcefield::_beamplasticfemforcefield_
