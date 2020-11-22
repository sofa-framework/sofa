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
#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_INL

#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/GridTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/rmath.h>
#include <cassert>
#include <iostream>
#include <set>
#include <sofa/helper/system/gl.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>

#include "BeamPlasticFEMForceField.h"
#include "../StiffnessContainer.h"
#include "../PoissonContainer.h"
#include "RambergOsgood.h"

namespace sofa
{

namespace component
{

namespace forcefield
{

namespace _beamplasticfemforcefield_
{

template<class DataTypes>
BeamPlasticFEMForceField<DataTypes>::BeamPlasticFEMForceField()
    : m_beamsData(initData(&m_beamsData, "beamsData", "Internal element data"))
    , m_indexedElements(NULL)
    , d_poissonRatio(initData(&d_poissonRatio,(Real)0.3f,"poissonRatio","Potion Ratio"))
    , d_youngModulus(initData(&d_youngModulus, (Real)5000, "youngModulus", "Young Modulus"))
    , d_yieldStress(initData(&d_yieldStress,(Real)6.0e8,"yieldStress","yield stress"))
    , d_usePrecomputedStiffness(initData(&d_usePrecomputedStiffness, true, "usePrecomputedStiffness",
                                        "indicates if a precomputed elastic stiffness matrix is used, instead of being computed by reduced integration"))
    , d_useConsistentTangentOperator(initData(&d_useConsistentTangentOperator, false, "useConsistentTangentOperator",
        "indicates wether to use a consistent tangent operator in the computation of the plastic stiffness matrix"))
    , d_isPerfectlyPlastic(initData(&d_isPerfectlyPlastic, false, "isPerfectlyPlastic", "indicates wether the behaviour model is perfectly plastic"))
    , d_modelName(initData(&d_modelName, std::string("RambergOsgood"), "modelName", "the name of the 1D contitutive law model to be used in plastic deformation"))
    , d_zSection(initData(&d_zSection, (Real)0.2, "zSection", "length of the section in the z direction for rectangular beams"))
    , d_ySection(initData(&d_ySection, (Real)0.2, "ySection", "length of the section in the y direction for rectangular beams"))
    , d_useSymmetricAssembly(initData(&d_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , d_isTimoshenko(initData(&d_isTimoshenko,false,"isTimoshenko","implements a Timoshenko beam model"))
    , m_edgeHandler(NULL)
{
    m_edgeHandler = new BeamFFEdgeHandler(this, &m_beamsData);

    d_poissonRatio.setRequired(true);
    d_youngModulus.setReadOnly(true);
}

template<class DataTypes>
BeamPlasticFEMForceField<DataTypes>::BeamPlasticFEMForceField(Real poissonRatio, Real youngModulus, Real yieldStress, Real zSection,
                                                    Real ySection, bool useVD, bool isPlasticMuller, bool isTimoshenko,
                                                    bool isPlasticKrabbenhoft, bool isPerfectlyPlastic,
                                                    helper::vector<defaulttype::Quat> localOrientations)
    : m_beamsData(initData(&m_beamsData, "beamsData", "Internal element data"))
    , m_indexedElements(NULL)
    , d_poissonRatio(initData(&d_poissonRatio,(Real)poissonRatio,"poissonRatio","Potion Ratio"))
    , d_youngModulus(initData(&d_youngModulus,(Real)youngModulus,"youngModulus","Young Modulus"))
    , d_yieldStress(initData(&d_yieldStress, (Real)yieldStress, "yieldStress", "yield stress"))
    , d_usePrecomputedStiffness(initData(&d_usePrecomputedStiffness, true, "usePrecomputedStiffness",
                                        "indicates if a precomputed elastic stiffness matrix is used, instead of being computed by reduced integration"))
    , d_useConsistentTangentOperator(initData(&d_useConsistentTangentOperator, false, "useConsistentTangentOperator", 
                                             "indicates wether to use a consistent tangent operator in the computation of the plastic stiffness matrix"))
    , d_isPerfectlyPlastic(initData(&d_isPerfectlyPlastic, false, "isPerfectlyPlastic", "indicates wether the behaviour model is perfectly plastic"))
    , d_modelName(initData(&d_modelName, std::string("RambergOsgood"), "modelName", "the name of the 1D contitutive law model to be used in plastic deformation"))
    , d_zSection(initData(&d_zSection, (Real)zSection, "zSection", "length of the section in the z direction for rectangular beams"))
    , d_ySection(initData(&d_ySection, (Real)ySection, "ySection", "length of the section in the y direction for rectangular beams"))
    , d_useSymmetricAssembly(initData(&d_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , d_isTimoshenko(initData(&d_isTimoshenko, isTimoshenko, "isTimoshenko", "implements a Timoshenko beam model"))
    , m_edgeHandler(NULL)
{
    m_edgeHandler = new BeamFFEdgeHandler(this, &m_beamsData);

    d_poissonRatio.setRequired(true);
    d_youngModulus.setReadOnly(true);
}

template<class DataTypes>
BeamPlasticFEMForceField<DataTypes>::~BeamPlasticFEMForceField()
{
    if (m_edgeHandler) delete m_edgeHandler;
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
    this->core::behavior::ForceField<DataTypes>::init();
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    m_topology = context->getMeshTopology();

    m_stiffnessContainer = context->core::objectmodel::BaseContext::get<container::StiffnessContainer>();
    m_poissonContainer = context->core::objectmodel::BaseContext::get<container::PoissonContainer>();

    // Retrieving the 1D plastic constitutive law model
    std::string constitutiveModel = d_modelName.getValue();
    if (constitutiveModel == "RambergOsgood")
    {
        Real youngModulus = d_youngModulus.getValue();
        Real yieldStress = d_yieldStress.getValue();
        fem::RambergOsgood<DataTypes> *RambergOsgoodModel = new (fem::RambergOsgood<DataTypes>)(youngModulus, yieldStress);
        m_ConstitutiveLaw = RambergOsgoodModel;
        if (this->f_printLog.getValue())
            msg_info() << "The model is " << constitutiveModel;
    }
    else
    {
        msg_error() << "constitutive law model name " << constitutiveModel << " is not valid (should be RambergOsgood)";
    }


    if (m_topology==NULL)
    {
        serr << "ERROR(BeamPlasticFEMForceField): object must have a BaseMeshTopology (i.e. EdgeSetTopology or MeshTopology)."<<sendl;
        return;
    }
    else
    {
        if (m_topology->getNbEdges()==0)
        {
            serr << "ERROR(BeamPlasticFEMForceField): topology is empty."<<sendl;
            return;
        }
        m_indexedElements = &m_topology->getEdges();
    }

    m_beamsData.createTopologicalEngine(m_topology,m_edgeHandler);
    m_beamsData.registerTopologicalData();

    reinit();
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::reinit()
{
    size_t n = m_indexedElements->size();

    //Initialises the lastPos field with the rest position
    m_lastPos = mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    m_prevStresses.resize(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 27; j++)
            m_prevStresses[i][j] = VoigtTensor2::Zero();

    if (d_useConsistentTangentOperator.getValue())
    {
        // No need to store elastic predictors at each iteration if the consistent
        // tangent operator is not used.
        m_elasticPredictors.resize(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 27; j++)
                m_elasticPredictors[i][j] = VoigtTensor2::Zero();
    }

    initBeams( n );
    for (unsigned int i=0; i<n; ++i)
        reinitBeam(i);
    msg_info() << "reinit OK, "<<n<<" elements." ;
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::initBeams(size_t size)
{
    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
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
    if (m_stiffnessContainer)
        stiffness = m_stiffnessContainer->getStiffness(i) ;
    else
        stiffness =  d_youngModulus.getValue() ;

    yieldStress = d_yieldStress.getValue();
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
    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    StiffnessMatrix& Kt_loc = bd[i]._Kt_loc;
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
    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
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
        phiY = (Real)(12.0*_E*_Iy / (kappaZ*_G*_A*L2));
        phiZ = (Real)(12.0*_E*_Iz / (kappaY*_G*_A*L2));
    }

    double phiYInv = (Real)(1 / (1 + phiY));
    double phiZInv = (Real)(1 / (1 + phiZ));

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

        _BeMatrices[i].block<6, 1>(0, 6) = -_BeMatrices[i].block<6, 1>(0, 0);

        _BeMatrices[i].block<6, 1>(0, 7) = -_BeMatrices[i].block<6, 1>(0, 1);

        _BeMatrices[i].block<6, 1>(0, 8) = -_BeMatrices[i].block<6, 1>(0, 2);

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
    // NB: each vector (of type helper::fixed_array) contains 27 components,
    // associated with the 27 Gauss points used for reduced integration
    _pointMechanicalState.assign(ELASTIC);
    _beamMechanicalState = ELASTIC;
    
    _localYieldStresses.assign(yS);
    _backStresses.assign(VoigtTensor2::Zero()); // TO DO: check if zero is correct
    _effectivePlasticStrains.assign(0.0);


    //**********************************//
}

template <class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::reset()
{
    //serr<<"BeamPlasticFEMForceField<DataTypes>::reset"<<sendl;

    for (unsigned i = 0; i < m_prevStresses.size(); ++i)
        for (unsigned j = 0; j < 27; ++j)
            m_prevStresses[i][j] = VoigtTensor2::Zero();

    if (d_useConsistentTangentOperator.getValue())
    {
        for (unsigned i = 0; i < m_elasticPredictors.size(); ++i)
            for (unsigned j = 0; j < 27; ++j)
                m_elasticPredictors[i][j] = VoigtTensor2::Zero();

    }

    // TO DO: call to init?
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeStiffness(int i, Index, Index)
{
    Real   phiy, phiz;
    Real _L = (Real)m_beamsData.getValue()[i]._L;
    Real _A = (Real)m_beamsData.getValue()[i]._A;
    Real _nu = (Real)m_beamsData.getValue()[i]._nu;
    Real _E = (Real)m_beamsData.getValue()[i]._E;
    Real _Iy = (Real)m_beamsData.getValue()[i]._Iy;
    Real _Iz = (Real)m_beamsData.getValue()[i]._Iz;
    Real _G = (Real)m_beamsData.getValue()[i]._G;
    Real _J = (Real)m_beamsData.getValue()[i]._J;
    Real L2 = (Real)(_L * _L);
    Real L3 = (Real)(L2 * _L);
    Real EIy = (Real)(_E * _Iy);
    Real EIz = (Real)(_E * _Iz);

    // Find shear-deformation parameters
    if (_A == 0)
    {
        phiy = 0.0;
        phiz = 0.0;
    }
    else
    {
        phiy = (Real)(24.0 * (1.0 + _nu) * _Iz / (_A * L2));
        phiz = (Real)(24.0 * (1.0 + _nu) * _Iy / (_A * L2));
    }

    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    StiffnessMatrix& k_loc = bd[i]._k_loc;

    // Define stiffness matrix 'k' in local coordinates
    k_loc.clear();
    k_loc[6][6] = k_loc[0][0] = _E*_A / _L;
    k_loc[7][7] = k_loc[1][1] = (Real)(12.0*EIz / (L3*(1.0 + phiy)));
    k_loc[8][8] = k_loc[2][2] = (Real)(12.0*EIy / (L3*(1.0 + phiz)));
    k_loc[9][9] = k_loc[3][3] = _G*_J / _L;
    k_loc[10][10] = k_loc[4][4] = (Real)((4.0 + phiz)*EIy / (_L*(1.0 + phiz)));
    k_loc[11][11] = k_loc[5][5] = (Real)((4.0 + phiy)*EIz / (_L*(1.0 + phiy)));

    k_loc[4][2] = (Real)(-6.0*EIy / (L2*(1.0 + phiz)));
    k_loc[5][1] = (Real)(6.0*EIz / (L2*(1.0 + phiy)));
    k_loc[6][0] = -k_loc[0][0];
    k_loc[7][1] = -k_loc[1][1];
    k_loc[7][5] = -k_loc[5][1];
    k_loc[8][2] = -k_loc[2][2];
    k_loc[8][4] = -k_loc[4][2];
    k_loc[9][3] = -k_loc[3][3];
    k_loc[10][2] = k_loc[4][2];
    k_loc[10][4] = (Real)((2.0 - phiz)*EIy / (_L*(1.0 + phiz)));
    k_loc[10][8] = -k_loc[4][2];
    k_loc[11][1] = k_loc[5][1];
    k_loc[11][5] = (Real)((2.0 - phiy)*EIz / (_L*(1.0 + phiy)));
    k_loc[11][7] = -k_loc[5][1];

    for (int i = 0; i <= 10; i++)
        for (int j = i + 1; j<12; j++)
            k_loc[i][j] = k_loc[j][i];

    m_beamsData.endEdit();
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::BeamFFEdgeHandler::applyCreateFunction(unsigned int edgeIndex, BeamInfo &ei, const core::topology::BaseMeshTopology::Edge &, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
    {
        ff->reinitBeam(edgeIndex);
        ei = ff->m_beamsData.getValue()[edgeIndex];
    }
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
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

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
    Real k = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    defaulttype::BaseMatrix* mat = r.matrix;

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

            defaulttype::Quat& q = beamQuat(i); //x[a].getOrientation();
            q.normalize();
            Transformation R,Rt;
            q.toMatrix(R);
            Rt.transpose(R);
            StiffnessMatrix K;
            bool exploitSymmetry = d_useSymmetricAssembly.getValue();

            StiffnessMatrix K0;
            if (beamMechanicalState == PLASTIC)
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
                        defaulttype::Mat<3, 3, Real> m;
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
                        defaulttype::Mat<3, 3, Real> m;
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

    std::vector<defaulttype::Vector3> centrelinePoints[1];
    std::vector<defaulttype::Vector3> gaussPoints[1];
    std::vector<defaulttype::Vec<4, float>> colours[1];

    for (unsigned int i=0; i<m_indexedElements->size(); ++i)
        drawElement(i, gaussPoints, centrelinePoints, colours, x);

    vparams->drawTool()->setPolygonMode(2, true);
    vparams->drawTool()->setLightingEnabled(true);
    vparams->drawTool()->drawPoints(gaussPoints[0], 3, colours[0]);
    vparams->drawTool()->drawLines(centrelinePoints[0], 1.0, defaulttype::Vec<4, float>(0.24f, 0.72f, 0.96f, 1.0f));
    vparams->drawTool()->setLightingEnabled(false);
    vparams->drawTool()->setPolygonMode(0, false);
}

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::drawElement(int i, std::vector< defaulttype::Vector3 >* gaussPoints,
                                                 std::vector< defaulttype::Vector3 >* centrelinePoints,
                                                 std::vector<defaulttype::Vec<4, float>>* colours,
                                                 const VecCoord& x)
{
    Index a = (*m_indexedElements)[i][0];
    Index b = (*m_indexedElements)[i][1];

    defaulttype::Vec3d pa, pb;
    pa = x[a].getCenter();
    pb = x[b].getCenter();

    defaulttype::Vec3d beamVec;
    const defaulttype::Quat& q = beamQuat(i);

    //***** Gauss points *****//

    //Compute current displacement

    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    beamQuat(i) = x[a].getOrientation();
    beamQuat(i).normalize();

    m_beamsData.endEdit();

    defaulttype::Vec<3, Real> u, P1P2, P1P2_0;
    // local displacement
    Eigen::Matrix<double, 12, 1> disp;

    // translations //
    P1P2_0 = x0[b].getCenter() - x0[a].getCenter();
    P1P2_0 = x0[a].getOrientation().inverseRotate(P1P2_0);
    P1P2 = x[b].getCenter() - x[a].getCenter();
    P1P2 = x[a].getOrientation().inverseRotate(P1P2);
    u = P1P2 - P1P2_0;

    disp[0] = 0.0; 	disp[1] = 0.0; 	disp[2] = 0.0;
    disp[6] = u[0]; disp[7] = u[1]; disp[8] = u[2];

    // rotations //
    defaulttype::Quat dQ0, dQ;

    dQ0 = qDiff(x0[b].getOrientation(), x0[a].getOrientation());
    dQ = qDiff(x[b].getOrientation(), x[a].getOrientation());

    dQ0.normalize();
    dQ.normalize();

    defaulttype::Quat tmpQ = qDiff(dQ, dQ0);
    tmpQ.normalize();

    u = tmpQ.quatToRotationVector();

    disp[3] = 0.0; 	disp[4] = 0.0; 	disp[5] = 0.0;
    disp[9] = u[0]; disp[10] = u[1]; disp[11] = u[2];

    //Compute the positions of the Gauss points
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    Eigen::Matrix<double, 3, 12> N;
    const helper::fixed_array<MechanicalState, 27>& pointMechanicalState = m_beamsData.getValue()[i]._pointMechanicalState;
    int gaussPointIt = 0; //incremented in the lambda function to iterate over Gauss points

    LambdaType computeGaussCoordinates = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        //Shape function
        N = m_beamsData.getValue()[i]._N[gaussPointIt];
        Eigen::Matrix<double, 3, 1> u = N*disp;

        defaulttype::Vec3d beamVec = {u[0]+u1, u[1]+u2, u[2]+u3};
        defaulttype::Vec3d gp = pa + q.rotate(beamVec);
        gaussPoints[0].push_back(gp);

        if (pointMechanicalState[gaussPointIt] == ELASTIC)
            colours[0].push_back({1.0f,0.015f,0.015f,1.0f}); //RED
        else if (pointMechanicalState[gaussPointIt] == PLASTIC)
            colours[0].push_back({0.051f,0.15f,0.64f,1.0f}); //BLUE
        else
            colours[0].push_back({0.078f,0.41f,0.078f,1.0f}); //GREEN

        gaussPointIt++; //next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeGaussCoordinates);

    //****** Centreline ******//
    int nbSeg = m_beamsData.getValue()[i]._nbCentrelineSeg; //number of segments descretising the centreline

    centrelinePoints[0].push_back(pa);

    Eigen::Matrix<double, 3, 12> drawN;
    const double L = m_beamsData.getValue()[i]._L;
    for (int drawPointIt = 0; drawPointIt < nbSeg - 1; drawPointIt++)
    {
        //Shape function of the centreline point
        drawN = m_beamsData.getValue()[i]._drawN[drawPointIt];
        Eigen::Matrix<double, 3, 1> u = drawN*disp;

        defaulttype::Vec3d beamVec = {u[0] + (drawPointIt +1)*(L/nbSeg), u[1], u[2]};
        defaulttype::Vec3d clp = pa + q.rotate(beamVec);
        centrelinePoints[0].push_back(clp); //First time as the end of the former segment
        centrelinePoints[0].push_back(clp); //Second time as the beginning of the next segment
    }

    centrelinePoints[0].push_back(pb);
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
        const defaulttype::Vector3 &pt = p[i].getCenter();

        for (int c = 0; c<3; c++)
        {
            if (pt[c] > maxBBox[c]) maxBBox[c] = pt[c];
            if (pt[c] < minBBox[c]) minBBox[c] = pt[c];
        }
    }

    this->f_bbox.setValue(params, sofa::defaulttype::TBoundingBox<Real>(minBBox, maxBBox));

}


/*****************************************************************************/
/*                        PLASTIC IMPLEMENTATION                             */
/*****************************************************************************/


/***************************** Virtual Displacement **************************/

template<class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeVDStiffness(int i, Index, Index)
{
    Real _L = (Real)m_beamsData.getValue()[i]._L;
    Real _yDim = (Real)m_beamsData.getValue()[i]._yDim;
    Real _zDim = (Real)m_beamsData.getValue()[i]._zDim;

    const double E = (Real)m_beamsData.getValue()[i]._E;
    const double nu = (Real)m_beamsData.getValue()[i]._nu;

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[i]._materialBehaviour;
    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    StiffnessMatrix& Ke_loc = bd[i]._Ke_loc;
    Ke_loc.clear();

    // Reduced integration
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    // Setting variables for the reduced intergation process defined in quadrature.h

    Eigen::Matrix<double, 6, 12> Be;
    // Stress matrix, to be integrated
    Eigen::Matrix<double, 12, 12> stiffness = Eigen::Matrix<double, 12, 12>::Zero();

    int gaussPointIterator = 0; //incremented in the lambda function to iterate over Gauss points

    LambdaType computeStressMatrix = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        Be = m_beamsData.getValue()[i]._BeMatrices[gaussPointIterator];

        stiffness += (w1*w2*w3)*beTCBeMult(Be.transpose(), C, nu, E);

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

    Real youngModulus = (Real)m_beamsData.getValue()[i]._E;
    Real poissonRatio = (Real)m_beamsData.getValue()[i]._nu;

    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());

    Eigen::Matrix<double, 6, 6>& C = bd[i]._materialBehaviour;
    // Material behaviour matrix, here: Hooke's law
    //TO DO: handle incompressible materials (with nu = 0.5)
    C(0, 0) = C(1, 1) = C(2, 2) = 1;
    C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = poissonRatio / (1 - poissonRatio);
    C(0, 3) = C(0, 4) = C(0, 5) = 0;
    C(1, 3) = C(1, 4) = C(1, 5) = 0;
    C(2, 3) = C(2, 4) = C(2, 5) = 0;
    C(3, 0) = C(3, 1) = C(3, 2) = C(3, 4) = C(3, 5) = 0;
    C(4, 0) = C(4, 1) = C(4, 2) = C(4, 3) = C(4, 5) = 0;
    C(5, 0) = C(5, 1) = C(5, 2) = C(5, 3) = C(5, 4) = 0;
    C(3, 3) = C(4, 4) = C(5, 5) = (1 - 2 * poissonRatio) / (1 - poissonRatio);
    C *= (youngModulus*(1 - poissonRatio)) / ((1 + poissonRatio)*(1 - 2 * poissonRatio));

    m_beamsData.endEdit();
};

template< class DataTypes>
bool BeamPlasticFEMForceField<DataTypes>::goToPlastic(const VoigtTensor2 &stressTensor,
                                                 const double yieldStress,
                                                 const bool verbose /*=FALSE*/)
{
    double threshold = 1e2; //TO DO: choose adapted threshold

    double yield = vonMisesYield(stressTensor, yieldStress);
    if (verbose)
    {
        std::cout.precision(17);
        std::cout << yield << std::scientific << " "; //DEBUG
    }
    return yield > threshold;
}

template< class DataTypes>
bool BeamPlasticFEMForceField<DataTypes>::goToPostPlastic(const VoigtTensor2 &stressTensor,
                                                     const VoigtTensor2 &stressIncrement,
                                                     const bool verbose /*=FALSE*/)
{
    double threshold = -0.f; //TO DO: use proper threshold

    // Computing the unit normal to the yield surface from the Von Mises gradient
    VoigtTensor2 gradient = vonMisesGradient(stressTensor);
    VoigtTensor2 yieldNormal = helper::rsqrt(2.0 / 3)*gradient;

    // Computing the dot product with the incremental elastic predictor
    double cp = voigtDotProduct(yieldNormal.transpose(), stressIncrement);
    if (verbose)
    {
        std::cout.precision(17);
        std::cout << cp << std::scientific << " "; //DEBUG
    }
    return (cp < threshold); //if true, the stress point goes into post-plastic phase
}

template< class DataTypes>
Eigen::Matrix<double, 6, 1> BeamPlasticFEMForceField<DataTypes>::deviatoricStress(const VoigtTensor2 &stressTensor)
{
    // Returns the deviatoric stress from a given stress tensor in Voigt notation

    VoigtTensor2 deviatoricStress = stressTensor;
    double mean = (stressTensor[0] + stressTensor[1] + stressTensor[2]) / 3.0;
    for (int i = 0; i < 3; i++)
        deviatoricStress[i] -= mean;

    return deviatoricStress;
}

template< class DataTypes>
double BeamPlasticFEMForceField<DataTypes>::equivalentStress(const VoigtTensor2 &stressTensor)
{
    double res = 0.0;
    double sigmaX = stressTensor[0];
    double sigmaY = stressTensor[1];
    double sigmaZ = stressTensor[2];
    double sigmaYZ = stressTensor[3];
    double sigmaZX = stressTensor[4];
    double sigmaXY = stressTensor[5];

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
Eigen::Matrix<double, 6, 1> BeamPlasticFEMForceField<DataTypes>::vonMisesGradient(const VoigtTensor2 &stressTensor)
{
    // NB: this gradient represent the normal to the yield surface
    // in case the Von Mises yield criterion is used.
    // /!\ the Norm of the gradient is sqrt(3/2): it has to be multiplied
    // by sqrt(2/3) to give the unit normal to the yield surface

    VoigtTensor2 gradient = VoigtTensor2::Zero();

    if (stressTensor.isZero())
        return gradient; //TO DO: is that correct ?

    double sigmaX = stressTensor[0];
    double sigmaY = stressTensor[1];
    double sigmaZ = stressTensor[2];
    double sigmaYZ = stressTensor[3];
    double sigmaZX = stressTensor[4];
    double sigmaXY = stressTensor[5];

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
Eigen::Matrix<double, 9, 9> BeamPlasticFEMForceField<DataTypes>::vonMisesHessian(const VoigtTensor2 &stressTensor,
                                                                            const double yieldStress)
{
    VectTensor4 hessian = VectTensor4::Zero();

    if (stressTensor.isZero())
        return hessian; //TO DO: is that correct ?

    //Order 1 terms
    double sigmaXX = stressTensor[0];
    double sigmaYY = stressTensor[1];
    double sigmaZZ = stressTensor[2];
    double sigmaYZ = stressTensor[3];
    double sigmaZX = stressTensor[4];
    double sigmaXY = stressTensor[5];

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
    double sigmaXX = stressTensor[0];
    double sigmaXY = stressTensor[1];
    double sigmaXZ = stressTensor[2];
    double sigmaYX = stressTensor[3];
    double sigmaYY = stressTensor[4];
    double sigmaYZ = stressTensor[5];
    double sigmaZX = stressTensor[6];
    double sigmaZY = stressTensor[7];
    double sigmaZZ = stressTensor[8];

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

    return helper::rsqrt(3.0 / 2.0)*vectDevStress.norm();
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
Eigen::Matrix<double, 9, 1> BeamPlasticFEMForceField<DataTypes>::vectVonMisesGradient(const VectTensor2 &stressTensor)
{
    // Computation of Von Mises yield function gradient,
    // in vector notation

    VectTensor2 gradient = VectTensor2::Zero();

    if (stressTensor.isZero())
        return gradient; //TO DO: is that correct ?

    double sigmaXX = stressTensor[0];
    double sigmaXY = stressTensor[1];
    double sigmaXZ = stressTensor[2];
    double sigmaYX = stressTensor[3];
    double sigmaYY = stressTensor[4];
    double sigmaYZ = stressTensor[5];
    double sigmaZX = stressTensor[6];
    double sigmaZY = stressTensor[7];
    double sigmaZZ = stressTensor[8];

    gradient[0] = 2 * sigmaXX - sigmaYY - sigmaZZ;
    gradient[1] = 3 * sigmaXY;
    gradient[2] = 3 * sigmaXZ;
    gradient[3] = 3 * sigmaYX;
    gradient[4] = 2 * sigmaYY - sigmaZZ - sigmaXX;
    gradient[5] = 3 * sigmaYZ;
    gradient[6] = 3 * sigmaZX;
    gradient[7] = 3 * sigmaZY;
    gradient[8] = 2 * sigmaZZ - sigmaXX - sigmaYY;

    double sigmaEq = vectEquivalentStress(stressTensor);
    gradient *= 1 / (2 * sigmaEq);

    return gradient;
}


template< class DataTypes>
Eigen::Matrix<double, 6, 1> BeamPlasticFEMForceField<DataTypes>::devVonMisesGradient(const VoigtTensor2 &stressTensor)
{
    // Computation of the gradient of the Von Mises function, at stressTensor,
    // using the expression of the deviatoric stress tensor

    VoigtTensor2 gradient = VoigtTensor2::Zero();

    if (stressTensor.isZero())
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
Eigen::Matrix<double, 12, 1> BeamPlasticFEMForceField<DataTypes>::beTTensor2Mult(const Eigen::Matrix<double, 12, 6> &BeT,
    const VoigtTensor2 &T)
{
    // In Voigt notation, 3 rows in Be (i.e. 3 columns in Be^T) are missing.
    // These rows correspond to the 3 symmetrical non-diagonal elements of the
    // tensors, which are not expressed in Voigt notation.
    // We have to add the contribution of these rows in the computation BeT*Tensor.

    Eigen::Matrix<double, 12, 1> res = Eigen::Matrix<double, 12, 1>::Zero();

    res += BeT*T; // contribution of the 6 first columns

                  // We compute the contribution of the 3 missing columns.
                  // This can be achieved with block computation.

    Eigen::Matrix<double, 12, 3> additionalColumns = Eigen::Matrix<double, 12, 3>::Zero();
    additionalColumns.block<12, 1>(0, 0) = BeT.block<12, 1>(0, 3); // T_yz
    additionalColumns.block<12, 1>(0, 1) = BeT.block<12, 1>(0, 4); // T_zx
    additionalColumns.block<12, 1>(0, 2) = BeT.block<12, 1>(0, 5); // T_xy

    Eigen::Matrix<double, 3, 1> additionalTensorElements = Eigen::Matrix<double, 3, 1>::Zero();
    additionalTensorElements[0] = T[3]; // T_yz
    additionalTensorElements[1] = T[4]; // T_zx
    additionalTensorElements[2] = T[5]; // T_xy

    res += additionalColumns*additionalTensorElements;
    return res;
}

template< class DataTypes>
Eigen::Matrix<double, 12, 12> BeamPlasticFEMForceField<DataTypes>::beTCBeMult(const Eigen::Matrix<double, 12, 6> &BeT,
    const VoigtTensor4 &C,
    const double nu, const double E)
{
    // In Voigt notation, 3 rows in Be (i.e. 3 columns in Be^T) are missing.
    // These rows correspond to the 3 symmetrical non-diagonal elements of the
    // tensors, which are not expressed in Voigt notation.
    // We have to add the contribution of these rows in the computation BeT*C*Be.

    // First part of the computation in Voigt Notation
    Eigen::Matrix<double, 12, 12> res = Eigen::Matrix<double, 12, 12>::Zero();
    res += BeT*C*(BeT.transpose()); // contribution of the 6 first columns

                                    // Second part : contribution of the missing rows in Be
    Eigen::Matrix<double, 12, 3> leftTerm = Eigen::Matrix<double, 12, 3>::Zero();
    leftTerm.block<12, 1>(0, 0) = BeT.block<12, 1>(0, 3); // S_3N^T
    leftTerm.block<12, 1>(0, 1) = BeT.block<12, 1>(0, 4); // S_4N^T
    leftTerm.block<12, 1>(0, 2) = BeT.block<12, 1>(0, 5); // S_5N^T

    Eigen::Matrix<double, 3, 12> rightTerm = leftTerm.transpose();
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

    Eigen::Matrix<double, 12, 1> fint = Eigen::VectorXd::Zero(12);

    if (d_isPerfectlyPlastic.getValue())
    {
        //const MechanicalState beamMechanicalState = m_beamsData.getValue()[i]._beamMechanicalState;
        //if (beamMechanicalState == ELASTIC)
        //    computeElasticForce(fint, x, i, a, b);
        //else if (beamMechanicalState == PLASTIC)
        //    computePlasticForce(fint, x, i, a, b);
        //else
        //    computePostPlasticForce(fint, x, i, a, b);

        computeForceWithPerfectPlasticity(fint, x, i, a, b);
    }
    else
    {
        computeForceWithHardening(fint, x, i, a, b);
    }


    //Passes the contribution to the global system
    nodalForces force;

    for (int i = 0; i < 12; i++)
        force[i] = fint(i);

    Vec3 fa1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[0], force[1], force[2]));
    Vec3 fa2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[3], force[4], force[5]));

    Vec3 fb1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[6], force[7], force[8]));
    Vec3 fb2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[9], force[10], force[11]));

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
    Displacement local_depl;
    defaulttype::Vec<3, Real> u;
    defaulttype::Quat& q = beamQuat(i); //x[a].getOrientation();
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
    defaulttype::Vec<12, Real> local_dforce;

    // The stiffness matrix we use depends on the mechanical state of the beam element

    if (beamMechanicalState == PLASTIC)
        local_dforce = m_beamsData.getValue()[i]._Kt_loc * local_depl;
    else
    {
        if (d_usePrecomputedStiffness.getValue())
            // this computation can be optimised: (we know that half of "depl" is null)
            local_dforce = m_beamsData.getValue()[i]._k_loc * local_depl;
        else
            local_dforce = m_beamsData.getValue()[i]._Ke_loc * local_depl;
    }

    Vec3 fa1 = q.rotate(defaulttype::Vec3d(local_dforce[0], local_dforce[1], local_dforce[2]));
    Vec3 fa2 = q.rotate(defaulttype::Vec3d(local_dforce[3], local_dforce[4], local_dforce[5]));
    Vec3 fb1 = q.rotate(defaulttype::Vec3d(local_dforce[6], local_dforce[7], local_dforce[8]));
    Vec3 fb2 = q.rotate(defaulttype::Vec3d(local_dforce[9], local_dforce[10], local_dforce[11]));

    df[a] += Deriv(-fa1, -fa2) * fact;
    df[b] += Deriv(-fb1, -fb2) * fact;
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::updateTangentStiffness(int i,
                                                            Index a,
                                                            Index b)
{
    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    StiffnessMatrix& Kt_loc = bd[i]._Kt_loc;
    const Eigen::Matrix<double, 6, 6>& C = bd[i]._materialBehaviour;
    const double E = bd[i]._E;
    const double nu = bd[i]._nu;
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[i]._pointMechanicalState;

    // Reduced integration
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    // Setting variables for the reduced intergation process defined in quadrature.h
    Eigen::Matrix<double, 6, 12> Be;
    VoigtTensor4 Cep = VoigtTensor4::Zero(); //plastic behaviour tensor
    VoigtTensor2 gradient;

    //Auxiliary matrices
    VoigtTensor2 Cgrad;
    Eigen::Matrix<double, 1, 6> gradTC;

    //Result matrix
    Eigen::Matrix<double, 12, 12> tangentStiffness = Eigen::Matrix<double, 12, 12>::Zero();

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
            if (gradient.isZero() || pointMechanicalState[gaussPointIt] != PLASTIC)
                Cep = C; //TO DO: is that correct ?
            else
            {
                if (!d_isPerfectlyPlastic.getValue())
                {
                    VoigtTensor2 normal = helper::rsqrt(2.0 / 3.0)*gradient;
                    VectTensor2 vectNormal = voigtToVect2(normal);
                    VectTensor4 vectC = voigtToVect4(C);
                    VectTensor4 vectCep = VectTensor4::Zero();

                    VectTensor2 CN = vectC*vectNormal;
                    // NtC = (NC)t because of C symmetry
                    vectCep = vectC - (CN*CN.transpose()) / (vectNormal.transpose()*CN + (2.0 / 3.0)*plasticModulus);

                    // DEBUG : old version
                    //double mu = E / (2 * (1 + nu)); // Lame coefficient
                    //VectTensor4 vectCep2 = VectTensor4::Zero();
                    //vectCep2 = vectC - (1.0 / (1.0 + plasticModulus / (3 * mu)))*vectC*(vectNormal*vectNormal.transpose());
                    //double diffCep = (vectCep - vectCep2).norm();
                    //std::cout << "Norme de la difference dans le calcul de Cep : " << diffCep << std::endl;

                    Cep = vectToVoigt4(vectCep);
                }
                else
                {
                    VoigtTensor2 normal = helper::rsqrt(2.0 / 3.0)*gradient;
                    VectTensor2 vectNormal = voigtToVect2(normal);
                    VectTensor4 vectC = voigtToVect4(C);
                    VectTensor4 vectCep = VectTensor4::Zero();

                    VectTensor2 CN = vectC*vectNormal;
                    vectCep = vectC - (CN*CN.transpose()) / (vectNormal.transpose()*CN);

                    // DEBUG : old version
                    /*Cgrad = C*gradient;
                    gradTC = Cgrad.transpose();
                    Cep = C - (Cgrad*gradTC) / (voigtDotProduct(gradTC, gradient));
                    VectTensor4 vectCep2 = voigtToVect4(Cep);
                    double diffCep = (vectCep2 - vectCep).norm();
                    std::cout << "Norme de la difference dans le calcul de Cep : " << diffCep << std::endl;*/

                    Cep = vectToVoigt4(vectCep);
                }
            }
        }
        else // d_useConsistentTangentOperator = true
        {
            if (pointMechanicalState[gaussPointIt] != PLASTIC)
                Cep = C; //TO DO: is that correct ?
            else
            {
                if (!d_isPerfectlyPlastic.getValue())
                {
                    //Computation of matrix H as in Studies in anisotropic plasticity with reference to the Hill criterion, De Borst and Feenstra, 1990
                    VectTensor4 H = VectTensor4::Zero();
                    VectTensor4 I = VectTensor4::Identity();
                    VoigtTensor2 elasticPredictor = m_elasticPredictors[i][gaussPointIt];

                    VectTensor2 vectGradient = voigtToVect2(gradient);
                    VectTensor4 vectC = voigtToVect4(C);
                    double yieldStress = m_beamsData.getValue()[i]._localYieldStresses[gaussPointIt];
                    double DeltaLambda = vonMisesYield(elasticPredictor, yieldStress) / (vectGradient.transpose()*vectC*vectGradient);
                    VectTensor4 vectHessian = vonMisesHessian(elasticPredictor, yieldStress);

                    VectTensor4 M = (I + DeltaLambda*vectC*vectHessian);
                    // M is symmetric positive definite, we perform Cholesky decomposition to invert it
                    VectTensor4 invertedM = M.llt().solve(I);
                    H = invertedM*vectC;

                    //Computation of Cep
                    if (gradient.isZero())
                        Cep = vectToVoigt4(H);
                    else
                    {
                        VectTensor4 consistentCep = VectTensor4::Zero();
                        Eigen::Matrix<double, 1, 9> gradTH = vectGradient.transpose()*H;
                        consistentCep = H - (H*vectGradient*gradTH) / ((gradTH*vectGradient) + plasticModulus);
                        Cep = vectToVoigt4(consistentCep);

                        ////DEBUG : comparison with classic tangent operator
                        //VoigtTensor2 normal = helper::rsqrt(2.0 / 3.0)*gradient;
                        //VectTensor2 vectNormal = voigtToVect2(normal);
                        //VectTensor4 vectCep2 = VectTensor4::Zero();
                        //VectTensor2 CN = vectC*vectNormal;
                        //vectCep2 = vectC - (CN*CN.transpose()) / (vectNormal.transpose()*CN + (2.0 / 3.0)*plasticModulus);
                        //double diffCep = (vectCep2 - consistentCep).norm();
                        //std::cout << "Norme de la difference dans le calcul de Cep : " << diffCep << std::endl;
                    }
                }
                else
                {
                    //Computation of matrix H as in Studies in anisotropic plasticity with reference to the Hill criterion, De Borst and Feenstra, 1990
                    VectTensor4 H = VectTensor4::Zero();
                    VectTensor4 I = VectTensor4::Identity();
                    VoigtTensor2 elasticPredictor = m_elasticPredictors[i][gaussPointIt];

                    VectTensor2 vectGradient = voigtToVect2(gradient);
                    VectTensor4 vectC = voigtToVect4(C);
                    double yieldStress = m_beamsData.getValue()[i]._localYieldStresses[gaussPointIt];
                    // NB: the gradient is the same between the elastic predictor and the new stress
                    double DeltaLambda = vonMisesYield(elasticPredictor, yieldStress) / (vectGradient.transpose()*vectC*vectGradient);
                    VectTensor4 vectHessian = vonMisesHessian(elasticPredictor, yieldStress);

                    VectTensor4 M = (I + DeltaLambda*vectC*vectHessian);
                    // M is symmetric positive definite, we perform Cholesky decomposition to invert it
                    VectTensor4 invertedM = M.llt().solve(I);
                    H = invertedM*vectC;

                    //Computation of Cep
                    if (gradient.isZero())
                        Cep = vectToVoigt4(H);
                    else
                    {
                        VectTensor4 consistentCep = VectTensor4::Zero();
                        Eigen::Matrix<double, 1, 9> gradTH = vectGradient.transpose()*H;
                        consistentCep = H - (H*vectGradient*gradTH) / (gradTH*vectGradient);
                        Cep = vectToVoigt4(consistentCep);

                        ////DEBUG : comparison with classic tangent operator
                        //VoigtTensor2 normal = helper::rsqrt(2.0 / 3.0)*gradient;
                        //VectTensor2 vectNormal = voigtToVect2(normal);
                        //VectTensor4 vectCep2 = VectTensor4::Zero();
                        //VectTensor2 CN = vectC*vectNormal;
                        //vectCep2 = vectC - (CN*CN.transpose()) / (vectNormal.transpose()*CN);
                        //double diffCep = (vectCep2 - consistentCep).norm();
                        //std::cout << "Norme de la difference dans le calcul de Cep : " << diffCep << std::endl;
                    }
                } // end if d_isPerfectlyPlastic = true
            } // end if pointMechanicalState[gaussPointIt] == PLASTIC
        } // end if d_useConsistentTangentOperator = true

        tangentStiffness += (w1*w2*w3)*beTCBeMult(Be.transpose(), Cep, nu, E);

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
void BeamPlasticFEMForceField<DataTypes>::computeLocalDisplacement(const VecCoord& x, Displacement &localDisp,
                                                              int i, Index a, Index b)
{
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    beamQuat(i) = x[a].getOrientation();
    beamQuat(i).normalize();

    m_beamsData.endEdit();

    defaulttype::Vec<3, Real> u, P1P2, P1P2_0;

    // translations //
    P1P2_0 = x0[b].getCenter() - x0[a].getCenter();
    P1P2_0 = x0[a].getOrientation().inverseRotate(P1P2_0);
    P1P2 = x[b].getCenter() - x[a].getCenter();
    P1P2 = x[a].getOrientation().inverseRotate(P1P2);
    u = P1P2 - P1P2_0;

    localDisp[0] = 0.0; localDisp[1] = 0.0; localDisp[2] = 0.0;
    localDisp[6] = u[0]; localDisp[7] = u[1]; localDisp[8] = u[2];

    // rotations //
    defaulttype::Quat dQ0, dQ;

    // dQ = QA.i * QB ou dQ = QB * QA.i() ??
    dQ0 = qDiff(x0[b].getOrientation(), x0[a].getOrientation()); // x0[a].getOrientation().inverse() * x0[b].getOrientation();
    dQ = qDiff(x[b].getOrientation(), x[a].getOrientation()); // x[a].getOrientation().inverse() * x[b].getOrientation();
                                                              //u = dQ.toEulerVector() - dQ0.toEulerVector(); // Consider to use quatToRotationVector instead of toEulerVector to have the rotation vector

    dQ0.normalize();
    dQ.normalize();

    defaulttype::Quat tmpQ = qDiff(dQ, dQ0);
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
void BeamPlasticFEMForceField<DataTypes>::computeDisplacementIncrement(const VecCoord& pos, const VecCoord& lastPos, Displacement &currentDisp,
                                                                  Displacement &lastDisp, Displacement &dispIncrement, int i, Index a, Index b)
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
double BeamPlasticFEMForceField<DataTypes>::computePlasticModulusFromStress(const Eigen::Matrix<double, 6, 1> &stressState)
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

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeElasticForce(Eigen::Matrix<double, 12, 1> &internalForces,
                                                         const VecCoord& x, int index, Index a, Index b)
{
    // Here, all Gauss points are assumed to be in an ELASTIC state. Consequently,
    // the internal forces can be computed directly (i.e. not incrementally) with
    // the elastic stiffness matrix, and there is no need to compute a tangent
    // stiffness matrix.
    // A new stress tensor has to be computed (elastically) for the Gauss points,
    // to check whether any of them enters a PLASTIC state. If at least one of
    // them does, we update the mechanical states accordingly, and call
    // computePlasticForce, to carry out the approriate (incremental) computation.

    Displacement localDisp;
    computeLocalDisplacement(x, localDisp, index, a, b);

    Eigen::Matrix<double, 12, 1> eigenDepl;
    for (int i = 0; i < 12; i++)
        eigenDepl(i) = localDisp[i];

    //***** Test if we enter in plastic deformation *****//

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;

    Real yieldStress;
    VoigtTensor2 newStress;
    helper::fixed_array<VoigtTensor2, 27> newStresses;

    //For each Gauss point, we update the stress value for next iteration
    for (int gaussPointIt = 0; gaussPointIt < 27; gaussPointIt++)
    {
        yieldStress = m_beamsData.getValue()[index]._localYieldStresses[gaussPointIt];
        Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];
        newStress = C*Be*eigenDepl; // Point is assumed to be ELASTIC

                                    // Checking if the deformation becomes plastic

        bool isNewPlastic = goToPlastic(newStress, yieldStress);
        if (isNewPlastic)
        {
            // If a point is detected as entering a PLASTIC state, we stop
            // the computation and call computePlasticForce instead.
            // The computation of the internal forces will thus be carried out
            // incrementally, which will change nothing for the points remaining
            // in an ELASTIC state, but will allow the new PLASTIC points to be
            // handled correctly.
            computePlasticForce(internalForces, x, index, a, b);
            helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
            MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
            beamMechanicalState = PLASTIC;
            m_beamsData.endEdit();
            return;
        }

        newStresses[gaussPointIt] = newStress;
    }
    //***************************************************//

    // Here, all Gauss points remained ELASTIC (otherwise the method execution
    // would have been stopped by a call to computePlasticForce).

    // Storing the new stresses for the next time step, in case plasticity occurs.
    m_prevStresses[index] = newStresses;

    // As all the points are in an ELASTIC state, it is not necessary
    // to use reduced integration (all the computation is linear).
    nodalForces auxF = m_beamsData.getValue()[index]._Ke_loc * localDisp;

    for (int i = 0; i<12; i++)
        internalForces(i) = auxF[i]; //TO DO: not very efficient, we should settle for one data structure only

}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computePlasticForce(Eigen::Matrix<double, 12, 1> &internalForces,
                                                         const VecCoord& x, int index, Index a, Index b)
{
    // Here, at least one Gauss point in the element is in a PLASTIC state. Other
    // points can be also PLASTIC, but ELASTIC or POST-PLASTIC as well. Because
    // of the PLASTIC points, the computation has to be incremental, i.e. we
    // compute for each point the corresponding stress tensor increment from
    // last time step.
    //   - For ELASTIC points, the computation is equivalent to the direct
    // formulation used in computeElasticForce.
    //   - For PLASTIC points, we use the radial return algorithm, either with
    // a perfect plasticity model, or a isotropic hardening.
    //   - For POSTPLASTIC points, the computation is made directly, but takes
    // into account the plastic deformation history.
    // Stress increments are computed point by point in the dedicated method
    // (computeStressIncrement). The resulting new stress tensors are then
    // used in reduced integration to compute internal forces. The tangent
    // stiffness matrix is also updated accordingly.


    // Computes displacement increment, from last system solution
    Displacement currentDisp;
    Displacement lastDisp;
    Displacement dispIncrement;
    computeDisplacementIncrement(x, m_lastPos, currentDisp, lastDisp, dispIncrement, index, a, b);

    // Converts to Eigen data structure
    EigenDisplacement displacementIncrement;
    for (int k = 0; k < 12; k++)
        displacementIncrement(k) = dispIncrement[k];

    //All the rest of the force computation is made inside of the lambda function
    //as the stress and strain are computed for each Gauss point

    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;

    VoigtTensor2 initialStressPoint = VoigtTensor2::Zero();
    VoigtTensor2 strainIncrement = VoigtTensor2::Zero();
    VoigtTensor2 newStressPoint = VoigtTensor2::Zero();
    double lambdaIncrement = 0;

    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[index]._pointMechanicalState;
    bool isPlasticBeam = false;
    int gaussPointIt = 0;

    // Computation of the new stress point, through material point iterations as in Krabbenhoft lecture notes

    // This function is to be called if the last stress point corresponded to elastic deformation
    LambdaType computePlastic = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];
        MechanicalState &mechanicalState = pointMechanicalState[gaussPointIt];

        //Strain
        strainIncrement = Be*displacementIncrement;

        //Stress
        initialStressPoint = m_prevStresses[index][gaussPointIt];
        computeStressIncrement(index, gaussPointIt, initialStressPoint, newStressPoint,
            strainIncrement, lambdaIncrement, mechanicalState, currentDisp);

        isPlasticBeam = isPlasticBeam || (mechanicalState == PLASTIC);

        m_prevStresses[index][gaussPointIt] = newStressPoint;

        internalForces += (w1*w2*w3)*beTTensor2Mult(Be.transpose(), newStressPoint);

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[index]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computePlastic);

    // Updates the beam mechanical state information
    if (!isPlasticBeam)
    {
        MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
        beamMechanicalState = POSTPLASTIC;
    }
    m_beamsData.endEdit();

    //Update the tangent stiffness matrix with the new computed stresses
    //This matrix will then be used in addDForce and addKToMatrix methods
    if (isPlasticBeam)
        updateTangentStiffness(index, a, b);
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computePostPlasticForce(Eigen::Matrix<double, 12, 1> &internalForces,
    const VecCoord& x, int index, Index a, Index b)
{
    // Here, we assume that no Gauss point is in PLASTIC state, but the
    // plastic deformation history has to be taken into account for the points
    // in a POSTPLASTIC state.
    // TO DO: we assume that because the stress computation involves an additional
    // plastic strain term compared to the purely ELASTIC deformation, the
    // internal forces have to be computed through reduced integration. A proof
    // should be provided, including the computation of the stiffness matrix,
    // if necessary

    Displacement localDisp;
    computeLocalDisplacement(x, localDisp, index, a, b);

    Eigen::Matrix<double, 12, 1> eigenDepl;
    for (int i = 0; i < 12; i++)
        eigenDepl(i) = localDisp[i];

    //***** Test if we enter in plastic deformation *****//
    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;
    const helper::fixed_array<MechanicalState, 27>& pointMechanicalState = m_beamsData.getValue()[index]._pointMechanicalState;

    const MechanicalState& beamMechanicalState = m_beamsData.getValue()[index]._beamMechanicalState;
    const helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = m_beamsData.getValue()[index]._plasticStrainHistory;
    Real yieldStress;
    VoigtTensor2 newStress;
    helper::fixed_array<VoigtTensor2, 27> newStresses;

    //For each Gauss point, we compute the new stress tensor
    // Distinction has to be made depending if the points are ELASTIC or
    // POST-PLASTIC.
    // If one point enters a PLASTIC state, the computation is stopped, and
    // computePlasticForce is called, to update properly the points' mechanical
    // states
    for (int gaussPointIt = 0; gaussPointIt < 27; gaussPointIt++)
    {
        yieldStress = m_beamsData.getValue()[index]._localYieldStresses[gaussPointIt];
        Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];

        if (pointMechanicalState[gaussPointIt] == ELASTIC)
            newStress = C*Be*eigenDepl;
        else //POST-PLASTIC
        {
            VoigtTensor2 elasticStrain = Be*eigenDepl;
            VoigtTensor2 plasticStrain = plasticStrainHistory[gaussPointIt];
            newStress = C*(elasticStrain - plasticStrain);
        }

        // Checking if the deformation becomes plastic
        bool isNewPlastic = goToPlastic(newStress, yieldStress);
        if (isNewPlastic)
        {
            // If a point is detected as entering a PLASTIC state, we stop
            // the computation and call computePlasticForce instead.
            // The computation of the internal forces will thus be carried out
            // incrementally, which will change nothing for the points remaining
            // in an ELASTIC or POSTPLASTIC state, but will allow the new
            // PLASTIC points to be handled correctly.
            computePlasticForce(internalForces, x, index, a, b);
            helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
            MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
            beamMechanicalState = PLASTIC;
            m_beamsData.endEdit();
            return;
        }

        newStresses[gaussPointIt] = newStress;
    }

    // Here, all Gauss points remained in the same mechanical state (either
    // ELASTIC or POSTPLASTIC). Otherwise the method execution would have
    // been stopped by a call to computePlasticForce.

    // Storing the new stresses for the next time step, in case plasticity occurs.
    m_prevStresses[index] = newStresses;

    // Computation of the resulting internal forces, using Gaussian reduced integration.
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    int gaussPointIt = 0;
    LambdaType computePostPlastic = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];

        internalForces += (w1*w2*w3)*beTTensor2Mult(Be.transpose(), newStresses[gaussPointIt]);

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[index]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computePostPlastic);
}

template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeStressIncrement(int index,
                                                            int gaussPointIt,
                                                            const VoigtTensor2 &initialStress,
                                                            VoigtTensor2 &newStressPoint,
                                                            const VoigtTensor2 &strainIncrement,
                                                            double &lambdaIncrement,
                                                            MechanicalState &pointMechanicalState,
                                                            const Displacement &currentDisp)
{
    /** Material point iterations **/
    //NB: we consider that the yield function and the plastic flow are equal (f=g)
    //    This corresponds to an associative flow rule (for plasticity)

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour; //Matrix D in Krabbenhoft's
    const Real yieldStress = m_beamsData.getValue()[index]._localYieldStresses[gaussPointIt];

    //First we compute the elastic predictor
    VoigtTensor2 elasticIncrement = C*strainIncrement;
    VoigtTensor2 currentStressPoint = initialStress + elasticIncrement;

    if (d_useConsistentTangentOperator.getValue())
        m_elasticPredictors[index][gaussPointIt] = currentStressPoint;


    /***************** Determination of the mechanical state *****************/

    if (pointMechanicalState == ELASTIC)
    {
        //If the point is still in elastic state, we have to check if the
        //deformation becomes plastic

        bool isNewPlastic = goToPlastic(initialStress + elasticIncrement, yieldStress);

        if (!isNewPlastic)
        {
            newStressPoint = currentStressPoint;
            return;
        }
        else
        {
            pointMechanicalState = PLASTIC;
            //The new point is then computed plastically below
        }
    }

    else if (pointMechanicalState == POSTPLASTIC)
    {
        //We take into account the plastic history
        const helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = m_beamsData.getValue()[index]._plasticStrainHistory;
        const Eigen::Matrix<double, 6, 12> &Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];

        EigenDisplacement eigenCurrentDisp;
        for (int k = 0; k < 12; k++)
            eigenCurrentDisp(k) = currentDisp[k];

        VoigtTensor2 elasticStrain = Be*eigenCurrentDisp;
        VoigtTensor2 plasticStrain = plasticStrainHistory[gaussPointIt];

        currentStressPoint = C*(elasticStrain - plasticStrain);

        //If the point is still in elastic state, we have to check if the
        //deformation becomes plastic
        bool isNewPlastic = goToPlastic(currentStressPoint, yieldStress);

        if (!isNewPlastic)
        {
            newStressPoint = currentStressPoint;
            return;
        }
        else
        {
            pointMechanicalState = PLASTIC;
            //The new point is then computed plastically below
        }
    }

    else
    {
        bool isPostPlastic;
        isPostPlastic = goToPostPlastic(initialStress, elasticIncrement);

        if (isPostPlastic)
        {
            //The new computed stress for this Gauss point doesn't correspond
            // anymore to plastic deformation. We must re-compute it in an
            // elastic way, while taking into account the plastic deformation
            // history.

            pointMechanicalState = POSTPLASTIC;
            newStressPoint = currentStressPoint;

            // Recomputation of the stress, taking off the plastic strain history
            const Eigen::Matrix<double, 6, 12> &Be = m_beamsData.getValue()[index]._BeMatrices[gaussPointIt];
            const helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = m_beamsData.getValue()[index]._plasticStrainHistory;

            EigenDisplacement eigenCurrentDisp;
            for (int k = 0; k < 12; k++)
                eigenCurrentDisp(k) = currentDisp[k];

            VoigtTensor2 elasticStrain = Be*eigenCurrentDisp;
            VoigtTensor2 plasticStrain = plasticStrainHistory[gaussPointIt];

            currentStressPoint = C*(elasticStrain - plasticStrain);
            newStressPoint = currentStressPoint;

            return;
        }
    }

    /*****************  Plastic stress increment computation *****************/

    /* For a perfectly plastic material, the assumption is made that during
    * the plastic phase, the deformation is entirely plastic. In this case,
    * all the corresponding deformation energy is dissipated in the form of
    * plastic strain (i.e. during a plastic deformation, the elastic part
    * of the strain is null).
    */

    /**** Litterature implementation ****/
    // Ref: Theoretical foundation for large scale computations for nonlinear
    // material behaviour, Hugues (et al) 1984

    //Computation of the deviatoric stress tensor
    VoigtTensor2 elasticPredictor = currentStressPoint;
    double meanStress = (1.0 / 3)*(elasticPredictor[0] + elasticPredictor[1] + elasticPredictor[2]);
    VoigtTensor2 elasticDeviatoricStress = elasticPredictor;
    elasticDeviatoricStress[0] -= meanStress;
    elasticDeviatoricStress[1] -= meanStress;
    elasticDeviatoricStress[2] -= meanStress;

    double sigmaEq = equivalentStress(elasticPredictor);
    double sqrtA = helper::rsqrt(2.0 / 3) * sigmaEq;
    double R = helper::rsqrt(2.0 / 3) * yieldStress;

    // Computing the new stress
    VoigtTensor2 voigtIdentityTensor = VoigtTensor2::Zero();
    voigtIdentityTensor[0] = 1;
    voigtIdentityTensor[1] = 1;
    voigtIdentityTensor[2] = 1;

    newStressPoint = (R / sqrtA)*elasticDeviatoricStress + meanStress*voigtIdentityTensor;

    // Updating the plastic strain
    VoigtTensor2 yieldNormal = helper::rsqrt(3.0 / 2)*(1 / sigmaEq)*elasticDeviatoricStress;

    double lambda = voigtDotProduct(yieldNormal, strainIncrement);

    VoigtTensor2 plasticStrainIncrement = lambda*yieldNormal;
    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = bd[index]._plasticStrainHistory;
    plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;
    m_beamsData.endEdit();
}


template<class DataTypes>
Eigen::Matrix<double, 9, 1> BeamPlasticFEMForceField<DataTypes>::voigtToVect2(const VoigtTensor2 &voigtTensor)
{
    // This function aims at vectorising a second-order tensor taking into account
    // all 9 elements of the tensor. The result is thus a 9x1 vector (contrarily
    // to the Voigt vector representation which is only 6x1).
    // By default, we order the elements line by line, so that:
    // res[0] = T_11, res[1] = T_12, res[2] = T_13
    // res[3] = T_12, res[4] = T_22, res[5] = T_23
    // res[6] = T_13, res[7] = T_23, res[8] = T_33
    // where T is the second-order (9 element) tensor in matrix form.

    VectTensor2 res = VectTensor2::Zero();

    res[0] = voigtTensor[0];
    res[4] = voigtTensor[1];
    res[8] = voigtTensor[2];
    res[5] = res[7] = voigtTensor[3];
    res[2] = res[6] = voigtTensor[4];
    res[1] = res[3] = voigtTensor[5];

    return res;
}


template<class DataTypes>
Eigen::Matrix<double, 9, 9> BeamPlasticFEMForceField<DataTypes>::voigtToVect4(const VoigtTensor4 &voigtTensor)
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

    VectTensor4 res = VectTensor4::Zero();

    // 1st row
    res(0, 0) = voigtTensor(0, 0);
    res(0, 4) = voigtTensor(0, 1);
    res(0, 8) = voigtTensor(0, 2);
    res(0, 1) = res(0, 3) = voigtTensor(0, 5);
    res(0, 2) = res(0, 6) = voigtTensor(0, 4);
    res(0, 5) = res(0, 7) = voigtTensor(0, 3);

    // 5th row
    res(4, 0) = voigtTensor(1, 0);
    res(4, 4) = voigtTensor(1, 1);
    res(4, 8) = voigtTensor(1, 2);
    res(4, 1) = res(4, 3) = voigtTensor(1, 5);
    res(4, 2) = res(4, 6) = voigtTensor(1, 4);
    res(4, 5) = res(4, 7) = voigtTensor(1, 3);

    // 9th row
    res(8, 0) = voigtTensor(2, 0);
    res(8, 4) = voigtTensor(2, 1);
    res(8, 8) = voigtTensor(2, 2);
    res(8, 1) = res(8, 3) = voigtTensor(2, 5);
    res(8, 2) = res(8, 6) = voigtTensor(2, 4);
    res(8, 5) = res(8, 7) = voigtTensor(2, 3);

    // 2nd and 4th rows
    res(3, 0) = res(1, 0) = voigtTensor(5, 0);
    res(3, 4) = res(1, 4) = voigtTensor(5, 1);
    res(3, 8) = res(1, 8) = voigtTensor(5, 2);
    res(3, 1) = res(3, 3) = res(1, 1) = res(1, 3) = voigtTensor(5, 5);
    res(3, 2) = res(3, 6) = res(1, 2) = res(1, 6) = voigtTensor(5, 4);
    res(3, 5) = res(3, 7) = res(1, 5) = res(1, 7) = voigtTensor(5, 3);

    // 3rd and 7th rows
    res(6, 0) = res(2, 0) = voigtTensor(4, 0);
    res(6, 4) = res(2, 4) = voigtTensor(4, 1);
    res(6, 8) = res(2, 8) = voigtTensor(4, 2);
    res(6, 1) = res(6, 3) = res(2, 1) = res(2, 3) = voigtTensor(4, 5);
    res(6, 2) = res(6, 6) = res(2, 2) = res(2, 6) = voigtTensor(4, 4);
    res(6, 5) = res(6, 7) = res(2, 5) = res(2, 7) = voigtTensor(4, 3);

    // 6th and 8th rows
    res(5, 0) = res(7, 0) = voigtTensor(3, 0);
    res(5, 4) = res(7, 4) = voigtTensor(3, 1);
    res(5, 8) = res(7, 8) = voigtTensor(3, 2);
    res(5, 1) = res(5, 3) = res(7, 1) = res(7, 3) = voigtTensor(3, 5);
    res(5, 2) = res(5, 6) = res(7, 2) = res(7, 6) = voigtTensor(3, 4);
    res(5, 5) = res(5, 7) = res(7, 5) = res(7, 7) = voigtTensor(3, 3);

    return res;
}


template<class DataTypes>
Eigen::Matrix<double, 6, 1> BeamPlasticFEMForceField<DataTypes>::vectToVoigt2(const VectTensor2 &vectTensor)
{
    // This function aims at reducing the expression of a second-order tensor
    // using Voigt notation. The tensor is initially expressed in vector form
    // as:
    // res[0] = T_11, res[1] = T_12, res[2] = T_13
    // res[3] = T_12, res[4] = T_22, res[5] = T_23
    // res[6] = T_13, res[7] = T_23, res[8] = T_33
    // where T is the second-order (9 element) tensor in matrix form.

    VoigtTensor2 res = VoigtTensor2::Zero();

    res[0] = vectTensor[0];
    res[1] = vectTensor[4];
    res[2] = vectTensor[8];
    res[3] = vectTensor[5];
    res[4] = vectTensor[2];
    res[5] = vectTensor[1];

    return res;
}


template<class DataTypes>
Eigen::Matrix<double, 6, 6> BeamPlasticFEMForceField<DataTypes>::vectToVoigt4(const VectTensor4 &vectTensor)
{
    // This function aims at reducing the expression of a fourth-order tensor
    // using Voigt notation. The tensor is initially expressed in vector form
    // as:
    // res(0, *) = (T_1111, T_1112, T_1113, T_1112, T_1122, T_1123, T_1113, T_1123, T_1133)
    // res(1, *) = (T_1211, T_1212, T_1213, T_1212, T_1222, T_1223, T_1213, T_1223, T_1233)
    // etc.
    // where T is the fourth-order (81 element) tensor in matrix form.

    VoigtTensor4 res = VoigtTensor4::Zero();

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


// Hugues implementation (perfectly plastic and mixed hardening)
template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeForceWithHardening(Eigen::Matrix<double, 12, 1> &internalForces,
    const VecCoord& x, int index, Index a, Index b)
{
    // Computes displacement increment, from last system solution
    Displacement currentDisp;
    Displacement lastDisp;
    Displacement dispIncrement;
    computeDisplacementIncrement(x, m_lastPos, currentDisp, lastDisp, dispIncrement, index, a, b);

    // Converts to Eigen data structure
    EigenDisplacement displacementIncrement;
    for (int k = 0; k < 12; k++)
        displacementIncrement(k) = dispIncrement[k];

    //All the rest of the force computation is made inside of the lambda function
    //as the stress and strain are computed for each Gauss point

    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;

    VoigtTensor2 initialStressPoint = VoigtTensor2::Zero();
    VoigtTensor2 strainIncrement = VoigtTensor2::Zero();
    VoigtTensor2 newStressPoint = VoigtTensor2::Zero();

    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[index]._pointMechanicalState;
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

        isPlasticBeam = isPlasticBeam || (mechanicalState == PLASTIC);

        m_prevStresses[index][gaussPointIt] = newStressPoint;

        internalForces += (w1*w2*w3)*beTTensor2Mult(Be.transpose(), newStressPoint);

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[index]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeStress);

    // Updates the beam mechanical state information
    if (!isPlasticBeam)
    {
        MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
        beamMechanicalState = POSTPLASTIC;
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

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour; //Matrix D in Krabbenhoft's

    /***************************************************/
    /*      Radial return with hardening - Hugues      */
    /***************************************************/

    //First we compute the trial stress, taking into account the back stress
    // (i.e. the centre of the yield surface)

    VoigtTensor2 elasticIncrement = C*strainIncrement;
    VoigtTensor2 trialStress = lastStress + elasticIncrement;

    if (d_useConsistentTangentOperator.getValue())
        m_elasticPredictors[index][gaussPointIt] = trialStress;

    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());

    helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &backStresses = bd[index]._backStresses;
    Eigen::Matrix<double, 6, 1> &backStress = backStresses[gaussPointIt];

    helper::fixed_array<Real, 27> &localYieldStresses = bd[index]._localYieldStresses;
    Real &yieldStress = localYieldStresses[gaussPointIt];

    if (!goToPlastic(trialStress - backStress, yieldStress))
    {
        // The Gauss point is in elastic state: the back stress and yield stress
        // remain constant, and the new stress is equal to the trial stress.
        newStressPoint = trialStress;

        // If the Gauss point was initially plastic, we update its mechanical state
        if (pointMechanicalState == PLASTIC)
            pointMechanicalState = POSTPLASTIC;
    }
    else
    {
        // If the Gauss point was initially elastic, we update its mechanical state
        if (pointMechanicalState == POSTPLASTIC || pointMechanicalState == ELASTIC)
            pointMechanicalState = PLASTIC;

        /*******************************************/
        /*             Explicit method             */
        /*******************************************/
        {
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

            //helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit()); //Done in the beginning to modify the yield and back stresses
            helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = bd[index]._plasticStrainHistory;
            VoigtTensor2 plasticStrainIncrement = helper::rsqrt(3.0/2.0)*plasticMultiplier*finalN;
            plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;
            //m_beamsData.endEdit();

            helper::fixed_array<Real, 27> &effectivePlasticStrain = bd[index]._effectivePlasticStrains;
            effectivePlasticStrain[gaussPointIt] += plasticMultiplier;

            m_beamsData.endEdit(); //end edit _backStresses, _localYieldStresses, _plasticStrainHistory, and _effectivePlasticStrains
        }

        /*******************************************/
        /*             Implicit method             */
        /*******************************************/
        {
            //***** Constructing the system for the first iteration *****/

            //helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());

            //helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &backStresses = bd[index]._backStresses;
            //Eigen::Matrix<double, 6, 1> &backStress = backStresses[gaussPointIt];
            //helper::fixed_array<Real, 27> &localYieldStresses = bd[index]._localYieldStresses;
            //Real &yieldStress = localYieldStresses[gaussPointIt];

            //const VectTensor4 vectC = voigtToVect4(C);
            //const double E = m_beamsData.getValue()[index]._E;
            //const double beta = 0.5;

            //VoigtTensor2 currentStress = trialStress;
            //VectTensor2 vectCurrentStress = voigtToVect2(currentStress);

            //double yieldCondition = vonMisesYield(currentStress - backStress, yieldStress);

            //// Variables
            //Eigen::Matrix<double, 10, 1> newIncrement = Eigen::Matrix<double, 10, 1>::Zero();
            //Eigen::Matrix<double, 9, 1> vectLastStress = voigtToVect2(lastStress);
            //Eigen::Matrix<double, 10, 1> totalIncrement = Eigen::Matrix<double, 10, 1>::Zero();
            //totalIncrement.block<9, 1>(0, 0) = voigtToVect2(elasticIncrement);
            //double lambda = 0.0;

            //// Shifted stress point
            //VoigtTensor2 shiftedStress = trialStress - backStress; // At 1st step, lambda=0

            //// Normal
            //VoigtTensor2 trialNormal = helper::rsqrt(2.0 / 3.0)*vonMisesGradient(trialStress);
            //VectTensor2 vectTrialNormal = voigtToVect2(trialNormal);
            //VoigtTensor2 shiftedGradient = vonMisesGradient(shiftedStress);
            //VectTensor2 vectShiftedGradient = voigtToVect2(shiftedGradient);

            //// Plastic modulus
            //double plasticModulus = computeConstPlasticModulus();
            //
            //// Jacobian
            //Eigen::Matrix<double, 10, 10> J = Eigen::Matrix<double, 10, 10>::Zero();
            //Eigen::Matrix<double, 9, 9> I9 = Eigen::Matrix<double, 9, 9>::Identity();
            //
            //J.block<9, 9>(0, 0) = I9; // At 1st step, lambda=0
            //J.block<1, 9>(9, 0) = vectShiftedGradient.transpose()*I9; // At 1st step, lambda=0
            //J.block<9, 1>(0, 9) = vectC*vectTrialNormal;
            //double scalarProduct = vectTrialNormal.transpose()*vectShiftedGradient;
            //J(9, 9) = -(2.0/3.0)*(1-beta)*plasticModulus*scalarProduct - helper::rsqrt(2.0 / 3.0)*beta*plasticModulus;

            //// Second member b
            //Eigen::Matrix<double, 10, 1> b = Eigen::Matrix<double, 10, 1>::Zero();
            //// b.block<9, 1>(0, 0) is zero
            //b(9, 0) = equivalentStress(shiftedStress) - yieldStress;

            //// Solver
            //Eigen::FullPivLU<Eigen::Matrix<double, 10, 10> > LU(J.rows(), J.cols());
            //LU.compute(J);

            //// First iteration
            //newIncrement = LU.solve(-b);

            //// Updating the variables
            //totalIncrement += newIncrement;
            //lambda = totalIncrement[9];
            //vectCurrentStress += newIncrement.block<9, 1>(0, 0);
            //currentStress = vectToVoigt2(vectCurrentStress);

            //yieldStress += helper::rsqrt(2.0 / 3.0)*beta*plasticModulus*lambda;
            //backStress += (2.0 / 3.0)*(1 - beta)*plasticModulus*lambda*helper::rsqrt(2.0/3.0)*vonMisesGradient(currentStress);

            ////Testing the consistency condition
            //yieldCondition = vonMisesYield(currentStress - backStress, yieldStress);

            //// Testing if the result of the first iteration is satisfaying
            //double threshold = 1.0; //TO DO: choose coherent value
            //bool consistencyTestIsPositive = helper::rabs(yieldCondition) <= threshold;

            //// Declaration of variables for the iterations
            //VoigtTensor2 currentNormal = helper::rsqrt(2.0 / 3.0)*vonMisesGradient(currentStress);

            //if (!consistencyTestIsPositive)
            ////if (false)
            //{
            //    /* If the new stress point computed after one iteration of the implicit
            //    * method satisfied the consistency condition, we could stop the
            //    * iterative procedure at this point.
            //    * Otherwise the solution found with the first iteration does not
            //    * satisfy the consistency condition. In this case, we need to go
            //    * through more iterations to find a more correct solution.
            //    */
            //    unsigned int nbMaxIterations = 100;

            //    // Updates for next iteration
            //    currentNormal = helper::rsqrt(2.0 / 3.0)*vonMisesGradient(currentStress);
            //    VectTensor2 vectCurrentNormal = voigtToVect2(currentNormal);
            //    shiftedStress = currentStress - backStress; // both backStress and currentStress already updated
            //    shiftedGradient = vonMisesGradient(shiftedStress);
            //    vectShiftedGradient = voigtToVect2(shiftedGradient);
            //    VectTensor4 vectHessian = vonMisesHessian(currentStress, yieldStress);

            //    unsigned int count = 1;
            //    while (helper::rabs(yieldCondition) >= threshold && count < nbMaxIterations)
            //    {
            //        //Updates J and b
            //        J.block<9, 9>(0, 0) = I9 + helper::rsqrt(2.0 / 3.0)*lambda*vectC*vectHessian;
            //        J.block<1, 9>(9, 0) = vectShiftedGradient.transpose()*(I9 - (2.0/3.0)*helper::rsqrt(2.0/3.0)*(1-beta)*plasticModulus*lambda*vectHessian);
            //        J.block<9, 1>(0, 9) = vectC*vectCurrentNormal;
            //        // J(9, 9) = -helper::rsqrt(2.0 / 3.0)*beta*plasticModulus; // Optionnal with constant plastic modulus

            //        // Second member b
            //        Eigen::Matrix<double, 10, 1> b = Eigen::Matrix<double, 10, 1>::Zero();
            //        b(9, 0) = equivalentStress(shiftedStress) - yieldStress; // both backStress and yieldStress already updated
            //        b.block<9, 1>(0, 0) = totalIncrement.block<9, 1>(0, 0) - voigtToVect2(elasticIncrement) + lambda*vectC*vectCurrentNormal;

            //        //Computes the new increment
            //        LU.compute(J);
            //        newIncrement = LU.solve(-b);

            //        totalIncrement += newIncrement;

            //        // Update the variables (currentStress, backStress, yieldStress, yieldCondition, gradient and hessian)
            //        lambda = totalIncrement[9];
            //        vectCurrentStress += newIncrement.block<9, 1>(0, 0);
            //        currentStress = vectToVoigt2(vectCurrentStress);
            //        yieldStress += helper::rsqrt(2.0 / 3.0)*beta*plasticModulus*lambda;
            //        backStress += (2.0 / 3.0)*(1 - beta)*plasticModulus*lambda*helper::rsqrt(2.0 / 3.0)*vonMisesGradient(currentStress);
            //        yieldCondition = vonMisesYield(currentStress - backStress, yieldStress);

            //        // TO DO: optimise the variable update by placing it at the beginning of the loop,
            //        // and merging it with the update before the first iteration
            //        currentNormal = helper::rsqrt(2.0 / 3.0)*vonMisesGradient(currentStress);
            //        vectCurrentNormal = voigtToVect2(currentNormal);
            //        shiftedStress = currentStress - backStress;
            //        shiftedGradient = vonMisesGradient(shiftedStress);
            //        vectShiftedGradient = voigtToVect2(shiftedGradient);
            //        vectHessian = vonMisesHessian(currentStress, yieldStress);

            //        count++;
            //    }

            //} // endif (!consistencyTestIsPositive)

            //VectTensor2 vectTotalIncrement = totalIncrement.block<9, 1>(0, 0);
            //newStressPoint = lastStress + vectToVoigt2(vectTotalIncrement);

            //helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = bd[index]._plasticStrainHistory;
            //VoigtTensor2 plasticStrainIncrement = lambda*currentNormal;
            //plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;

            //m_beamsData.endEdit(); //end edit _backStresses, _localYieldStresses, _plasticStrainHistory
        }
    }
}


// TESTING : Incremental implementation for perfect plasticity
template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computeForceWithPerfectPlasticity(Eigen::Matrix<double, 12, 1> &internalForces,
    const VecCoord& x, int index, Index a, Index b)
{
    // Computes displacement increment, from last system solution
    Displacement currentDisp;
    Displacement lastDisp;
    Displacement dispIncrement;
    computeDisplacementIncrement(x, m_lastPos, currentDisp, lastDisp, dispIncrement, index, a, b);

    // Converts to Eigen data structure
    EigenDisplacement displacementIncrement;
    for (int k = 0; k < 12; k++)
        displacementIncrement(k) = dispIncrement[k];

    //All the rest of the force computation is made inside of the lambda function
    //as the stress and strain are computed for each Gauss point

    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;

    VoigtTensor2 initialStressPoint = VoigtTensor2::Zero();
    VoigtTensor2 strainIncrement = VoigtTensor2::Zero();
    VoigtTensor2 newStressPoint = VoigtTensor2::Zero();

    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[index]._pointMechanicalState;
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

        isPlasticBeam = isPlasticBeam || (mechanicalState == PLASTIC);

        m_prevStresses[index][gaussPointIt] = newStressPoint;

        internalForces += (w1*w2*w3)*beTTensor2Mult(Be.transpose(), newStressPoint);

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = m_beamsData.getValue()[index]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeStress);

    // Updates the beam mechanical state information
    if (!isPlasticBeam)
    {
        MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
        beamMechanicalState = POSTPLASTIC;
    }
    m_beamsData.endEdit();

    //Update the tangent stiffness matrix with the new computed stresses
    //This matrix will then be used in addDForce and addKToMatrix methods
    updateTangentStiffness(index, a, b);
}


template< class DataTypes>
void BeamPlasticFEMForceField<DataTypes>::computePerfectPlasticStressIncrement(int index,
                                                                          int gaussPointIt,
                                                                          const VoigtTensor2 &lastStress,
                                                                          VoigtTensor2 &newStressPoint,
                                                                          const VoigtTensor2 &strainIncrement,
                                                                          MechanicalState &pointMechanicalState)
{
    /** Material point iterations **/
    //NB: we consider that the yield function and the plastic flow are equal (f=g)
    //    This corresponds to an associative flow rule (for plasticity)

    const Eigen::Matrix<double, 6, 6>& C = m_beamsData.getValue()[index]._materialBehaviour; //Matrix D in Krabbenhoft's

    /***************************************************/
    /*  Radial return in perfect plasticity - Hugues   */
    /***************************************************/
    {
        //First we compute the trial stress, taking into account the back stress
        // (i.e. the centre of the yield surface)

        VoigtTensor2 elasticIncrement = C*strainIncrement;
        VoigtTensor2 trialStress = lastStress + elasticIncrement;

        if (d_useConsistentTangentOperator.getValue())
            m_elasticPredictors[index][gaussPointIt] = trialStress;

        helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());

        helper::fixed_array<Real, 27> &localYieldStresses = bd[index]._localYieldStresses;
        Real &yieldStress = localYieldStresses[gaussPointIt];

        VoigtTensor2 devTrialStress = deviatoricStress(trialStress);

        double A = voigtDotProduct(devTrialStress, devTrialStress);
        double R = helper::rsqrt(2.0 / 3) * yieldStress;
        const double R2 = R*R;

        if (A <= R2) //TO DO: proper comparison
        {
            // The Gauss point is in elastic state: the back stress and yield stress
            // remain constant, and the new stress is equal to the trial stress.
            newStressPoint = trialStress;

            // If the Gauss point was initially plastic, we update its mechanical state
            if (pointMechanicalState == PLASTIC)
                pointMechanicalState = POSTPLASTIC;
        }
        else
        {
            // If the Gauss point was initially elastic, we update its mechanical state
            if (pointMechanicalState == POSTPLASTIC || pointMechanicalState == ELASTIC)
                pointMechanicalState = PLASTIC;

            // We then compute the new stress

            /**** Litterature implementation ****/
            // Ref: Theoretical foundation for large scale computations for nonlinear
            // material behaviour, Hugues (et al) 1984

            double meanStress = (1.0 / 3)*(trialStress[0] + trialStress[1] + trialStress[2]);

            // Computing the new stress
            VoigtTensor2 voigtIdentityTensor = VoigtTensor2::Zero();
            voigtIdentityTensor[0] = 1;
            voigtIdentityTensor[1] = 1;
            voigtIdentityTensor[2] = 1;

            newStressPoint = (R / helper::rsqrt(A))*devTrialStress + meanStress*voigtIdentityTensor;

            // Updating the plastic strain
            VoigtTensor2 yieldNormal = helper::rsqrt(3.0 / 2)*(1.0 / equivalentStress(trialStress))*devTrialStress;

            double lambda = voigtDotProduct(yieldNormal, strainIncrement);

            VoigtTensor2 plasticStrainIncrement = lambda*yieldNormal;
            helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
            helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = bd[index]._plasticStrainHistory;
            plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;
            m_beamsData.endEdit();
        }

    }
}





/*****************************************************************************/
/*                              MISCELLANEOUS                                */
/*****************************************************************************/


inline defaulttype::Quat qDiff(defaulttype::Quat a, const defaulttype::Quat& b)
{
    if (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]<0)
    {
        a[0] = -a[0];
        a[1] = -a[1];
        a[2] = -a[2];
        a[3] = -a[3];
    }
    defaulttype::Quat q = b.inverse() * a;
    //sout << "qDiff("<<a<<","<<b<<")="<<q<<", bq="<<(b*q)<<sendl;
    return q;
}

template<class DataTypes>
defaulttype::Quat& BeamPlasticFEMForceField<DataTypes>::beamQuat(int i)
{
    helper::vector<BeamInfo>& bd = *(m_beamsData.beginEdit());
    return bd[i].quat;
}





/**************************************************************************/

} // namespace beamplasticforcefield

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BEAMPLASTICFEMFORCEFIELD_INL
