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
#ifndef SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_INL

#include <SofaBaseTopology/TopologyData.inl>
#include "MultiBeamForceField.h"
#include "RambergOsgood.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/GridTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>
#include <set>
#include <sofa/helper/system/gl.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>

#include "../StiffnessContainer.h"
#include "../PoissonContainer.h"

namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
MultiBeamForceField<DataTypes>::MultiBeamForceField()
    : beamsData(initData(&beamsData, "beamsData", "Internal element data"))
    , _indexedElements(NULL)
    , _poissonRatio(initData(&_poissonRatio,(Real)0.3f,"poissonRatio","Potion Ratio"))
    , _youngModulus(initData(&_youngModulus, (Real)5000, "youngModulus", "Young Modulus"))
    , _yieldStress(initData(&_yieldStress,(Real)6.0e8,"yieldStress","yield stress"))
    , _virtualDisplacementMethod(initData(&_virtualDisplacementMethod, true, "virtualDisplacementMethod", "indicates if the stiffness matrix is computed following the virtual displacement method"))
    , _inputLocalOrientations(initData(&_inputLocalOrientations, { defaulttype::Quat(0, 0, 0, 1) }, "beamLocalOrientations", "local orientation of each beam element"))
    , _isPlasticKrabbenhoft(initData(&_isPlasticKrabbenhoft, false, "isPlasticKrabbenhoft", "indicates wether the behaviour model is plastic, as in Krabbenhoft 2002"))
    , _isPerfectlyPlastic(initData(&_isPerfectlyPlastic, false, "isPerfectlyPlastic", "indicates wether the behaviour model is perfectly plastic"))
    , d_modelName(initData(&d_modelName, std::string("RambergOsgood"), "modelName", "the name of the 1D contitutive law model to be used in plastic deformation"))
    , _isPlasticMuller(initData(&_isPlasticMuller, false, "isPlasticMuller", "indicates wether the behaviour model is plastic, as in Muller et al 2004"))
    , _zSection(initData(&_zSection, (Real)0.2, "zSection", "length of the section in the z direction for rectangular beams"))
    , _ySection(initData(&_ySection, (Real)0.2, "ySection", "length of the section in the y direction for rectangular beams"))
    , _useSymmetricAssembly(initData(&_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , _isTimoshenko(initData(&_isTimoshenko,false,"isTimoshenko","implements a Timoshenko beam model"))
    , _updateStiffnessMatrix(true)
    , _assembling(false)
    , edgeHandler(NULL)
{
    edgeHandler = new BeamFFEdgeHandler(this, &beamsData);

    _poissonRatio.setRequired(true);
    _youngModulus.setReadOnly(true);
}

template<class DataTypes>
MultiBeamForceField<DataTypes>::MultiBeamForceField(Real poissonRatio, Real youngModulus, Real yieldStress, Real zSection,
                                                    Real ySection, bool useVD, bool isPlasticMuller, bool isTimoshenko,
                                                    bool isPlasticKrabbenhoft, bool isPerfectlyPlastic,
                                                    helper::vector<defaulttype::Quat> localOrientations)
    : beamsData(initData(&beamsData, "beamsData", "Internal element data"))
    , _indexedElements(NULL)
    , _poissonRatio(initData(&_poissonRatio,(Real)poissonRatio,"poissonRatio","Potion Ratio"))
    , _youngModulus(initData(&_youngModulus,(Real)youngModulus,"youngModulus","Young Modulus"))
    , _yieldStress(initData(&_yieldStress, (Real)yieldStress, "yieldStress", "yield stress"))
    , _virtualDisplacementMethod(initData(&_virtualDisplacementMethod, true, "virtualDisplacementMethod", "indicates if the stiffness matrix is computed following the virtual displacement method"))
    , _inputLocalOrientations(initData(&_inputLocalOrientations, localOrientations, "beamLocalOrientations", "local orientation of each beam element"))
    , _isPlasticKrabbenhoft(initData(&_isPlasticKrabbenhoft, false, "isPlasticKrabbenhoft", "indicates wether the behaviour model is plastic, as in Krabbenhoft 2002"))
    , _isPerfectlyPlastic(initData(&_isPerfectlyPlastic, false, "isPerfectlyPlastic", "indicates wether the behaviour model is perfectly plastic"))
    , d_modelName(initData(&d_modelName, std::string("RambergOsgood"), "modelName", "the name of the 1D contitutive law model to be used in plastic deformation"))
    , _isPlasticMuller(initData(&_isPlasticMuller, false, "isPlasticMuller", "indicates wether the behaviour model is plastic, as in Muller et al 2004"))
    , _zSection(initData(&_zSection, (Real)zSection, "zSection", "length of the section in the z direction for rectangular beams"))
    , _ySection(initData(&_ySection, (Real)ySection, "ySection", "length of the section in the y direction for rectangular beams"))
    , _useSymmetricAssembly(initData(&_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , _isTimoshenko(initData(&_isTimoshenko, isTimoshenko, "isTimoshenko", "implements a Timoshenko beam model"))
    , _updateStiffnessMatrix(true)
    , _assembling(false)
    , edgeHandler(NULL)
{
    edgeHandler = new BeamFFEdgeHandler(this, &beamsData);

    _poissonRatio.setRequired(true);
    _youngModulus.setReadOnly(true);
}

template<class DataTypes>
MultiBeamForceField<DataTypes>::~MultiBeamForceField()
{
    if(edgeHandler) delete edgeHandler;
}

template <class DataTypes>
void MultiBeamForceField<DataTypes>::bwdInit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);
    matS.resize(state->getMatrixSize(),state->getMatrixSize());
    lastUpdatedStep=-1.0;
}



template <class DataTypes>
void MultiBeamForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    _topology = context->getMeshTopology();

    stiffnessContainer = context->core::objectmodel::BaseContext::get<container::StiffnessContainer>();
    poissonContainer = context->core::objectmodel::BaseContext::get<container::PoissonContainer>();

    if (_virtualDisplacementMethod.getValue())
    {
        _VDPlasticYieldThreshold = (Real)0.1f;
        _VDPlasticCreep = (Real)0.7f;
    }
    else
    {
        _VFPlasticYieldThreshold = { (Real)0.0001f, (Real)0.0001f, (Real)0.0001f, (Real)0.0001f, (Real)0.0001f, (Real)0.0001f };
        _VFPlasticMaxThreshold = (Real)0.f;
        _VFPlasticCreep = (Real)0.1f;
    }

    // Retrieving the 1D plastic constitutive law model
    std::string constitutiveModel = d_modelName.getValue();
    if (constitutiveModel == "RambergOsgood")
    {
        Real youngModulus = _youngModulus.getValue();
        Real yieldStress = _yieldStress.getValue();
        fem::RambergOsgood<DataTypes> *RambergOsgoodModel = new (fem::RambergOsgood<DataTypes>)(youngModulus, yieldStress);
        m_ConstitutiveLaw = RambergOsgoodModel;
        if (this->f_printLog.getValue())
            msg_info() << "The model is " << constitutiveModel;
    }
    else
    {
        msg_error() << "constitutive law model name " << constitutiveModel << " is not valid (should be RambergOsgood)";
    }


    if (_topology==NULL)
    {
        serr << "ERROR(MultiBeamForceField): object must have a BaseMeshTopology (i.e. EdgeSetTopology or MeshTopology)."<<sendl;
        return;
    }
    else
    {
        if(_topology->getNbEdges()==0)
        {
            serr << "ERROR(MultiBeamForceField): topology is empty."<<sendl;
            return;
        }
        _indexedElements = &_topology->getEdges();
    }

    beamsData.createTopologicalEngine(_topology,edgeHandler);
    beamsData.registerTopologicalData();

    reinit();
}

template <class DataTypes>
void MultiBeamForceField<DataTypes>::reinit()
{
    size_t n = _indexedElements->size();
    _forces.resize( this->mstate->getSize() );

    if (_virtualDisplacementMethod.getValue())
    {
        //Initialises the lastPos field with the rest position
        _lastPos = mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        /***** Krabbenhoft plasticity *****/
        _NRThreshold = 0.0; //to be changed during iterations
        _NRMaxIterations = 25;

        /***** Local orientations handling *****/
        size_t nbQuat = _inputLocalOrientations.getValue().size();
        _beamLocalOrientations.resize(n);
        if (nbQuat != n)
        {
            //The vector of orientations given by the user doesn't match the topology
            //TO DO: for now we set all of them with the neutral quaternion, but this
            //       should be computed automatically from topology
            for (size_t i = 0; i < n; i++)
                _beamLocalOrientations[i] = defaulttype::Quat(0, 0, 0, 1);
        }
        else
        {
            for (size_t i = 0; i < n; i++)
                _beamLocalOrientations[i] = _inputLocalOrientations.getValue()[i];
        }



        _prevStresses.resize(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 27; j++)
                _prevStresses[i][j] = VoigtTensor::Zero();

        _VDPlasticStrains.resize(n);
        for (unsigned int i = 0; i < n; i++)
            _VDPlasticStrains[i] = elementPlasticStrain::Zero();
    }
    else
    {
        _VFPlasticStrains.resize(n);
        _VFTotalStrains.resize(n);
        _nodalForces.resize(n);
        _plasticZones.resize(n);
        _isPlasticZoneComplete.resize(n);
    }

    initBeams( n );
    for (unsigned int i=0; i<n; ++i)
        reinitBeam(i);
    msg_info() << "reinit OK, "<<n<<" elements." ;
}

template <class DataTypes>
void MultiBeamForceField<DataTypes>::reinitBeam(unsigned int i)
{
    double stiffness, yieldStress, length, poisson, zSection, ySection;
    Index a = (*_indexedElements)[i][0];
    Index b = (*_indexedElements)[i][1];

    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    if (stiffnessContainer)
        stiffness = stiffnessContainer->getStiffness(i) ;
    else
        stiffness =  _youngModulus.getValue() ;

    yieldStress = _yieldStress.getValue();
    length = (x0[a].getCenter()-x0[b].getCenter()).norm() ;

    zSection = _zSection.getValue();
    ySection = _ySection.getValue();
    poisson = _poissonRatio.getValue();

    setBeam(i, stiffness, yieldStress, length, poisson, zSection, ySection);

    if (_virtualDisplacementMethod.getValue())
    {
        //In first step, we assume elastic deformation
        computeMaterialBehaviour(i, a, b);
        computeVDStiffness(i, a, b);
    }
    else
    {
        computeStiffness(i, a, b);

        if (_isPlasticMuller.getValue())
            initPlasticityMatrix(i, a, b);
    }

    // Initialisation of the beam element orientation
    //TO DO: is necessary ?
    beamQuat(i) = x0[a].getOrientation();
    beamQuat(i).normalize();

    beamsData.endEdit();

}

template< class DataTypes>
void MultiBeamForceField<DataTypes>::BeamFFEdgeHandler::applyCreateFunction(unsigned int edgeIndex, BeamInfo &ei, const core::topology::BaseMeshTopology::Edge &, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if(ff)
    {
        ff->reinitBeam(edgeIndex);
        ei = ff->beamsData.getValue()[edgeIndex];
    }
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & /*dataV*/ )
{
    VecDeriv& f = *(dataF.beginEdit());
    const VecCoord& p=dataX.getValue();
    f.resize(p.size());

    //// First compute each node rotation
    typename VecElement::const_iterator it;

    unsigned int i;

    for (it = _indexedElements->begin(), i = 0; it != _indexedElements->end(); ++it, ++i)
    {

        Index a = (*it)[0];
        Index b = (*it)[1];

        // The choice of computational method (elastic, plastic, or post-plastic)
        // is made in accumulateNonLinearForce
        accumulateNonLinearForce(f, p, i, a, b);
    }

    // Save the current positions as a record for the next time step.
    // This has to be done after the call to accumulateForceLarge (otherwise
    // the current position will be used instead in the computation)
    //TO DO: check if this is copy operator
    _lastPos = p;

    dataF.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams *mparams, DataVecDeriv& datadF , const DataVecDeriv& datadX)
{
    VecDeriv& df = *(datadF.beginEdit());
    const VecDeriv& dx=datadX.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());

    typename VecElement::const_iterator it;
    unsigned int i = 0;
    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];

        const bool beamMechanicalState = beamsData.getValue()[i]._beamMechanicalState;

        if (!beamMechanicalState)
            applyStiffnessLarge(df, dx, i, a, b, kFactor);
        else
            applyNonLinearStiffness(df, dx, i, a, b, kFactor);
    }

    datadF.endEdit();
}

template<class DataTypes>
typename MultiBeamForceField<DataTypes>::Real MultiBeamForceField<DataTypes>::peudo_determinant_for_coef ( const defaulttype::Mat<2, 3, Real>&  M )
{
    return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::computeStiffness(int i, Index , Index )
{
    Real   phiy, phiz;
    Real _L = (Real)beamsData.getValue()[i]._L;
    Real _A = (Real)beamsData.getValue()[i]._A;
    Real _nu = (Real)beamsData.getValue()[i]._nu;
    Real _E = (Real)beamsData.getValue()[i]._E;
    Real _Iy = (Real)beamsData.getValue()[i]._Iy;
    Real _Iz = (Real)beamsData.getValue()[i]._Iz;
    Real _Asy = (Real)beamsData.getValue()[i]._Asy;
    Real _Asz = (Real)beamsData.getValue()[i]._Asz;
    Real _G = (Real)beamsData.getValue()[i]._G;
    Real _J = (Real)beamsData.getValue()[i]._J;
    Real L2 = (Real) (_L * _L);
    Real L3 = (Real) (L2 * _L);
    Real EIy = (Real)(_E * _Iy);
    Real EIz = (Real)(_E * _Iz);

    // Find shear-deformation parameters
    if (_Asy == 0)
        phiy = 0.0;
    else
        phiy = (Real)(24.0*(1.0+_nu)*_Iz/(_Asy*L2));

    if (_Asz == 0)
        phiz = 0.0;
    else
        phiz = (Real)(24.0*(1.0+_nu)*_Iy/(_Asz*L2));
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    StiffnessMatrix& k_loc = bd[i]._k_loc;

    // Define stiffness matrix 'k' in local coordinates
    k_loc.clear();
    k_loc[6][6]   = k_loc[0][0]   = _E*_A/_L;
    k_loc[7][7]   = k_loc[1][1]   = (Real)(12.0*EIz/(L3*(1.0+phiy)));
    k_loc[8][8]   = k_loc[2][2]   = (Real)(12.0*EIy/(L3*(1.0+phiz)));
    k_loc[9][9]   = k_loc[3][3]   = _G*_J/_L;
    k_loc[10][10] = k_loc[4][4]   = (Real)((4.0+phiz)*EIy/(_L*(1.0+phiz)));
    k_loc[11][11] = k_loc[5][5]   = (Real)((4.0+phiy)*EIz/(_L*(1.0+phiy)));

    k_loc[4][2]   = (Real)(-6.0*EIy/(L2*(1.0+phiz)));
    k_loc[5][1]   = (Real)( 6.0*EIz/(L2*(1.0+phiy)));
    k_loc[6][0]   = -k_loc[0][0];
    k_loc[7][1]   = -k_loc[1][1];
    k_loc[7][5]   = -k_loc[5][1];
    k_loc[8][2]   = -k_loc[2][2];
    k_loc[8][4]   = -k_loc[4][2];
    k_loc[9][3]   = -k_loc[3][3];
    k_loc[10][2]  = k_loc[4][2];
    k_loc[10][4]  = (Real)((2.0-phiz)*EIy/(_L*(1.0+phiz)));
    k_loc[10][8]  = -k_loc[4][2];
    k_loc[11][1]  = k_loc[5][1];
    k_loc[11][5]  = (Real)((2.0-phiy)*EIz/(_L*(1.0+phiy)));
    k_loc[11][7]  = -k_loc[5][1];

    for (int i=0; i<=10; i++)
        for (int j=i+1; j<12; j++)
            k_loc[i][j] = k_loc[j][i];

    //DEBUG
    //std::cout << "k_loc pour l'element " << i << " : " << std::endl;
    //for (int i = 0; i < 12; i++)
    //{
    //    for (int j = 0; j < 12; j++)
    //    {
    //        std::cout << k_loc[i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl << std::endl;

    beamsData.endEdit();
}

inline defaulttype::Quat qDiff(defaulttype::Quat a, const defaulttype::Quat& b)
{
    if (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]<0)
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
void MultiBeamForceField<DataTypes>::accumulateForceLarge( VecDeriv& f, const VecCoord & x, int i, Index a, Index b )
{
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    beamQuat(i)= x[a].getOrientation();
    beamQuat(i).normalize();

    beamsData.endEdit();

    defaulttype::Vec<3,Real> u, P1P2, P1P2_0;
    // local displacement
    Displacement depl;

    // translations //
    P1P2_0 = x0[b].getCenter() - x0[a].getCenter();
    P1P2_0 = x0[a].getOrientation().inverseRotate(P1P2_0);
    P1P2 = x[b].getCenter() - x[a].getCenter();
    P1P2 = x[a].getOrientation().inverseRotate(P1P2);
    u = P1P2 - P1P2_0;

    depl[0] = 0.0; 	depl[1] = 0.0; 	depl[2] = 0.0;
    depl[6] = u[0]; depl[7] = u[1]; depl[8] = u[2];

    // rotations //
    defaulttype::Quat dQ0, dQ;

    // dQ = QA.i * QB ou dQ = QB * QA.i() ??
    dQ0 = qDiff(x0[b].getOrientation(), x0[a].getOrientation()); // x0[a].getOrientation().inverse() * x0[b].getOrientation();
    dQ =  qDiff(x[b].getOrientation(), x[a].getOrientation()); // x[a].getOrientation().inverse() * x[b].getOrientation();
    //u = dQ.toEulerVector() - dQ0.toEulerVector(); // Consider to use quatToRotationVector instead of toEulerVector to have the rotation vector

    dQ0.normalize();
    dQ.normalize();

    defaulttype::Quat tmpQ = qDiff(dQ,dQ0);
    tmpQ.normalize();

    u = tmpQ.quatToRotationVector(); //dQ.quatToRotationVector() - dQ0.quatToRotationVector();  // Use of quatToRotationVector instead of toEulerVector:
                                                                                                // this is done to keep the old behavior (before the
                                                                                                // correction of the toEulerVector  function). If the
                                                                                                // purpose was to obtain the Eulerian vector and not the
                                                                                                // rotation vector please use the following line instead
    //u = tmpQ.toEulerVector(); //dQ.toEulerVector() - dQ0.toEulerVector();

    depl[3] = 0.0; 	depl[4] = 0.0; 	depl[5] = 0.0;
    depl[9] = u[0]; depl[10]= u[1]; depl[11]= u[2];
    
    nodalForces force, plasticForce;

    if (_virtualDisplacementMethod.getValue())
    {

        if (_isPlasticKrabbenhoft.getValue())
        {            
            //Purely elastic deformation
            //Here _isDeformingPlastically = false, verified juste before the call to accumulateForceLarge in addForce

            helper::fixed_array<VoigtTensor2, 27> newStressVector;
            helper::fixed_array<VoigtTensor2, 27> newStrainVector; //for post-plastic deformation

            Eigen::Matrix<double, 12, 1> eigenDepl;
            for (int i = 0; i < 12; i++)
                eigenDepl(i) = depl[i];

            //***** Test if we enter in plastic deformation *****//
            const Eigen::Matrix<double, 6, 6>& C = beamsData.getValue()[i]._materialBehaviour;
            Eigen::Matrix<double, 6, 12> Be;
            helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
            helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[i]._pointMechanicalState;
            MechanicalState& beamMechanicalState = bd[i]._beamMechanicalState;

            VoigtTensor2 newStress;
            double yieldStress;
            bool res, isPlasticBeam = false;

            //For each Gauss point, we update the stress value for next iteration
            for (int gaussPointIt = 0; gaussPointIt < 27; gaussPointIt++)
            {
                Be = beamsData.getValue()[i]._BeMatrices[gaussPointIt];
                newStress = C*Be*eigenDepl;
                yieldStress = beamsData.getValue()[i]._yS;

                res = goToPlastic(newStress, yieldStress);
                if (res)
                {
                    pointMechanicalState[gaussPointIt] = PLASTIC;
                }
                isPlasticBeam = isPlasticBeam || res;

                newStressVector[gaussPointIt] = newStress;
            }
            beamsData.endEdit();
            //std::cout << std::endl; //DEBUG
            //***************************************************//

            if (isPlasticBeam)
            {
                // The computation of these new stresses should be plastic
                beamMechanicalState = PLASTIC;
                accumulateNonLinearForce(f, x, i, a, b);
            }
            else
            {
                force = beamsData.getValue()[i]._Ke_loc * depl;
                _prevStresses[i] = newStressVector; //for next time step

                std::cout << "Fint_local pour l'element " << i << " : " << std::endl << force << " " << std::endl << std::endl; //DEBUG

                // Apply lambda transpose (we use the rotation value of point a for the beam)
                Vec3 fa1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[0], force[1], force[2]));
                Vec3 fa2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[3], force[4], force[5]));

                Vec3 fb1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[6], force[7], force[8]));
                Vec3 fb2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[9], force[10], force[11]));

                f[a] += Deriv(-fa1, -fa2);
                f[b] += Deriv(-fb1, -fb2);
            }

        } //endif _isPlasticKrabbenhoft

        else
        {
            force = beamsData.getValue()[i]._Ke_loc * depl;

            Vec3 fa1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[0], force[1], force[2]));
            Vec3 fa2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[3], force[4], force[5]));

            Vec3 fb1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[6], force[7], force[8]));
            Vec3 fb2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[9], force[10], force[11]));

            f[a] += Deriv(-fa1, -fa2);
            f[b] += Deriv(-fb1, -fb2);
        }
    } //endif _virtualDisplacementMethod

    else
    {
        // this computation can be optimised: (we know that half of "depl" is null)
        force = beamsData.getValue()[i]._k_loc * depl;

        if (_isPlasticMuller.getValue())
        {
            //Update nodal forces to compute plasticity
            //TO DO: use forces from the previous step?
            _nodalForces[i] = force;

            updatePlasticity(i, a, b);
            plasticForce = beamsData.getValue()[i]._M_loc * _VFPlasticStrains[i];
            force -= plasticForce;
        }

        // Apply lambda transpose (we use the rotation value of point a for the beam)
        Vec3 fa1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[0], force[1], force[2]));
        Vec3 fa2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[3], force[4], force[5]));

        Vec3 fb1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[6], force[7], force[8]));
        Vec3 fb2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[9], force[10], force[11]));


        f[a] += Deriv(-fa1, -fa2);
        f[b] += Deriv(-fb1, -fb2);
    }
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::applyStiffnessLarge(VecDeriv& df, const VecDeriv& dx, int i, Index a, Index b, double fact)
{
    //const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    Displacement local_depl;
    defaulttype::Vec<3,Real> u;
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

    //if (_isPlasticKrabbenhoft.getValue())
    //    std::cout << "deplacement vitesse (elastique) pour l'element " << i << " : " << std::endl << local_depl << " " << std::endl << std::endl; //DEBUG

    Displacement local_force;

    if (_virtualDisplacementMethod.getValue())
    {
        local_force = beamsData.getValue()[i]._Ke_loc * local_depl;
    }
    else
    {
        // this computation can be optimised: (we know that half of "depl" is null)
        local_force = beamsData.getValue()[i]._k_loc * local_depl;

        //TO DO: plasticity?
    }

    //if (_isPlasticKrabbenhoft.getValue())
        //std::cout << "K*v_local pour l'element " << i << " : " << std::endl << local_force << " " << std::endl << std::endl; //DEBUG

    Vec3 fa1 = q.rotate(defaulttype::Vec3d(local_force[0],local_force[1] ,local_force[2] ));
    Vec3 fa2 = q.rotate(defaulttype::Vec3d(local_force[3],local_force[4] ,local_force[5] ));
    Vec3 fb1 = q.rotate(defaulttype::Vec3d(local_force[6],local_force[7] ,local_force[8] ));
    Vec3 fb2 = q.rotate(defaulttype::Vec3d(local_force[9],local_force[10],local_force[11]));

    //TO DO: beamsData.endEdit(); consecutive to the call to beamQuat

    df[a] += Deriv(-fa1,-fa2) * fact;
    df[b] += Deriv(-fb1,-fb2) * fact;

    //if (_isPlasticKrabbenhoft.getValue())
    //    std::cout << "K*v_tot (elastique) pour l'element " << i << " : " << std::endl << df << " " << std::endl << std::endl; //DEBUG
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real k = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    defaulttype::BaseMatrix* mat = r.matrix;

    if (r)
    {
        unsigned int i=0;
        unsigned int &offset = r.offset;

        typename VecElement::const_iterator it;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];

            const bool beamMechanicalState = beamsData.getValue()[i]._beamMechanicalState;

            defaulttype::Quat& q = beamQuat(i); //x[a].getOrientation();
            q.normalize();
            Transformation R,Rt;
            q.toMatrix(R);
            Rt.transpose(R);
            StiffnessMatrix K;
            bool exploitSymmetry = _useSymmetricAssembly.getValue();

            if (_virtualDisplacementMethod.getValue())
            {
                StiffnessMatrix K0;
                if (beamMechanicalState == PLASTIC)
                    K0 = beamsData.getValue()[i]._Kt_loc;
                else
                    K0 = beamsData.getValue()[i]._Ke_loc; //TO TO: distinguish ELASTIC and POST-PLASTIC ?

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

            } // end if (_virtualDisplacementMethod)
            else
            {
                const StiffnessMatrix& K0 = beamsData.getValue()[i]._k_loc;
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
            }
            //TO DO: beamsData.endEdit(); consecutive to the call to beamQuat

        } // end for _indexedElements
    } // end if(r)
}


template<class DataTypes>
void MultiBeamForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    std::vector<defaulttype::Vector3> centrelinePoints[1];
    std::vector<defaulttype::Vector3> gaussPoints[1];
    std::vector<defaulttype::Vec<4, float>> colours[1];

    for (unsigned int i=0; i<_indexedElements->size(); ++i)
        drawElement(i, gaussPoints, centrelinePoints, colours, x);

    vparams->drawTool()->setPolygonMode(2, true);
    vparams->drawTool()->setLightingEnabled(true);
    vparams->drawTool()->drawPoints(gaussPoints[0], 1, colours[0]);
    vparams->drawTool()->drawLines(centrelinePoints[0], 1.0, defaulttype::Vec<4, float>(0.24f, 0.72f, 0.96f, 1.0f));
    vparams->drawTool()->setLightingEnabled(false);
    vparams->drawTool()->setPolygonMode(0, false);
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::drawElement(int i, std::vector< defaulttype::Vector3 >* gaussPoints,
                                                 std::vector< defaulttype::Vector3 >* centrelinePoints,
                                                 std::vector<defaulttype::Vec<4, float>>* colours,
                                                 const VecCoord& x)
{
    Index a = (*_indexedElements)[i][0];
    Index b = (*_indexedElements)[i][1];

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

    beamsData.endEdit();

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
    const helper::fixed_array<MechanicalState, 27>& pointMechanicalState = beamsData.getValue()[i]._pointMechanicalState;
    int gaussPointIt = 0; //incremented in the lambda function to iterate over Gauss points

    LambdaType computeGaussCoordinates = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        //Shape function
        N = beamsData.getValue()[i]._N[gaussPointIt];
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

    ozp::quadrature::detail::Interval<3> interval = beamsData.getValue()[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeGaussCoordinates);

    //****** Centreline ******//
    int nbSeg = beamsData.getValue()[i]._nbCentrelineSeg; //number of segments descretising the centreline

    centrelinePoints[0].push_back(pa);

    Eigen::Matrix<double, 3, 12> drawN;
    const double L = beamsData.getValue()[i]._L;
    for (int drawPointIt = 0; drawPointIt < nbSeg - 1; drawPointIt++)
    {
        //Shape function of the centreline point
        drawN = beamsData.getValue()[i]._drawN[drawPointIt];
        Eigen::Matrix<double, 3, 1> u = drawN*disp;

        defaulttype::Vec3d beamVec = {u[0] + (drawPointIt +1)*(L/nbSeg), u[1], u[2]};
        defaulttype::Vec3d clp = pa + q.rotate(beamVec);
        centrelinePoints[0].push_back(clp); //First time as the end of the former segment
        centrelinePoints[0].push_back(clp); //Second time as the beginning of the next segment
    }

    centrelinePoints[0].push_back(pb);
}



template<class DataTypes>
void MultiBeamForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
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




template<class DataTypes>
void MultiBeamForceField<DataTypes>::initBeams(size_t size)
{
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    bd.resize(size);
    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::setBeam(unsigned int i, double E, double yS, double L, double nu, double zSection, double ySection)
{
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    bd[i].init(E,yS,L,nu,zSection,ySection, _isTimoshenko.getValue());
    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::BeamInfo::init(double E, double yS, double L, double nu, double zSection, double ySection, bool isTimoshenko)
{
    _E = E;
    _E0 = E;
    _yS = yS;
    _nu = nu;
    _L = L;
    _zDim = zSection;
    _yDim = ySection;

    _G=_E/(2.0*(1.0+_nu));
    _Iz = ySection*ySection*ySection*zSection / 12;

    _Iy = zSection*zSection*zSection*ySection / 12;
    _J = _Iz+_Iy;
    _A = zSection*ySection;

    _Asy = _A;
    _Asz = _A;

    double phiY, phiZ;
    double L2 = L*L;
    double kappaY = 1;
    double kappaZ = 1;

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
        _BeMatrices[i](0, 1) = (phiZInv * 6 * eta * (1 - 2*xi)) / _L;
        _BeMatrices[i](0, 2) = (phiYInv * 6 * zeta * (1 - 2*xi)) / _L;
        _BeMatrices[i](0, 3) = 0;
        _BeMatrices[i](0, 4) = phiYInv * zeta * (6*xi - 4 - phiY);
        _BeMatrices[i](0, 5) = phiZInv * eta * (4 - 6*xi + phiZ);
        _BeMatrices[i](0, 6) = 1 / _L;
        _BeMatrices[i](0, 7) = (phiZInv * 6 * eta * (2*xi - 1)) / _L;
        _BeMatrices[i](0, 8) = (phiYInv * 6 * zeta * (2*xi - 1)) / _L;
        _BeMatrices[i](0, 9) = 0;
        _BeMatrices[i](0, 10) = phiYInv * zeta * (6*xi - 2 + phiY);
        _BeMatrices[i](0, 11) = phiZInv * eta * (2 - 6*xi - phiZ);

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
        _BeMatrices[i](4, 3) = - eta / 2;
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
        _BeMatrices[i](5, 9) = - zeta / 2;
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
        _N[gaussPointIt](0, 4) = (1 - 4*xi + 3*xi2)*_L*zeta;
        _N[gaussPointIt](0, 5) = (-1 + 4*xi - 3*xi2)*_L*eta;
        _N[gaussPointIt](0, 6) = xi;
        _N[gaussPointIt](0, 7) = 6 * (-xi + xi2)*eta;
        _N[gaussPointIt](0, 8) = 6 * (-xi + xi2)*zeta;
        _N[gaussPointIt](0, 9) = 0;
        _N[gaussPointIt](0, 10) = (-2*xi + 3*xi2)*_L*zeta;
        _N[gaussPointIt](0, 11) = (2*xi - 3*xi2)*_L*eta;

        _N[gaussPointIt](1, 0) = 0;
        _N[gaussPointIt](1, 1) = 1 - 3*xi2 + 2*xi3;
        _N[gaussPointIt](1, 2) = 0;
        _N[gaussPointIt](1, 3) = (xi - 1)*_L*zeta;
        _N[gaussPointIt](1, 4) = 0;
        _N[gaussPointIt](1, 5) = (xi - 2*xi2 + xi3)*_L;
        _N[gaussPointIt](1, 6) = 0;
        _N[gaussPointIt](1, 7) = 3*xi2 - 2*xi3;
        _N[gaussPointIt](1, 8) = 0;
        _N[gaussPointIt](1, 9) = -_L*xi*zeta;
        _N[gaussPointIt](1, 10) = 0;
        _N[gaussPointIt](1, 11) = (-xi2 + xi3)*_L;

        _N[gaussPointIt](2, 0) = 0;
        _N[gaussPointIt](2, 1) = 0;
        _N[gaussPointIt](2, 2) = 1 - 3*xi2 + 2*xi3;
        _N[gaussPointIt](2, 3) = (1 - xi)*_L*eta;
        _N[gaussPointIt](2, 4) = (-xi + 2*xi2 - xi3)*_L;
        _N[gaussPointIt](2, 5) = 0;
        _N[gaussPointIt](2, 6) = 0;
        _N[gaussPointIt](2, 7) = 0;
        _N[gaussPointIt](2, 8) = 3*xi2 - 2*xi3;
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
        _N[gaussPointIt](0, 4) =  _L * phiYInv * (1 - 4*xi + 3*xi2 + phiY*(1 - xi))*zeta;
        _N[gaussPointIt](0, 5) = -_L * phiZInv * (1 - 4*xi + 3*xi2 + phiZ*(1 - xi))*eta;
        _N[gaussPointIt](0, 6) = xi;
        _N[gaussPointIt](0, 7) = 6 * phiZInv * (-xi + xi2)*eta;
        _N[gaussPointIt](0, 8) = 6 * phiYInv * (-xi + xi2)*zeta;
        _N[gaussPointIt](0, 9) = 0;
        _N[gaussPointIt](0, 10) =  _L * phiYInv * (-2*xi + 3*xi2 + phiY*xi)*zeta;
        _N[gaussPointIt](0, 11) = -_L * phiZInv * (-2*xi + 3*xi2 + phiZ*xi)*eta;

        _N[gaussPointIt](1, 0) = 0;
        _N[gaussPointIt](1, 1) = phiZInv * (1 - 3*xi2 + 2*xi3 + phiZ*(1 - xi));
        _N[gaussPointIt](1, 2) = 0;
        _N[gaussPointIt](1, 3) = (xi - 1)*_L*zeta;
        _N[gaussPointIt](1, 4) = 0;
        _N[gaussPointIt](1, 5) = _L * phiZInv * (xi - 2*xi2 + xi3 + (phiZ/2)*(xi - xi2));
        _N[gaussPointIt](1, 6) = 0;
        _N[gaussPointIt](1, 7) = phiZInv * (3*xi2 - 2*xi3 + phiZ*xi);
        _N[gaussPointIt](1, 8) = 0;
        _N[gaussPointIt](1, 9) = -_L * xi  *zeta;
        _N[gaussPointIt](1, 10) = 0;
        _N[gaussPointIt](1, 11) = _L * phiZInv *(-xi2 + xi3 - (phiZ/2)*(xi - xi2));

        _N[gaussPointIt](2, 0) = 0;
        _N[gaussPointIt](2, 1) = 0;
        _N[gaussPointIt](2, 2) = phiYInv * (1 - 3*xi2 + 2*xi3 + phiY*(1 - xi));
        _N[gaussPointIt](2, 3) = (1 - xi) * _L * eta;
        _N[gaussPointIt](2, 4) = -_L * phiYInv * (xi - 2*xi2 + xi3 + (phiY/2)*(xi - xi2));
        _N[gaussPointIt](2, 5) = 0;
        _N[gaussPointIt](2, 6) = 0;
        _N[gaussPointIt](2, 7) = 0;
        _N[gaussPointIt](2, 8) = phiYInv * (3*xi2 - 2*xi3 + phiY*xi);
        _N[gaussPointIt](2, 9) = _L * xi * eta;
        _N[gaussPointIt](2, 10) = -_L * phiYInv * (-xi2 + xi3 - (phiY/2)*(xi - xi2));
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
            _drawN[i - 1](1, 1) = phiZInv * (1 - 3*xi2 + 2*xi3 + phiZ*(1 - xi));
            _drawN[i - 1](1, 2) = 0;
            _drawN[i - 1](1, 3) = 0;
            _drawN[i - 1](1, 4) = 0;
            _drawN[i - 1](1, 5) = _L * phiZInv * (xi - 2*xi2 + xi3 + (phiZ/2)*(xi - xi2));
            _drawN[i - 1](1, 6) = 0;
            _drawN[i - 1](1, 7) = phiZInv * (3*xi2 - 2*xi3 + phiZ*xi);
            _drawN[i - 1](1, 8) = 0;
            _drawN[i - 1](1, 9) = 0;
            _drawN[i - 1](1, 10) = 0;
            _drawN[i - 1](1, 11) = _L * phiZInv *(-xi2 + xi3 - (phiZ/2)*(xi - xi2));

            _drawN[i - 1](2, 0) = 0;
            _drawN[i - 1](2, 1) = 0;
            _drawN[i - 1](2, 2) = phiYInv * (1 - 3*xi2 + 2*xi3 + phiY*(1 - xi));
            _drawN[i - 1](2, 3) = 0;
            _drawN[i - 1](2, 4) = -_L * phiYInv * (xi - 2*xi2 + xi3 + (phiY/2)*(xi - xi2));
            _drawN[i - 1](2, 5) = 0;
            _drawN[i - 1](2, 6) = 0;
            _drawN[i - 1](2, 7) = 0;
            _drawN[i - 1](2, 8) = phiYInv * (3*xi2 - 2*xi3 + phiY*xi);
            _drawN[i - 1](2, 9) = 0;
            _drawN[i - 1](2, 10) = -_L * phiYInv * (-xi2 + xi3 - (phiY/2)*(xi - xi2));
            _drawN[i - 1](2, 11) = 0;
        }
    }

    //Initialises the plastic indicators
    _pointMechanicalState.assign(ELASTIC);
    _beamMechanicalState = ELASTIC;

    //**********************************//
}



/**************************************************************************/
/*                      Plasticity - virtual forces                       */
/**************************************************************************/


template <class DataTypes>
void MultiBeamForceField<DataTypes>::reset()
{
    //serr<<"MultiBeamForceField<DataTypes>::reset"<<sendl;

    for (unsigned i = 0; i < _VFPlasticStrains.size(); ++i)
    {
        _VFPlasticStrains[i].clear();
    }
}


template<class DataTypes>
void MultiBeamForceField<DataTypes>::initPlasticityMatrix(int i, Index a, Index b) 
{
    Real phiy, phiz;
    Real _L = (Real)beamsData.getValue()[i]._L;
    Real _A = (Real)beamsData.getValue()[i]._A;
    Real _E = (Real)beamsData.getValue()[i]._E;
    Real _Iy = (Real)beamsData.getValue()[i]._Iy;
    Real _Iz = (Real)beamsData.getValue()[i]._Iz;
    Real _Asy = (Real)beamsData.getValue()[i]._Asy;
    Real _Asz = (Real)beamsData.getValue()[i]._Asz;
    Real _G = (Real)beamsData.getValue()[i]._G;
    
    Real L2 = (Real)(_L * _L);
    Real EIy = (Real)(_E * _Iy);
    Real EIz = (Real)(_E * _Iz);
    Real EA = (Real)(_E * _A);

    // Find shear-deformation parameters
    phiy = (Real)(12.0*_E*_Iz / (_G*_Asy*L2));
    phiz = (Real)(12.0*_E*_Iy / (_G*_Asz*L2));

    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    plasticityMatrix& M_loc = bd[i]._M_loc;

    // Define plasticity matrix 'M' in local coordinates
    M_loc.clear(); //TO DO: useful?
    M_loc[0][0] = -EA;
    M_loc[6][0] = EA;

    M_loc[1][1] = (Real)( 12.0*EIz / (L2 * (1.0 + phiy)) );
    M_loc[5][1] = (Real)( 6.0*EIz / (_L * (1.0 + phiy)) );
    M_loc[7][1] = (Real)( -12.0*EIz / (L2 * (1.0 + phiy)) );
    M_loc[11][1] = (Real)( 6.0*EIz / (_L * (1.0 + phiy)) );

    M_loc[2][2] = (Real)( 12.0*EIy / (L2 * (1 + phiz)) );
    M_loc[4][2] = (Real)( -6.0*EIy / (_L * (1 + phiz)) );
    M_loc[8][2] = (Real)( -12.0*EIy / (L2 * (1 + phiz)) );
    M_loc[10][2] = (Real)( -6.0*EIy / (_L * (1 + phiz)) );

    // Not necessary
    _isPlasticZoneComplete[i][0] = true;
    _isPlasticZoneComplete[i][1] = true;
    _isPlasticZoneComplete[i][2] = true;
    _isPlasticZoneComplete[i][3] = true;

    //NB: columns 4 to 7 are left empty, as they are updated in updatePlasticityMatrix

    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::updatePlasticityMatrix(int i, Index a, Index b)
{
    Real phiy, phiz;
    Real _L = (Real)beamsData.getValue()[i]._L;
    Real _E = (Real)beamsData.getValue()[i]._E;
    Real _Iy = (Real)beamsData.getValue()[i]._Iy;
    Real _Iz = (Real)beamsData.getValue()[i]._Iz;
    Real _Asy = (Real)beamsData.getValue()[i]._Asy;
    Real _Asz = (Real)beamsData.getValue()[i]._Asz;
    Real _G = (Real)beamsData.getValue()[i]._G;

    Real L2 = (Real)(_L * _L);
    Real EIy = (Real)(_E * _Iy);
    Real EIz = (Real)(_E * _Iz);

    Real aKy, bKy, aKz, bKz;

    aKy = _plasticZones[i][4][0];
    bKy = _plasticZones[i][4][1];
    aKz = _plasticZones[i][5][0];
    bKz = _plasticZones[i][5][1];

    Real _I4, _I5, _I6, _I7;
    _I4 = (Real)(bKy - aKy);
    _I5 = (Real)( ((2*aKy - _L)*(2*aKy - _L)*(2*aKy - _L) - (2*bKy - _L)*(2*bKy - _L)*(2*bKy - _L)) / (6*L2));

    _I6 =(Real)(aKz - bKz);
    _I7 = (Real)( ((2*bKz - _L)*(2*bKz - _L)*(2*bKz - _L) - (2*aKz - _L)*(2*aKz - _L)*(2*aKz - _L)) / (6*L2));

    // Find shear-deformation parameters
    phiy = (Real)(12.0*_E*_Iz / (_G*_Asy*L2));
    phiz = (Real)(12.0*_E*_Iy / (_G*_Asz*L2));

    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    plasticityMatrix& M_loc = bd[i]._M_loc;

    // Only columns 4 to 7 are none-constant in matrix M

    M_loc[4][4] = -EIy*_I4 / _L;
    M_loc[10][4] = EIy*_I4 / _L;

    M_loc[2][5] = (Real)(6*EIy*_I5 / (L2 * (1 + phiz)));
    M_loc[4][5] = (Real)(-3*EIy*_I5 / (_L * (1 + phiz)));
    M_loc[8][5] = (Real)(-6*EIy*_I5 / (L2 * (1 + phiz)));
    M_loc[10][5] = (Real)(-3*EIy*_I5 / (_L * (1 + phiz)));

    M_loc[5][6] = EIz*_I6 / _L;
    M_loc[11][6] = -EIz*_I6 / _L;

    M_loc[1][7] = (Real)(6*EIz*_I7 / (L2 * (1 + phiy)));
    M_loc[5][7] = (Real)(3*EIz*_I7 / (_L * (1 + phiy)));
    M_loc[7][7] = (Real)(-6*EIz*_I7 / (L2 * (1 + phiy)));
    M_loc[11][7] = (Real)(3*EIz*_I7 / (_L * (1 + phiy)));

    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::updatePlasticity(int i, Index a, Index b)
{
    Real epsilon, gammaXY, gammaXZ, kappaY1, kappaY2, kappaZ1, kappaZ2;
    bool update = false;

    Real _L = (Real)beamsData.getValue()[i]._L;

    //Updating total strains
    totalStrainEvaluation(i, a, b);
    epsilon = _VFTotalStrains[i][0];
    gammaXY = _VFTotalStrains[i][1];
    gammaXZ = _VFTotalStrains[i][2];
    kappaY1 = _VFTotalStrains[i][4];
    kappaY2 = _VFTotalStrains[i][5];
    kappaZ1 = _VFTotalStrains[i][6];
    kappaZ2 = _VFTotalStrains[i][7];

    // Strains are tested together if they have the same mechanical interpretation
    // No need to update plastic limits for epsilon, gammaXY, and gammaXZ which are constant over x
    if (epsilon > _VFPlasticYieldThreshold[0]) 
    {
        _VFPlasticStrains[i][0] += _VFPlasticCreep*epsilon;
    }

    if (gammaXY > _VFPlasticYieldThreshold[1])
    {
        _VFPlasticStrains[i][1] += _VFPlasticCreep*gammaXY;
    }

    if (gammaXZ > _VFPlasticYieldThreshold[2])
    {
        _VFPlasticStrains[i][2] += _VFPlasticCreep*gammaXZ;
    }

    Real cKy = _VFPlasticYieldThreshold[4];
    if (kappaY1*kappaY1 >= cKy)
    {
        // Plastic zone is [0,l]
        // NB: potentially, kappaY2 can be 0, but it doesn't change the computation
        _VFPlasticStrains[i][4] += _VFPlasticCreep*kappaY1;
        _VFPlasticStrains[i][5] += _VFPlasticCreep*kappaY2;

        if (!_isPlasticZoneComplete[i][4])
        {
            _plasticZones[i][4][0] = 0;
            _plasticZones[i][4][1] = _L;
            _isPlasticZoneComplete[i][4] = true;
            update = true;
        }
    }
    else
    {
        // Plastic zone limits have to be computed
        // They are computed using the condition: Norm2(kappaY1, kappaY2)^2 > _VFPlasticYieldThreshold[4]
        if (kappaY2 != 0)
        {
            Real aKy, bKy;
            aKy = (Real)((_L / 2) * (1 - (helper::rsqrt(cKy - kappaY1*kappaY1) / helper::rabs(kappaY2))));
            bKy = (Real)((_L / 2) * (1 + (helper::rsqrt(cKy - kappaY1*kappaY1) / helper::rabs(kappaY2))));

            if (aKy >= 0)
            {
                //Both limits are symetric regarding _L/2
                _plasticZones[i][4][0] = aKy;
                _plasticZones[i][4][1] = bKy;

                _VFPlasticStrains[i][4] += _VFPlasticCreep*kappaY1;
                _VFPlasticStrains[i][5] += _VFPlasticCreep*kappaY2;
                update = true;
            }            
        }
        //NB: if kappaY2 == 0 here, then there is no plasticity regarding kappaY
    }


    Real cKz = _VFPlasticYieldThreshold[5];
    if (kappaZ1*kappaZ1 >= cKz)
    {
        // Plastic zone is [0,l]
        // NB: potentially, kappaZ2 can be 0, but it doesn't change the computation
        _VFPlasticStrains[i][6] += _VFPlasticCreep*kappaZ1;
        _VFPlasticStrains[i][7] += _VFPlasticCreep*kappaZ2;

        if (!_isPlasticZoneComplete[i][5])
        {
            _plasticZones[i][5][0] = 0;
            _plasticZones[i][5][1] = _L;
            _isPlasticZoneComplete[i][5] = true;
            update = true;
        }
    }
    else
    {
        // Plastic zone limits have to be computed
        // They are computed using the condition: Norm2(kappaZ1, kappaZ2)^2 > _VFPlasticYieldThreshold[5]
        if (kappaZ2 != 0)
        {
            Real aKz, bKz;
            aKz = (Real)((_L / 2) * (1 - (helper::rsqrt(cKz - kappaZ1*kappaZ1) / helper::rabs(kappaZ2))));
            bKz = (Real)((_L / 2) * (1 + (helper::rsqrt(cKz - kappaZ1*kappaZ1) / helper::rabs(kappaZ2))));

            if (aKz >= 0)
            {
                //Both limits are symetric regarding _L/2
                _plasticZones[i][5][0] = aKz;
                _plasticZones[i][5][1] = bKz;

                _VFPlasticStrains[i][6] += _VFPlasticCreep*kappaZ1;
                _VFPlasticStrains[i][7] += _VFPlasticCreep*kappaZ2;
                update = true;
            }           
        }
        //NB: if kappaZ2 == 0 here, then there is no plasticity regarding kappaZ
    }

    if (update)
        updatePlasticityMatrix(i, a, b);

}


template<class DataTypes>
void MultiBeamForceField<DataTypes>::totalStrainEvaluation(int i, Index a, Index b)
{
 
    Real _L = (Real)beamsData.getValue()[i]._L;
    Real _A = (Real)beamsData.getValue()[i]._A;
    Real _E = (Real)beamsData.getValue()[i]._E;
    Real _Iy = (Real)beamsData.getValue()[i]._Iy;
    Real _Iz = (Real)beamsData.getValue()[i]._Iz;
    Real _Asy = (Real)beamsData.getValue()[i]._Asy;
    Real _Asz = (Real)beamsData.getValue()[i]._Asz;
    Real _G = (Real)beamsData.getValue()[i]._G;

    Real _N1 = _nodalForces[i][0];
    Real _Qy1 = _nodalForces[i][1];
    Real _Qz1 = _nodalForces[i][2];
    Real _My1 = _nodalForces[i][4];
    Real _Mz1 = _nodalForces[i][5];
    Real _N2 = _nodalForces[i][6];
    Real _Qy2 = _nodalForces[i][7];
    Real _Qz2 = _nodalForces[i][8];
    Real _My2 = _nodalForces[i][10];
    Real _Mz2 = _nodalForces[i][11];

    //Updating the total strain constants
    _VFTotalStrains[i][0] = (Real)( (_N2 - _N1) / (2*_E*_A) );
    _VFTotalStrains[i][1] = (Real)( (_Qy1 - _Qy2) / (2 *_G*_Asy) );
    _VFTotalStrains[i][2] = (Real)( (_Qz1 - _Qz2) / (2*_G*_Asz) );
    _VFTotalStrains[i][3] = 0;
    _VFTotalStrains[i][4] = (Real)( (_My2 - _My1) / (2*_E*_Iy) );
    _VFTotalStrains[i][5] = (Real)( (_My2 + _My1) / (2*_E*_Iy) );
    _VFTotalStrains[i][6] = (Real)( (_Mz2 - _Mz1) / (2*_E*_Iz) );
    _VFTotalStrains[i][7] = (Real)( (_Mz2 + _Mz1) / (2*_E*_Iz) );
}
/**************************************************************************/




/**************************************************************************/
/*                  Plasticity - virtual displacement                     */
/**************************************************************************/

template<class DataTypes>
void MultiBeamForceField<DataTypes>::computeVDStiffness(int i, Index, Index)
{
    Real _L = (Real)beamsData.getValue()[i]._L;
    Real _yDim = (Real)beamsData.getValue()[i]._yDim;
    Real _zDim = (Real)beamsData.getValue()[i]._zDim;

    const Eigen::Matrix<double, 6, 6>& C = beamsData.getValue()[i]._materialBehaviour;
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    StiffnessMatrix& Ke_loc = bd[i]._Ke_loc;
    StiffnessMatrix& Kt_loc = bd[i]._Kt_loc;

    Ke_loc.clear();
    Kt_loc.clear();

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
        Be = beamsData.getValue()[i]._BeMatrices[gaussPointIterator];

        stiffness += (w1*w2*w3)*Be.transpose()*C*Be;

        gaussPointIterator++; //next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = beamsData.getValue()[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeStressMatrix);

    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
        {
            Ke_loc[i][j] = stiffness(i, j);
            //Initialising the tangent stiffness matrix with Ke
            Kt_loc[i][j] = stiffness(i, j);
        }

    //DEBUG
    //std::cout << "Ke pour l'element " << i << " : " << std::endl << stiffness << " " << std::endl << std::endl;

    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::computeMaterialBehaviour(int i, Index a, Index b)
{

    Real youngModulus = (Real)beamsData.getValue()[i]._E;
    Real poissonRatio = (Real)beamsData.getValue()[i]._nu;

    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());

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
    //C(3, 3) = C(4, 4) = C(5, 5) = (1 - 2 * poissonRatio) / (2 * (1 - poissonRatio));
    C(3, 3) = C(4, 4) = C(5, 5) = (1 - 2 * poissonRatio) / (1 - poissonRatio);
    C *= (youngModulus*(1 - poissonRatio)) / ((1 + poissonRatio)*(1 - 2 * poissonRatio));

    //DEBUG
    //std::cout << "C pour l'element " << i << " : " << std::endl << C << " " << std::endl << std::endl;

    Eigen::Matrix<double, 6, 6>& S = bd[i]._materialInv;

    S(0, 0) = S(1, 1) = S(2, 2) = 1;
    S(0, 1) = S(0, 2) = S(1, 0) = S(1, 2) = S(2, 0) = S(2, 1) = -poissonRatio;
    S(0, 3) = S(0, 4) = S(0, 5) = 0;
    S(1, 3) = S(1, 4) = S(1, 5) = 0;
    S(2, 3) = S(2, 4) = S(2, 5) = 0;
    S(3, 0) = S(3, 1) = S(3, 2) = S(3, 4) = S(3, 5) = 0;
    S(4, 0) = S(4, 1) = S(4, 2) = S(4, 3) = S(4, 5) = 0;
    S(5, 0) = S(5, 1) = S(5, 2) = S(5, 3) = S(5, 4) = 0;
    S(3, 3) = S(4, 4) = S(5, 5) = 1 + poissonRatio;
    S *= 1/youngModulus;

    beamsData.endEdit();
};

template<class DataTypes>
void MultiBeamForceField<DataTypes>::computePlasticForces(int i, Index a, Index b, const Displacement& totalDisplacement, nodalForces& plasticForces)
{
    // Computes the contribution of plasticity by numerical integration
    // Step 1: total strain is computed at each integration node
    // Step 2: if totalStrain > plasticThreshold for a node, plastic strain is updated
    // Step 3: the corresponding force is computed through numerical integration

    Real _L = (Real)beamsData.getValue()[i]._L;
    Real _yDim = (Real)beamsData.getValue()[i]._yDim;
    Real _zDim = (Real)beamsData.getValue()[i]._zDim;
    const Eigen::Matrix<double, 6, 6>& C = beamsData.getValue()[i]._materialBehaviour;
    
    VoigtTensor totalStrain;
    
    Eigen::Matrix<double, 12, 1> totalDisp;
    for (int i=0; i<12; i++)
        totalDisp(i) = totalDisplacement[i];

    // Reduced integration
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    // Setting variables for the reduced intergation process defined in quadrature.h
    Eigen::Matrix<double, 6, 12> Be;
    Eigen::Matrix<double, 12, 1> fe_pla = Eigen::VectorXd::Zero(12);
    int gaussPointIterator = 0;

    // Stress matrix, to be integrated
    LambdaType computeStressMatrix = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        // Be
        Be = beamsData.getValue()[i]._BeMatrices[gaussPointIterator];

        totalStrain = Be*totalDisp;

        //Step 2: test of the plasticity threshold
        updatePlasticStrain(i, a, b, totalStrain, gaussPointIterator);
        
        //Step3: addition of this Gauss point contribution to the plastic forces
        VoigtTensor plasticStrain = _VDPlasticStrains[i].row(gaussPointIterator);
        fe_pla += (w1*w2*w3)*Be.transpose()*C*plasticStrain;
        gaussPointIterator++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = beamsData.getValue()[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeStressMatrix);

    //Transfer to the force compuation
    for (int i = 0; i < 12; i++)
        plasticForces[i] = fe_pla(i);

};


template<class DataTypes>
void MultiBeamForceField<DataTypes>::updatePlasticStrain(int i, Index a, Index b, VoigtTensor& totalStrain, int gaussPointIterator)
{
    // gaussPointIterator represents the ith Gauss point (over a total of 27) where the strain is computed
    // The number itself has no importance as the order is entirely fixed by the reduced integration procedure

    VoigtTensor plasticStrain = _VDPlasticStrains[i].row(gaussPointIterator);
    VoigtTensor elasticStrain = totalStrain - plasticStrain;
    double strainNorm = elasticStrain.squaredNorm();

    if (strainNorm > _VDPlasticYieldThreshold*_VDPlasticYieldThreshold) //TO DO: potentially the threshold should be squared
    {
        //Update this Gauss point plasticStrain
        _VDPlasticStrains[i].row(gaussPointIterator) = _VDPlasticCreep*totalStrain; // 6x1 vector
    }
};



template<class DataTypes>
void MultiBeamForceField<DataTypes>::updateYieldStress(int beamIndex, double yieldStressIncrement)
{
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    double &yieldStress = bd[beamIndex]._yS;
    yieldStress += yieldStressIncrement;
    beamsData.endEdit();
}



template< class DataTypes>
bool MultiBeamForceField<DataTypes>::goToPlastic(const VoigtTensor2 &stressTensor,
                                                            const double yieldStress,
                                                            const bool verbose /*=FALSE*/)
{
    double threshold = 1e-5; //TO DO: choose adapted threshold

    double yield = vonMisesYield(stressTensor, yieldStress);
    if (verbose)
    {
        std::cout.precision(17);
        std::cout << yield << std::scientific << " "; //DEBUG
    }
    return yield > threshold;
}


template< class DataTypes>
bool MultiBeamForceField<DataTypes>::goToPostPlastic(const VoigtTensor2 &stressTensor,
                                                     const VoigtTensor2 &stressIncrement,
                                                     const double yieldStress,
                                                     const bool verbose /*=FALSE*/)
{
    double threshold = -0.f; //TO DO: use proper threshold

    // Computing the normal to the yield surface using the deviatoric stress
    double meanStress = (1.0 / 3)*(stressTensor[0] + stressTensor[1] + stressTensor[2]);
    VoigtTensor2 deviatoricStress = stressTensor;
    deviatoricStress[0] -= meanStress;
    deviatoricStress[1] -= meanStress;
    deviatoricStress[2] -= meanStress;
    double sigmaEq = equivalentStress(stressTensor);
    VoigtTensor2 yieldNormal = helper::rsqrt(3.0 / 2)*(1 / sigmaEq)*deviatoricStress;

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
double MultiBeamForceField<DataTypes>::equivalentStress (const VoigtTensor2 &stressTensor)
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
double MultiBeamForceField<DataTypes>::vonMisesYield(const VoigtTensor2 &stressTensor,
                                                     const double yieldStress)
{
    double eqStress = equivalentStress(stressTensor);
    return eqStress - yieldStress;
}


template< class DataTypes>
Eigen::Matrix<double, 6, 1> MultiBeamForceField<DataTypes>::vonMisesGradient(const VoigtTensor2 &stressTensor,
                                                                             const double yieldStress)
{
    VoigtTensor2 res = VoigtTensor2::Zero();

    if (stressTensor.isZero())
        return res; //TO DO: is that correct ?

    double sigmaX = stressTensor[0];
    double sigmaY = stressTensor[1];
    double sigmaZ = stressTensor[2];
    double sigmaYZ = stressTensor[3];
    double sigmaZX = stressTensor[4];
    double sigmaXY = stressTensor[5];

    double fact = 1 / (2 * (vonMisesYield(stressTensor, yieldStress) + yieldStress));

    res[0] = 2 * sigmaX - sigmaY - sigmaZ;
    res[1] = 2 * sigmaY - sigmaZ - sigmaX;
    res[2] = 2 * sigmaZ - sigmaX - sigmaY;
    res[3] = 6 * sigmaYZ;
    res[4] = 6 * sigmaZX;
    res[5] = 6 * sigmaXY;

    res *= fact;
    return res;
}


template< class DataTypes>
Eigen::Matrix<double, 6, 6> MultiBeamForceField<DataTypes>::vonMisesHessian(const VoigtTensor2 &stressTensor,
                                                                            const double yieldStress)
{
    VoigtTensor4 res = VoigtTensor4::Zero();

    if (stressTensor.isZero())
        return res; //TO DO: is that correct ?

    //Order 1 terms
    double sigmaX = stressTensor[0];
    double sigmaY = stressTensor[1];
    double sigmaZ = stressTensor[2];
    double sigmaYZ = stressTensor[3];
    double sigmaZX = stressTensor[4];
    double sigmaXY = stressTensor[5];

    double auxX = 2*sigmaX - sigmaY - sigmaZ;
    double auxY = 2*sigmaY - sigmaZ - sigmaX;
    double auxZ = 2*sigmaZ - sigmaX - sigmaY;

    //Order 2 terms
    double sX2 = sigmaX*sigmaX;
    double sY2 = sigmaY*sigmaY;
    double sZ2 = sigmaZ*sigmaZ;
    double sYsZ = sigmaY*sigmaZ;
    double sZsX = sigmaZ*sigmaX;
    double sXsY = sigmaX*sigmaY;

    double sYZ2 = sigmaYZ*sigmaYZ;
    double sZX2 = sigmaZX*sigmaZX;
    double sXY2 = sigmaXY*sigmaXY;

    //Others
    double sigmaE = vonMisesYield(stressTensor, yieldStress) + yieldStress;
    double sigmaE3 = sigmaE*sigmaE*sigmaE;
    double invSigmaE = 1 / sigmaE;

    res(0, 0) = invSigmaE - (auxX*auxX / (4*sigmaE3));
    res(0, 1) = -0.5*invSigmaE - ((-2*sX2 - 2*sY2 + sZ2 - sYsZ - sZsX + 5*sXsY) / (4*sigmaE3));
    res(0, 2) = -0.5*invSigmaE - ((-2*sX2 + sY2 - 2*sZ2 - sYsZ + 5*sZsX - sXsY) / (4*sigmaE3));
    res(0, 3) = -(6*sigmaYZ*auxX / (4*sigmaE3));
    res(0, 4) = -(6*sigmaZX*auxX / (4*sigmaE3));
    res(0, 5) = -(6*sigmaXY*auxX / (4*sigmaE3));

    res(1, 0) = res(0, 1);
    res(1, 1) = invSigmaE - (auxY*auxY / (4 * sigmaE3));
    res(1, 2) = -0.5*invSigmaE - ((sX2 - 2*sY2 - 2*sZ2 + 5*sYsZ - sZsX - sXsY) / (4 * sigmaE3));
    res(1, 3) = -(6*sigmaYZ*auxY / (4 * sigmaE3));
    res(1, 4) = -(6*sigmaZX*auxY / (4 * sigmaE3));
    res(1, 5) = -(6*sigmaXY*auxY / (4 * sigmaE3));

    res(2, 0) = res(0, 2);
    res(2, 1) = res(1, 2);
    res(2, 2) = invSigmaE - (auxZ*auxZ / (4 * sigmaE3));
    res(2, 3) = -(6*sigmaYZ*auxZ / (4 * sigmaE3));
    res(2, 4) = -(6*sigmaZX*auxZ / (4 * sigmaE3));
    res(2, 5) = -(6*sigmaXY*auxZ / (4 * sigmaE3));

    res(3, 0) = res(0, 3);
    res(3, 1) = res(1, 3);
    res(3, 2) = res(2, 3);
    res(3, 3) = 3*invSigmaE - (9*sYZ2 / sigmaE3);
    res(3, 4) = -(9*sigmaYZ*sigmaZX / sigmaE3);
    res(3, 5) = -(9*sigmaYZ*sigmaXY / sigmaE3);

    res(4, 0) = res(0, 4);
    res(4, 1) = res(1, 4);
    res(4, 2) = res(2, 4);
    res(4, 3) = res(3, 4);
    res(4, 4) = 3*invSigmaE - (9*sZX2 / sigmaE3);
    res(4, 5) = -(9*sigmaZX*sigmaXY / sigmaE3);

    res(5, 0) = res(0, 5);
    res(5, 1) = res(1, 5);
    res(5, 2) = res(2, 5);
    res(5, 3) = res(3, 5);
    res(5, 4) = res(4, 5);
    res(5, 5) = 3*invSigmaE - (9*sXY2 / sigmaE3);

    return res;
}



//*****************************************************************************************//
//***************************************** DEBUG *****************************************//
//*****************************************************************************************//

template< class DataTypes>
Eigen::Matrix<double, 6, 1> MultiBeamForceField<DataTypes>::vonMisesGradientFD(const VoigtTensor2 &currentStressTensor,
                                                                               const double increment,
                                                                               const double yieldStress)
{
    //increment has to be non-zero

    VoigtTensor2 res = VoigtTensor2::Zero();

    if (currentStressTensor.isZero())
        return res;

    VoigtTensor2 newStress = currentStressTensor;
    double Ftph, Ft;

    Ft = vonMisesYield(currentStressTensor, yieldStress);

    for (int i = 0; i < 6; i++)
    {
        newStress[i] += increment;
        Ftph = vonMisesYield(newStress, yieldStress);
        res[i] = (Ftph - Ft) / increment;
        newStress[i] -= increment;
    }

    return res;
}


template< class DataTypes>
Eigen::Matrix<double, 6, 6> MultiBeamForceField<DataTypes>::vonMisesHessianFD(const VoigtTensor2 &lastStressTensor,
                                                                              const VoigtTensor2 &currentStressTensor,
                                                                              const double yieldStress)
{
    VoigtTensor4 res = VoigtTensor4::Zero();
    return res;
}

template< class DataTypes>
double MultiBeamForceField<DataTypes>::voigtDotProduct(const VoigtTensor2 &t1, const VoigtTensor2 &t2)
{
    // This method provides a correct implementation of the dot product for 2nd-order tensors represented
    // with Voigt notation. As the tensors are symmetric, then can be represented with only 6 elements,
    // but all non-diagonal elements have to be taken into account for a dot product.

    double res = 0.0;
    res += t1[0]*t2[0] + t1[1]*t2[1] + t1[2]*t2[2];      //diagonal elements
    res += 2*(t1[3]*t2[3] + t1[4]*t2[4] + t1[5]*t2[5]);  //non-diagonal elements
    return res;
}


//*****************************************************************************************//


template< class DataTypes>
void MultiBeamForceField<DataTypes>::solveDispIncrement(const tangentStiffnessMatrix &tangentStiffness,
                                                        EigenDisplacement &du,
                                                        const EigenNodalForces &residual)
{
    //Solve the linear system K*du = residual, for du

    //First try is with the dense LU decomposition provided by the Eigen library
    //NB: this is not inplace decomposition because we pass a const reference as argument
    Eigen::FullPivLU<tangentStiffnessMatrix> LU(tangentStiffness);
    du = LU.solve(residual);
}



template< class DataTypes>
void MultiBeamForceField<DataTypes>::computeLocalDisplacement(const VecCoord& x, Displacement &localDisp,
                                                              int i, Index a, Index b)
{
    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    beamQuat(i) = x[a].getOrientation();
    beamQuat(i).normalize();

    beamsData.endEdit();

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
void MultiBeamForceField<DataTypes>::computeDisplacementIncrement(const VecCoord& pos, const VecCoord& lastPos, Displacement &currentDisp,
                                                                  Displacement &lastDisp, Displacement &dispIncrement, int i, Index a, Index b)
{
    // ***** Displacement for current position *****//

    computeLocalDisplacement(pos, currentDisp, i, a, b);

    // ***** Displacement for last position *****//

    computeLocalDisplacement(lastPos, lastDisp, i, a, b);

    // ***** Displacement increment *****//

    dispIncrement = currentDisp - lastDisp;
}



template< class DataTypes>
void MultiBeamForceField<DataTypes>::computeStressIncrement(int index,
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

    const Eigen::Matrix<double, 6, 6>& C = beamsData.getValue()[index]._materialBehaviour; //Matrix D in Krabbenhoft's
    const double yieldStress = beamsData.getValue()[index]._yS;

    //First we compute the elastic predictor
    VoigtTensor2 elasticIncrement = C*strainIncrement;
    VoigtTensor2 currentStressPoint = initialStress + elasticIncrement;


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
        const helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = beamsData.getValue()[index]._plasticStrainHistory;
        const Eigen::Matrix<double, 6, 12> &Be = beamsData.getValue()[index]._BeMatrices[gaussPointIt];

        EigenDisplacement eigenCurrentDisp;
        for (int k = 0; k < 12; k++)
        {
            eigenCurrentDisp(k) = currentDisp[k];
        }

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
        isPostPlastic = goToPostPlastic(initialStress, elasticIncrement, yieldStress);

        if (isPostPlastic)
        {
            //The newly computed stresses don't correspond anymore to plastic
            //deformation. We must re-compute them in an elastic way, while
            //keeping track of the plastic deformation history

            pointMechanicalState = POSTPLASTIC;
            newStressPoint = currentStressPoint; //TO DO: check if we should take back the plastic strain

            //************ Computation of the remaining plastic strain ************//

            //TO DO: change with some kind of integration of the plastic strain increments over the plastic deformation path.
            //       For the moment the plastic strain increments are just summed up at the end of the plastic stress
            //       increment computation

            return;
        }
    }

    /*****************  Plastic stress increment computation *****************/

    // 1st SCENARIO : perfect plasticity
    if (_isPerfectlyPlastic.getValue())
    {
        /* For a perfectly plastic material, the assumption is made that during
         * the plastic phase, the deformation is entirely plastic. In this case,
         * the internal stress state of the material does not change during the
         * plastic deformation, remaining constant (in 1D, equal to the yield
         * stress value). All the corresponding deformation energy is dissipated
         * in the form of plastic strain (i.e. during a plastic deformation, the
         * elastic strain is null).
        */

        /**** Naive implementation ****/

        //// Updating the stress point
        //newStressPoint = initialStress; // delta_sigma = 0

        //// Updating the plastic strain
        //helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
        //helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = bd[index]._plasticStrainHistory;
        //plasticStrainHistory[gaussPointIt] += strainIncrement;
        //beamsData.endEdit();

        /**** Litterature implementation ****/
        // Ref: Theoritecal foundation for large scale computations for nonlinear material behaviour, Hugues (et al) 1984

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
        double lambda = voigtDotProduct(yieldNormal, elasticPredictor) / voigtDotProduct(yieldNormal, C*yieldNormal);

        VoigtTensor2 plasticStrainIncrement = lambda*yieldNormal;
        helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
        helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = bd[index]._plasticStrainHistory;
        plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;
        beamsData.endEdit();
    }

    // 2nd SCENARIO : hardening
    else
    {
        /*****************************/
        /*   Plastic return method   */
        /*****************************/

        ////First we compute the elastic predictor
        //VoigtTensor2 elasticIncrement = C*strainIncrement;

        //VoigtTensor2 currentStressPoint = initialStress + elasticIncrement; //point B in Krabbenhoft's

        //VoigtTensor2 grad, plasticStressIncrement;
        //double denum, dLambda;
        //double fSigma = vonMisesYield(currentStressPoint, yieldStress); //for the 1st iteration

        ////We start iterations with point B
        //for (unsigned int i = 0; i < nbMaxIterations; i++)
        //{
        //    grad = vonMisesGradient(currentStressPoint, yieldStress);
        //    denum = grad.transpose()*C*grad;
        //    dLambda = fSigma / denum;

        //    plasticStressIncrement = dLambda*C*grad;

        //    currentStressPoint -= plasticStressIncrement;

        //    fSigma = vonMisesYield(currentStressPoint, yieldStress);
        //    if (fSigma < iterationThreshold)
        //        break; //Yield condition : the current stress point is on the yield surface
        //}
        //// return the total stress increment
        ////TO DO: return the current stress directly instead?
        //stressIncrement = currentStressPoint - initialStress;

        /*****************************/
        /*      Implicit method      */
        /*****************************/

        /* With the initial conditions we use, the first system to be solve actually
        * admits a closed-form solution.
        * If a Von Mises yield criterion is used, the residual for this closed-form
        * solution should be 0 (left apart rounding errors caused by floating point
        * operations). The reason for this is the underlying regularity of the Von
        * Mises yield function, assuring a constant gradient between consecutive
        * stress points.
        * We implement the closed-form solution of the first step separately, in
        * order to save the computational cost of the system solution.
        */

        Eigen::Matrix<double, 7, 1> newIncrement = Eigen::Matrix<double, 7, 1>::Zero();
        double yieldCondition = vonMisesYield(currentStressPoint, yieldStress);
        VoigtTensor2 gradient = vonMisesGradient(currentStressPoint, yieldStress);

        newIncrement(6, 0) = yieldCondition / voigtDotProduct(gradient.transpose(), C*gradient);
        newIncrement.block<6, 1>(0, 0) = -newIncrement(6, 0)*C*gradient;

        Eigen::Matrix<double, 7, 1> totalIncrement = newIncrement;

        // Updating the stress point
        currentStressPoint += newIncrement.block<6, 1>(0, 0);

        yieldCondition = vonMisesYield(currentStressPoint, yieldStress);

        // Testing if the result of the first iteration is satisfaying
        double threshold = 2e-6; //TO DO: choose coherent value
        bool consistencyTestIsPositive = helper::rabs(yieldCondition) >= threshold;

        if (!consistencyTestIsPositive)
        {
            /* If the new stress point computed after one iteration of the implicit
            * method satisfied the consistency condition, we could stop the
            * iterative procedure at this point.
            * Otherwise the solution found with the first iteration does not
            * satisfy the consistency condition. In this case, we need to go
            * through more iterations to find a more correct solution.
            */
            unsigned int nbMaxIterations = 25;

            Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();

            // Jacobian J of the nonlinear system of equations
            Eigen::Matrix<double, 7, 7> J = Eigen::Matrix<double, 7, 7>::Zero();

            // Second member b
            Eigen::Matrix<double, 7, 1> b = Eigen::Matrix<double, 7, 1>::Zero();

            //solver
            Eigen::FullPivLU<Eigen::Matrix<double, 7, 7> > LU(J.rows(), J.cols());

            // Updating gradient
            gradient = vonMisesGradient(currentStressPoint, yieldStress);
            // Updating / initialising hessian
            VoigtTensor4 hessian = vonMisesHessian(currentStressPoint, yieldStress);
            // NB: yieldCondition was already updated after first iterations

            unsigned int count = 1;
            while (helper::rabs(yieldCondition) >= threshold && count < nbMaxIterations)
            {
                //Updates J and b
                J.block<6, 6>(0, 0) = I6 + totalIncrement(6, 0)*C*hessian;
                J.block<6, 1>(0, 6) = C*gradient;
                J.block<1, 6>(6, 0) = gradient.transpose();

                b.block<6, 1>(0, 0) = totalIncrement.block<6, 1>(0, 0) - elasticIncrement + C*totalIncrement(6, 0)*gradient;
                b(6, 0) = yieldCondition;

                //std::cout << "J pour le point " << gaussPointIt << " iteration " << count << " : " << std::endl << J << " " << std::endl << std::endl; //DEBUG
                //Computes the new increment
                LU.compute(J);
                //std::cout << "    Determinant LU : " << LU.determinant() << " " << std::endl << std::endl; //DEBUG
                newIncrement = LU.solve(-b);

                totalIncrement += newIncrement;

                // Update of the yield condition, gradient and hessian.
                currentStressPoint = initialStress + totalIncrement.block<6, 1>(0, 0);
                yieldCondition = vonMisesYield(currentStressPoint, yieldStress);
                gradient = vonMisesGradient(currentStressPoint, yieldStress);
                //gradientFD = vonMisesGradientFD(currentStressPoint, increment, yieldStress); //DEBUG
                hessian = vonMisesHessian(currentStressPoint, yieldStress);

                count++;
            }
        } // endif (!consistencyTestIsPositive)

        // Computation of the plastic strain increment, to keep track of the plastic loading history

        double plasticMultiplier = totalIncrement(6, 0);

        //TO DO: check if the increments should be stored independently rather than summed
        helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
        helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = bd[index]._plasticStrainHistory;
        VoigtTensor2 plasticStrainIncrement = plasticMultiplier*gradient;
        plasticStrainHistory[gaussPointIt] += plasticStrainIncrement;
        beamsData.endEdit();

        // Updating the new stress point
        newStressPoint = currentStressPoint;

        // Updating the yield stress [isotropic hardening]
        const double previousYieldStress = beamsData.getValue()[index]._yS;
        const double youngModulus = beamsData.getValue()[index]._E;
        const double tangentModulus = m_ConstitutiveLaw->getTangentModulus(previousYieldStress);

        const double H = tangentModulus / (1 - tangentModulus / youngModulus);
        // NB: Updating the yield stress requires the definition of an equivalent plastic
        //     strain, i.e. a scalar value computed from the plastic strain tensor.
        //     Using a Von Mises yield criterion and an associative flow rule, this
        //     equivalent plastic strain can be shown to be equal to the plastic multiplier.
        updateYieldStress(index, H*plasticMultiplier);

    } //end if (!_isPerfectlyPlastic)
}



template< class DataTypes>
void MultiBeamForceField<DataTypes>::computeElasticForce(Eigen::Matrix<double, 12, 1> &internalForces,
                                                         const VecCoord& x, int index, Index a, Index b)
{
    // Here, all Gauss points are assumed to be in an ELASTIC state. Consequently,
    // the internal forces can be computed directly (i.e. not incrementally) with
    // the elastic stiffness matrix, and there is no need to compute a tangent
    // stiffness matrix.
    // A new stress tensor has to be computed (elastically) for the Gauss points,
    // to check whether any of them enters a PLASTIC state. If at least one of
    // them does, we update the mechanical states accordingly, and call
    // computePlasticForces, to carry out the approriate (incremental) computation.

    Displacement localDisp;
    computeLocalDisplacement(x, localDisp, index, a, b);

    // Here _beamMechanicalState = true
    // Consequently, the deformation is plastic. We do not have to recompute
    // the stiffness matrix, but we have to take into account the plastic history
    // for POST-PLASTIC Gauss points

    Eigen::Matrix<double, 12, 1> eigenDepl;
    for (int i = 0; i < 12; i++)
        eigenDepl(i) = localDisp[i];

    //***** Test if we enter in plastic deformation *****//

    const Eigen::Matrix<double, 6, 6>& C = beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[index]._pointMechanicalState;

    MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
    const helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = beamsData.getValue()[index]._plasticStrainHistory;
    const double yieldStress = beamsData.getValue()[index]._yS;
    bool res;
    bool isPlasticBeam = false;
    VoigtTensor2 newStress;
    helper::fixed_array<VoigtTensor2, 27> newStresses;

    //For each Gauss point, we update the stress value for next iteration
    for (int gaussPointIt = 0; gaussPointIt < 27; gaussPointIt++)
    {
        Be = beamsData.getValue()[index]._BeMatrices[gaussPointIt];
        newStress = C*Be*eigenDepl; // Point is assumed to be ELASTIC

        // Checking if the deformation becomes plastic
        bool isNewPlastic = goToPlastic(newStress, yieldStress);
        if (isNewPlastic)
        {
            pointMechanicalState[gaussPointIt] = PLASTIC;
            //TO DO : call to computePlasticForce
        }
        isPlasticBeam = isPlasticBeam || isNewPlastic;

        newStresses[gaussPointIt] = newStress;

    }
    beamsData.endEdit();

    //***************************************************//

    if (isPlasticBeam)
    {
        // The computation of these new stresses should be plastic
        beamMechanicalState = PLASTIC;
        computePlasticForce(internalForces, x, index, a, b);
    }
    else
    {
        // Storing the new stresses for the next time step, in case plasticity occurs.
        _prevStresses[index] = newStresses;

        // As all the points are in an ELASTIC state, it is not necessary
        // to use reduced integration (all the computation is linear).
        nodalForces auxF = beamsData.getValue()[index]._Ke_loc * localDisp;

        for (int i=0; i<12; i++)
            internalForces(i) = auxF[i]; //TO DO: not very efficient, we should settle for one data structure only
    }
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::computePlasticForce(Eigen::Matrix<double, 12, 1> &internalForces,
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
    computeDisplacementIncrement(x, _lastPos, currentDisp, lastDisp, dispIncrement, index, a, b);

    // Converts to Eigen data structure
    EigenDisplacement displacementIncrement;
    for (int k = 0; k < 12; k++)
        displacementIncrement(k) = dispIncrement[k];

    //All the rest of the force computation is made inside of the lambda function
    //as the stress and strain are computed for each Gauss point

    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;

    const Eigen::Matrix<double, 6, 6>& C = beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;

    VoigtTensor2 initialStressPoint = VoigtTensor2::Zero();
    VoigtTensor2 strainIncrement = VoigtTensor2::Zero();
    VoigtTensor2 newStressPoint = VoigtTensor2::Zero();
    double lambdaIncrement = 0;

    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[index]._pointMechanicalState;
    MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
    bool isPlasticBeam = false;
    int gaussPointIt = 0;

    // Computation of the new stress point, through material point iterations as in Krabbenhoft lecture notes

    // This function is to be called if the last stress point corresponded to elastic deformation
    LambdaType computePlastic = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        Be = beamsData.getValue()[index]._BeMatrices[gaussPointIt];
        MechanicalState &mechanicalState = pointMechanicalState[gaussPointIt];

        //Strain
        strainIncrement = Be*displacementIncrement;

        //Stress
        initialStressPoint = _prevStresses[index][gaussPointIt];
        computeStressIncrement(index, gaussPointIt, initialStressPoint, newStressPoint,
                               strainIncrement, lambdaIncrement, mechanicalState, currentDisp);

        isPlasticBeam = isPlasticBeam || (mechanicalState == PLASTIC);

        _prevStresses[index][gaussPointIt] = newStressPoint;

        internalForces += (w1*w2*w3)*Be.transpose()*newStressPoint;

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = beamsData.getValue()[index]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computePlastic);

    // Updates the beam mechanical state information
    if (!isPlasticBeam)
        beamMechanicalState == POSTPLASTIC;

    beamsData.endEdit();

    //Update the tangent stiffness matrix with the new computed stresses
    //This matrix will then be used in addDForce and addKToMatrix methods
    if (isPlasticBeam)
        updateTangentStiffness(index, a, b);
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::computePostPlasticForce(Eigen::Matrix<double, 12, 1> &internalForces,
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

    // Here _beamMechanicalState = true
    // Consequently, the deformation is plastic. We do not have to recompute
    // the stiffness matrix, but we have to take into account the plastic history
    // for POST-PLASTIC Gauss points

    Eigen::Matrix<double, 12, 1> eigenDepl;
    for (int i = 0; i < 12; i++)
        eigenDepl(i) = localDisp[i];

    //***** Test if we enter in plastic deformation *****//
    const Eigen::Matrix<double, 6, 6>& C = beamsData.getValue()[index]._materialBehaviour;
    Eigen::Matrix<double, 6, 12> Be;
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[index]._pointMechanicalState;

    MechanicalState& beamMechanicalState = bd[index]._beamMechanicalState;
    const helper::fixed_array<Eigen::Matrix<double, 6, 1>, 27> &plasticStrainHistory = beamsData.getValue()[index]._plasticStrainHistory;
    const double yieldStress = beamsData.getValue()[index]._yS;
    bool res;
    bool isPlasticBeam = false;
    VoigtTensor2 newStress;
    helper::fixed_array<VoigtTensor2, 27> newStresses;

    //For each Gauss point, we compute the new stress tensor
    // Distinction has to be made depending if the points are ELASTIC or
    // POST-PLASTIC
    for (int gaussPointIt = 0; gaussPointIt < 27; gaussPointIt++)
    {
        Be = beamsData.getValue()[index]._BeMatrices[gaussPointIt];

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
            pointMechanicalState[gaussPointIt] = PLASTIC;
            //TO DO : call to computePlasticForce
        }
        isPlasticBeam = isPlasticBeam || isNewPlastic;

        newStresses[gaussPointIt] = newStress;
    }
    beamsData.endEdit();

    if (isPlasticBeam)
    {
        // The computation of these new stresses should be plastic
        beamMechanicalState == PLASTIC;
        computePlasticForce(internalForces, x, index, a, b);
    }
    else
    {
        // Storing the new stresses for the next time step, in case plasticity occurs.
        _prevStresses[index] = newStresses;

        // Computation of the resulting internal forces, using Gaussian reduced integration.

        typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;
        typedef std::function<void(double, double, double, double, double, double)> LambdaType;

        int gaussPointIt = 0;
        LambdaType computePostPlastic = [&](double u1, double u2, double u3, double w1, double w2, double w3)
        {
            Be = beamsData.getValue()[index]._BeMatrices[gaussPointIt];
            MechanicalState &mechanicalState = pointMechanicalState[gaussPointIt];
            internalForces += (w1*w2*w3)*Be.transpose()*newStresses[gaussPointIt];
            gaussPointIt++; //Next Gauss Point
        };

        ozp::quadrature::detail::Interval<3> interval = beamsData.getValue()[index]._integrationInterval;
        ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computePostPlastic);
    }
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::accumulateNonLinearForce(VecDeriv& f,
                                                              const VecCoord& x,
                                                              int i,
                                                              Index a, Index b)
{
    //Concrete implementation of addForce
    //Computes f += Kx, assuming that this component is linear
    //All non-linearity has to be handled here (including plasticity)

    Eigen::Matrix<double, 12, 1> fint = Eigen::VectorXd::Zero(12);

    const bool beamMechanicalState = beamsData.getValue()[i]._beamMechanicalState;

    if (beamMechanicalState == ELASTIC)
        computeElasticForce(fint, x, i, a, b);
    else if (beamMechanicalState == PLASTIC)
        computePlasticForce(fint, x, i, a, b);
    else
        computePostPlasticForce(fint, x, i, a, b);

    //Passes the contribution to the global system
    nodalForces force;

    for (int i = 0; i < 12; i++)
        force[i] = fint(i);

    //std::cout << "Fint_local pour l'element " << i << " : " << std::endl << force << " " << std::endl << std::endl; //DEBUG

    Vec3 fa1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[0], force[1], force[2]));
    Vec3 fa2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[3], force[4], force[5]));

    Vec3 fb1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[6], force[7], force[8]));
    Vec3 fb2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[9], force[10], force[11]));

    //std::cout << "Fint_monde pour l'element " << i << " : " << std::endl << fa1 << " " << fa2 << " " << fb1 << " " << fb2 << " " << std::endl << std::endl; //DEBUG

    f[a] += Deriv(-fa1, -fa2);
    f[b] += Deriv(-fb1, -fb2);

    //std::cout << "Ftot pour l'element " << i << " : " << std::endl << f << " " << std::endl << std::endl; //DEBUG
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::applyNonLinearStiffness(VecDeriv& df,
                                                             const VecDeriv& dx,
                                                             int i,
                                                             Index a, Index b, double fact)
{
    //Concrete implementation of addDForce
    //Computes df += Kdx, assuming that this component is linear
    //All non-linearity has already been handled through the call to
    //the addForce method

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

    //std::cout << "deplacement vitesse pour l'element " << i << " : " << std::endl << local_depl << " " << std::endl << std::endl; //DEBUG

    // If applyNonLinearStiffness is called, then the beam element is plastic
    // (i.e. at least one Gauss point is in PLASTIC state). Thus the tangent
    // stiffness matrix has to be used
    defaulttype::Vec<12, Real> local_dforce;
    local_dforce = beamsData.getValue()[i]._Kt_loc * local_depl;


    //std::cout << "K*v_local pour l'element " << i << " : " << std::endl << local_dforce << " " << std::endl << std::endl; //DEBUG

    Vec3 fa1 = q.rotate(defaulttype::Vec3d(local_dforce[0], local_dforce[1], local_dforce[2]));
    Vec3 fa2 = q.rotate(defaulttype::Vec3d(local_dforce[3], local_dforce[4], local_dforce[5]));
    Vec3 fb1 = q.rotate(defaulttype::Vec3d(local_dforce[6], local_dforce[7], local_dforce[8]));
    Vec3 fb2 = q.rotate(defaulttype::Vec3d(local_dforce[9], local_dforce[10], local_dforce[11]));

    df[a] += Deriv(-fa1, -fa2) * fact;
    df[b] += Deriv(-fb1, -fb2) * fact;

    //std::cout << "K*v_tot pour l'element " << i << " : " << std::endl << df << " " << std::endl << std::endl; //DEBUG
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::updateTangentStiffness(int i,
                                                            Index a,
                                                            Index b)
{
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    StiffnessMatrix& Kt_loc = bd[i]._Kt_loc;
    const Eigen::Matrix<double, 6, 6>& C = bd[i]._materialBehaviour;
    const double E = bd[i]._E;
    helper::fixed_array<MechanicalState, 27>& pointMechanicalState = bd[i]._pointMechanicalState;

    // Reduced integration
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    // Setting variables for the reduced intergation process defined in quadrature.h
    Eigen::Matrix<double, 6, 12> Be;
    VoigtTensor4 Cep = VoigtTensor4::Zero(); //plastic behaviour tensor
    VoigtTensor2 VMgradient;

    //Auxiliary matrices
    VoigtTensor2 Cgrad;
    Eigen::Matrix<double, 1, 6> gradTC;

    //Result matrix
    Eigen::Matrix<double, 12, 12> tangentStiffness = Eigen::Matrix<double, 12, 12>::Zero();

    VoigtTensor2 currentStressPoint;
    int gaussPointIt = 0;

    // Retrieving H, common to all Gauss points [isotropic hardening]
    double const yieldStress = bd[i]._yS;
    double const tangentModulus = m_ConstitutiveLaw->getTangentModulus(yieldStress);
    double const H = tangentModulus / (1 - tangentModulus / E);

    // Stress matrix, to be integrated
    LambdaType computeTangentStiffness = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        // Be
        Be = bd[i]._BeMatrices[gaussPointIt];

        // Cep
        currentStressPoint = _prevStresses[i][gaussPointIt];

        VMgradient = vonMisesGradient(currentStressPoint, yieldStress);

        if (VMgradient.isZero() || pointMechanicalState[gaussPointIt] != PLASTIC)
            Cep = C; //TO DO: is that correct ?
        else
        {
            Cgrad = C*VMgradient;
            gradTC = VMgradient.transpose()*C;
            //Assuming associative flow rule and isotropic hardening
            Cep = C - (Cgrad*gradTC) / (H + voigtDotProduct(gradTC,VMgradient));
        }

        tangentStiffness += (w1*w2*w3)*Be.transpose()*Cep*Be;

        gaussPointIt++; //Next Gauss Point
    };

    ozp::quadrature::detail::Interval<3> interval = bd[i]._integrationInterval;
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(interval, computeTangentStiffness);

    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            Kt_loc[i][j] = tangentStiffness(i, j);

    //std::cout << "Kt pour l'element " << i << " : " << std::endl << tangentStiffness << " " << std::endl << std::endl; //DEBUG

    beamsData.endEdit();
}
/**************************************************************************/

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_INL
