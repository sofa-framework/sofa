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

#include "../quadrature/Gaussian.h"
#include "../quadrature/quadrature.h"


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
    , _poissonRatio(initData(&_poissonRatio,(Real)0.49f,"poissonRatio","Potion Ratio"))
    , _youngModulus(initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
    , _virtualDisplacementMethod(initData(&_virtualDisplacementMethod, true, "virtualDisplacementMethod", "indicates if the stiffness matrix is computed following the virtual displacement method"))
    , _isPlastic(initData(&_isPlastic, false, "isPlastic", "indicates wether the behaviour model is plastic"))
    , _zSection(initData(&_zSection, (Real)0.2, "zSection", "length of the section in the z direction for rectangular beams"))
    , _ySection(initData(&_ySection, (Real)0.2, "ySection", "length of the section in the y direction for rectangular beams"))
    , _list_segment(initData(&_list_segment,"listSegment", "apply the forcefield to a subset list of beam segments. If no segment defined, forcefield applies to the whole topology"))
    , _useSymmetricAssembly(initData(&_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , _partial_list_segment(false)
    , _updateStiffnessMatrix(true)
    , _assembling(false)
    , edgeHandler(NULL)
{
    edgeHandler = new BeamFFEdgeHandler(this, &beamsData);

    _poissonRatio.setRequired(true);
    _youngModulus.setReadOnly(true);
}

template<class DataTypes>
MultiBeamForceField<DataTypes>::MultiBeamForceField(Real poissonRatio, Real youngModulus, Real zSection, Real ySection, bool useVD, bool isPlastic)
    : beamsData(initData(&beamsData, "beamsData", "Internal element data"))
    , _indexedElements(NULL)
    , _poissonRatio(initData(&_poissonRatio,(Real)poissonRatio,"poissonRatio","Potion Ratio"))
    , _youngModulus(initData(&_youngModulus,(Real)youngModulus,"youngModulus","Young Modulus"))
    , _virtualDisplacementMethod(initData(&_virtualDisplacementMethod, true, "virtualDisplacementMethod", "indicates if the stiffness matrix is computed following the virtual displacement method"))
    , _isPlastic(initData(&_isPlastic, isPlastic, "isPlastic", "indicates wether the behaviour model is plastic"))
    , _zSection(initData(&_zSection, (Real)zSection, "zSection", "length of the section in the z direction for rectangular beams"))
    , _ySection(initData(&_ySection, (Real)ySection, "ySection", "length of the section in the y direction for rectangular beams"))
    , _list_segment(initData(&_list_segment,"listSegment", "apply the forcefield to a subset list of beam segments. If no segment defined, forcefield applies to the whole topology"))
    , _useSymmetricAssembly(initData(&_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , _partial_list_segment(false)
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
        if (_list_segment.getValue().size() == 0)
        {
            sout<<"Forcefield named "<<this->getName()<<" applies to the whole topo"<<sendl;
            _partial_list_segment = false;
        }
        else
        {
            sout<<"Forcefield named "<<this->getName()<<" applies to a subset of edges"<<sendl;
            _partial_list_segment = true;

            for (unsigned int j=0; j<_list_segment.getValue().size(); j++)
            {
                unsigned int i = _list_segment.getValue()[j];
                if (i>=_indexedElements->size())
                {
                    serr<<"WARNING defined listSegment is not compatible with topology"<<sendl;
                    _partial_list_segment = false;
                }
            }
        }
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
    double stiffness, length, poisson, zSection, ySection;
    Index a = (*_indexedElements)[i][0];
    Index b = (*_indexedElements)[i][1];

    const VecCoord& x0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    if (stiffnessContainer)
        stiffness = stiffnessContainer->getStiffness(i) ;
    else
        stiffness =  _youngModulus.getValue() ;

    length = (x0[a].getCenter()-x0[b].getCenter()).norm() ;

    zSection = _zSection.getValue();
    ySection = _ySection.getValue();
    poisson = _poissonRatio.getValue() ;

    setBeam(i, stiffness, length, poisson, zSection, ySection);

    if (_virtualDisplacementMethod.getValue())
    {
        computeMaterialBehaviour(i, a, b);
        computeVDStiffness(i, a, b);
    }
    else
    {
        computeStiffness(i, a, b);

        if (_isPlastic.getValue())
            initPlasticityMatrix(i, a, b);
    }

    initLarge(i,a,b);
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

    if (_partial_list_segment)
    {

        for (unsigned int j=0; j<_list_segment.getValue().size(); j++)
        {
            unsigned int i = _list_segment.getValue()[j];
            Element edge= (*_indexedElements)[i];
            Index a = edge[0];
            Index b = edge[1];
            initLarge(i,a,b);
            accumulateForceLarge( f, p, i, a, b );
        }
    }
    else
    {
        unsigned int i;
        for(it=_indexedElements->begin(),i=0; it!=_indexedElements->end(); ++it,++i)
        {

            Index a = (*it)[0];
            Index b = (*it)[1];

            initLarge(i,a,b);
            accumulateForceLarge( f, p, i, a, b );
        }
    }

    dataF.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams *mparams, DataVecDeriv& datadF , const DataVecDeriv& datadX)
{
    VecDeriv& df = *(datadF.beginEdit());
    const VecDeriv& dx=datadX.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());

    if (_partial_list_segment)
    {
        for (unsigned int j=0; j<_list_segment.getValue().size(); j++)
        {
            unsigned int i = _list_segment.getValue()[j];
            Element edge= (*_indexedElements)[i];
            Index a = edge[0];
            Index b = edge[1];

            applyStiffnessLarge(df, dx, i, a, b, kFactor);
        }
    }
    else
    {
        typename VecElement::const_iterator it;
        unsigned int i = 0;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];

            applyStiffnessLarge(df, dx, i, a, b, kFactor);
        }
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

    std::cout << "k_loc pour l'element " << i << " : " << std::endl;
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 12; j++)
        {
            std::cout << k_loc[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

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

////////////// large displacements method
template<class DataTypes>
void MultiBeamForceField<DataTypes>::initLarge(int i, Index a, Index b)
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    defaulttype::Quat quatA, quatB, dQ;
    Vec3 dW;

    quatA = x[a].getOrientation();
    quatB = x[b].getOrientation();

    quatA.normalize();
    quatB.normalize();

    dQ = qDiff(quatB, quatA);
    dQ.normalize();

    dW = dQ.quatToRotationVector();     // Use of quatToRotationVector instead of toEulerVector:
                                        // this is done to keep the old behavior (before the
                                        // correction of the toEulerVector  function). If the
                                        // purpose was to obtain the Eulerian vector and not the
                                        // rotation vector please use the following line instead
//    dW = dQ.toEulerVector();

    SReal Theta = dW.norm();


    if(Theta>(SReal)0.0000001)
    {
        dW.normalize();

        beamQuat(i) = quatA*dQ.axisToQuat(dW, Theta/2);
        beamQuat(i).normalize();
    }
    else
        beamQuat(i)= quatA;


    beamsData.endEdit();
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
        force = beamsData.getValue()[i]._Ke_loc * depl;

        if (_isPlastic.getValue())
        {
            computePlasticForces(i, a, b, depl, plasticForce);
            force -= plasticForce;
        }
    }
    else
    {
        // this computation can be optimised: (we know that half of "depl" is null)
        force = beamsData.getValue()[i]._k_loc * depl;

        if (_isPlastic.getValue())
        {
            //Update nodal forces to compute plasticity
            //TO DO: use forces from the previous step?
            _nodalForces[i] = force;

            updatePlasticity(i, a, b);
            plasticForce = beamsData.getValue()[i]._M_loc * _VFPlasticStrains[i];
            force -= plasticForce;
        }
    }   

    // Apply lambda transpose (we use the rotation value of point a for the beam)

    Vec3 fa1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[0],force[1],force[2]));
    Vec3 fa2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[3],force[4],force[5]));

    Vec3 fb1 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[6],force[7],force[8]));
    Vec3 fb2 = x[a].getOrientation().rotate(defaulttype::Vec3d(force[9],force[10],force[11]));


    f[a] += Deriv(-fa1, -fa2);
    f[b] += Deriv(-fb1, -fb2);

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

    Vec3 fa1 = q.rotate(defaulttype::Vec3d(local_force[0],local_force[1] ,local_force[2] ));
    Vec3 fa2 = q.rotate(defaulttype::Vec3d(local_force[3],local_force[4] ,local_force[5] ));
    Vec3 fb1 = q.rotate(defaulttype::Vec3d(local_force[6],local_force[7] ,local_force[8] ));
    Vec3 fb2 = q.rotate(defaulttype::Vec3d(local_force[9],local_force[10],local_force[11]));


    df[a] += Deriv(-fa1,-fa2) * fact;
    df[b] += Deriv(-fb1,-fb2) * fact;
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

        if (_partial_list_segment)
        {

            for (unsigned int j=0; j<_list_segment.getValue().size(); j++)
            {

                i = _list_segment.getValue()[j];
                Element edge= (*_indexedElements)[i];
                Index a = edge[0];
                Index b = edge[1];

                //Displacement local_depl;
                //defaulttype::Vec3d u;
                defaulttype::Quat& q = beamQuat(i); //x[a].getOrientation();
                q.normalize();
                Transformation R,Rt;
                q.toMatrix(R);
                Rt.transpose(R);
                StiffnessMatrix K;

                if (_virtualDisplacementMethod.getValue())
                {
                    const StiffnessMatrix& K0 = beamsData.getValue()[i]._Ke_loc;
                    for (int x1 = 0; x1<12; x1 += 3)
                        for (int y1 = 0; y1<12; y1 += 3)
                        {
                            defaulttype::Mat<3, 3, Real> m;
                            K0.getsub(x1, y1, m);
                            m = R*m*Rt;
                            K.setsub(x1, y1, m);
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
                else
                {
                    const StiffnessMatrix& K0 = beamsData.getValue()[i]._k_loc;
                    for (int x1 = 0; x1<12; x1 += 3)
                        for (int y1 = 0; y1<12; y1 += 3)
                        {
                            defaulttype::Mat<3, 3, Real> m;
                            K0.getsub(x1, y1, m);
                            m = R*m*Rt;
                            K.setsub(x1, y1, m);
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

            } //end for _list_segment
        }
        else
        {
            typename VecElement::const_iterator it;
            for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
            {
                Index a = (*it)[0];
                Index b = (*it)[1];

                defaulttype::Quat& q = beamQuat(i); //x[a].getOrientation();
                q.normalize();
                Transformation R,Rt;
                q.toMatrix(R);
                Rt.transpose(R);
                StiffnessMatrix K;
                bool exploitSymmetry = _useSymmetricAssembly.getValue();
               
                if (_virtualDisplacementMethod.getValue())
                {
                   const StiffnessMatrix& K0 = beamsData.getValue()[i]._Ke_loc;
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
            } // end for _indexedElements
        } // end else !_partial_list_segment
    } // end if(r)
}


template<class DataTypes>
void MultiBeamForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    //std::vector< defaulttype::Vector3 > points[3];
    std::vector<defaulttype::Vector3> points[1];

    if (_partial_list_segment)
    {
        for (unsigned int j=0; j<_list_segment.getValue().size(); j++)
            drawElement(_list_segment.getValue()[j], points, x);
    }
    else
    {
        for (unsigned int i=0; i<_indexedElements->size(); ++i)
            drawElement(i, points, x);
    }

    vparams->drawTool()->setPolygonMode(2, true);
    vparams->drawTool()->setLightingEnabled(true);
    vparams->drawTool()->drawHexahedra(points[0], defaulttype::Vec<4,float>(0.24f,0.72f,0.96f,1.0f));
    vparams->drawTool()->setLightingEnabled(false);
    vparams->drawTool()->setPolygonMode(0, false);
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::drawElement(int i, std::vector< defaulttype::Vector3 >* points, const VecCoord& x)
{
    Index a = (*_indexedElements)[i][0];
    Index b = (*_indexedElements)[i][1];
    //sout << "edge " << i << " : "<<a<<" "<<b<<" = "<<x[a].getCenter()<<"  -  "<<x[b].getCenter()<<" = "<<beamsData[i]._L<<sendl;

    defaulttype::Vec3d pa, pb, v0, v1, v2, v3, v4, v5, v6, v7;
    double zDim, yDim;
    pa = x[a].getCenter();
    pb = x[b].getCenter();
    zDim = beamsData.getValue()[i]._zDim;
    yDim = beamsData.getValue()[i]._yDim;

    defaulttype::Vec3d beamVec;
    beamVec[0]=0.0; beamVec[1] = yDim*0.5; beamVec[2] = zDim*0.5;

    //Gathers the vertices of the solid corresponding to a rectangular cross-section beam
    //Vertices are listed starting from the lowest (y,z) coordinates of face A (local frame), counterclockwise
    const defaulttype::Quat& q = beamQuat(i);
    v0 = pa - q.rotate(beamVec);
    v4 = pb - q.rotate(beamVec);
    v2 = pa + q.rotate(beamVec);
    v6 = pb + q.rotate(beamVec);
    beamVec[1] = -yDim*0.5; beamVec[2] = zDim*0.5;
    v1 = pa - q.rotate(beamVec);
    v5 = pb - q.rotate(beamVec);
    v3 = pa + q.rotate(beamVec);
    v7 = pb + q.rotate(beamVec);

    points[0].push_back(v0);
    points[0].push_back(v1);
    points[0].push_back(v2);
    points[0].push_back(v3);
    points[0].push_back(v4);
    points[0].push_back(v5);
    points[0].push_back(v6);
    points[0].push_back(v7);
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::initBeams(size_t size)
{
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    bd.resize(size);
    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::setBeam(unsigned int i, double E, double L, double nu, double zSection, double ySection)
{
    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    bd[i].init(E,L,nu,zSection,ySection);
    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::BeamInfo::init(double E, double L, double nu, double zSection, double ySection)
{
    _E = E;
    _E0 = E;
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

    Ke_loc.clear();

    // Reduced integration
    typedef std::function<void(double, double, double, double, double, double)> LambdaType;
    typedef ozp::quadrature::Gaussian<3> GaussianQuadratureType;

    // Setting variables for the reduced intergation process defined in quadrature.h

    Eigen::Matrix<double, 6, 12> Be;
    Eigen::Matrix<double, 6, 12> BeT;
    Eigen::Matrix<double, 12, 12> stressMatrix;

    // Stress matrix, to be integrated
    LambdaType computeStressMatrix = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        double xi = u1 / _L;
        double eta = u2 / _L;
        double zeta = u3 / _L;

        // Be
        Be(0, 0) = -1 / _L;
        Be(1, 0) = Be(2, 0) = Be(3, 0) = Be(4, 0) = Be(5, 0) = 0.0;

        Be(0, 1) = (6 * eta*(1 - 2 * xi)) / _L;
        Be(1, 1) = Be(2, 1) = Be(3, 1) = Be(4, 1) = Be(5, 1) = 0.0;

        Be(0, 2) = (6 * zeta*(1 - 2 * xi)) / _L;
        Be(1, 2) = Be(2, 2) = Be(3, 2) = Be(4, 2) = Be(5, 2) = 0.0;

        Be(0, 3) = Be(1, 3) = Be(2, 3) = 0.0;
        Be(3, 3) = xi - 1;
        Be(4, 3) = eta / 2;
        Be(5, 3) = zeta / 2;

        Be(0, 4) = zeta * (6 * xi - 4);
        Be(1, 4) = Be(2, 4) = Be(3, 4) = Be(4, 4) = Be(5, 4) = 0.0;

        Be(0, 5) = eta * (4 - 6 * xi);
        Be(1, 5) = Be(2, 5) = Be(3, 5) = Be(4, 5) = Be(5, 5) = 0.0;

        Be.block<6, 1>(0, 6) = -Be.block<6, 1>(0, 0);

        Be.block<6, 1>(0, 7) = -Be.block<6, 1>(0, 1);

        Be.block<6, 1>(0, 8) = -Be.block<6, 1>(0, 2);

        Be(0, 9) = Be(1, 9) = Be(2, 9) = 0.0;
        Be(3, 9) = -xi;
        Be(4, 9) = -eta / 2;
        Be(5, 9) = -zeta / 2;

        Be(0, 10) = zeta * (6 * xi - 2);
        Be(1, 10) = Be(2, 10) = Be(3, 10) = Be(4, 10) = Be(5, 10) = 0.0;

        Be(0, 11) = eta * (2 - 6 * xi);
        Be(1, 11) = Be(2, 11) = Be(3, 11) = Be(4, 11) = Be(5, 11) = 0.0;

        stressMatrix = Be.transpose()*C*Be;
        stressMatrix *= w1*w2*w3;

        for (int i = 0; i < 12;  i++)
            for (int j= 0; j < 12; j++)
                Ke_loc[i][j] += stressMatrix(i, j);

    };

    ozp::quadrature::detail::Interval<3> integrationInterval = ozp::quadrature::make_interval(0, -_yDim/2, -_zDim/2, _L, _yDim/2, _zDim/2);
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(integrationInterval, computeStressMatrix);

    std::cout << "Ke_loc pour l'element " << i << " : " << std::endl;
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 12; j++)
        {
            std::cout << Ke_loc[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

    beamsData.endEdit();
}

template<class DataTypes>
void MultiBeamForceField<DataTypes>::computeMaterialBehaviour(int i, Index a, Index b)
{

    helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
    Eigen::Matrix<double, 6, 6>& C = bd[i]._materialBehaviour;

    Real youngModulus = (Real)beamsData.getValue()[i]._E;
    Real poissonRatio = (Real)beamsData.getValue()[i]._nu;

    // Material behaviour matrix, here: Hooke's law
    C(0, 0) = C(1, 1) = C(2, 2) = 1;
    C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = poissonRatio / (1 - poissonRatio);
    C(0, 3) = C(0, 4) = C(0, 5) = 0;
    C(1, 3) = C(1, 4) = C(1, 5) = 0;
    C(2, 3) = C(2, 4) = C(2, 5) = 0;
    C(3, 0) = C(3, 1) = C(3, 2) = C(3, 4) = C(3, 5) = 0;
    C(4, 0) = C(4, 1) = C(4, 2) = C(4, 3) = C(4, 5) = 0;
    C(5, 0) = C(5, 1) = C(5, 2) = C(5, 3) = C(5, 4) = 0;
    C(3, 3) = C(4, 4) = C(5, 5) = (1 - 2 * poissonRatio) / (2 * (1 - poissonRatio));
    C *= (youngModulus*(1 - poissonRatio)) / ((1 + poissonRatio)*(1 - 2 * poissonRatio));

    std::cout << "C pour l'element " << i << " : " << std::endl << C << " " << std::endl << std::endl;

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
    Eigen::Matrix<double, 6, 12> BeT;
    Eigen::Matrix<double, 12, 1> fe_pla = Eigen::VectorXd::Zero(12);
    int gaussPointIterator = 0;

    // Stress matrix, to be integrated
    LambdaType computeStressMatrix = [&](double u1, double u2, double u3, double w1, double w2, double w3)
    {
        // Step 1: total strain computation
        double xi = u1 / _L;
        double eta = u2 / _L;
        double zeta = u3 / _L;

        // Be
        Be(0, 0) = -1 / _L;
        Be(1, 0) = Be(2, 0) = Be(3, 0) = Be(4, 0) = Be(5, 0) = 0.0;

        Be(0, 1) = (6 * eta*(1 - 2 * xi)) / _L;
        Be(1, 1) = Be(2, 1) = Be(3, 1) = Be(4, 1) = Be(5, 1) = 0.0;

        Be(0, 2) = (6 * zeta*(1 - 2 * xi)) / _L;
        Be(1, 2) = Be(2, 2) = Be(3, 2) = Be(4, 2) = Be(5, 2) = 0.0;

        Be(0, 3) = Be(1, 3) = Be(2, 3) = 0.0;
        Be(3, 3) = xi - 1;
        Be(4, 3) = eta / 2;
        Be(5, 3) = zeta / 2;

        Be(0, 4) = zeta * (6 * xi - 4);
        Be(1, 4) = Be(2, 4) = Be(3, 4) = Be(4, 4) = Be(5, 4) = 0.0;

        Be(0, 5) = eta * (4 - 6 * xi);
        Be(1, 5) = Be(2, 5) = Be(3, 5) = Be(4, 5) = Be(5, 5) = 0.0;

        Be.block<6, 1>(0, 6) = -Be.block<6, 1>(0, 0);

        Be.block<6, 1>(0, 7) = -Be.block<6, 1>(0, 1);

        Be.block<6, 1>(0, 8) = -Be.block<6, 1>(0, 2);

        Be(0, 9) = Be(1, 9) = Be(2, 9) = 0.0;
        Be(3, 9) = -xi;
        Be(4, 9) = -eta / 2;
        Be(5, 9) = -zeta / 2;

        Be(0, 10) = zeta * (6 * xi - 2);
        Be(1, 10) = Be(2, 10) = Be(3, 10) = Be(4, 10) = Be(5, 10) = 0.0;

        Be(0, 11) = eta * (2 - 6 * xi);
        Be(1, 11) = Be(2, 11) = Be(3, 11) = Be(4, 11) = Be(5, 11) = 0.0;

        totalStrain = Be*totalDisp;

        //Step 2: test of the plasticity threshold
        updatePlasticStrain(i, a, b, totalStrain, gaussPointIterator);
        
        //Step3: addition of this Gauss point contribution to the plastic forces
        VoigtTensor plasticStrain = _VDPlasticStrains[i].row(gaussPointIterator);
        fe_pla += (w1*w2*w3)*Be.transpose()*C*plasticStrain;
        gaussPointIterator++;
    };

    ozp::quadrature::detail::Interval<3> integrationInterval = ozp::quadrature::make_interval(0, -_yDim / 2, -_zDim / 2, _L, _yDim / 2, _zDim / 2);
    ozp::quadrature::integrate <GaussianQuadratureType, 3, LambdaType>(integrationInterval, computeStressMatrix);

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

/**************************************************************************/


/**************************************************************************/
/*                          Plasticity Handler                            */
/**************************************************************************/

template< class DataTypes>
MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::MultiBeamPlasticityHandler(MultiBeamForceField<DataTypes>* ff)
    : ff(ff)
{
    _NRThreshold = 0.0; //to be changed during iterations
    _NRMaxIterations = 25;

    //Found UTS of 655MPa for an ASTM F75-07 cobalt-chrome alloy
    _UTS = 655000000.0;

    _tangentStiffness = tangentStiffnessMatrix::Zero();
    _nodalForces = plasticNodalForces::Zero();

    //Initialises the lastPos field with the rest position
    _lastPos = ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::updateIncrements(tangentStiffnessMatrix &tangentStiffness,
                                                                                  plasticNodalForces &plasticForces,
                                                                                  const plasticNodalForces &externalLoad)
{
    //Algo table 2 de Krabbenhoft

    // en local :
    //      plasticNodalForces externLoad (f_k), loadIncrement
    //      Displacement disp (u_k), dispIncrement
    //      matrice B
    //      VoigtTensor2 strainIncrement (DELTA_epsilon_k), stressIncrement (DELTA_sigma_k)
    //      ( plasticNodalForce residual )
    //      Structures pour l'intégration numérique (intervalle et points de Gauss x 1 // fonctions lambda x 2, pour K et q)
}


template< class DataTypes>
bool MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::inPlasticDeformation(const VoigtTensor2 &stressTensor)
{
    double yield = vonMisesYield(stressTensor, _UTS);
    return yield > 0;
}


template< class DataTypes>
bool MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::outOfPlasticDeformation(const VoigtTensor2 &stressTensor,
                                                                                         const VoigtTensor2 &stressIncrement)
{
    Eigen::Matrix<double, 6, 1> gradient = vonMisesGradient(stressTensor, _UTS);
    double cp = gradient.transpose()*stressIncrement;
    return cp < 0;
}


template< class DataTypes>
double MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::vonMisesYield(const VoigtTensor2 &stressTensor,
                                                                                 const double UTS)
{
    double res = 0.0;
    double sigmaX = stressTensor[0];
    double sigmaY = stressTensor[1];
    double sigmaZ = stressTensor[2];
    double sigmaYZ = stressTensor[3];
    double sigmaZX = stressTensor[4];
    double sigmaXY = stressTensor[5];

    double aux1 = 0.5*( (sigmaX - sigmaY)*(sigmaX - sigmaY) + (sigmaY - sigmaZ)*(sigmaY - sigmaZ) + (sigmaZ - sigmaX)*(sigmaZ - sigmaX) );
    double aux2 = 3.0*(sigmaYZ*sigmaYZ + sigmaZX*sigmaZX + sigmaXY*sigmaXY);

    res = helper::rsqrt(aux1 + aux2) - UTS;
    return res;
}


template< class DataTypes>
Eigen::Matrix<double, 6, 1> MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::vonMisesGradient(const VoigtTensor2 &stressTensor,
                                                                                                         const double UTS)
{
    VoigtTensor2 res = VoigtTensor2::Zero();
    double sigmaX = stressTensor[0];
    double sigmaY = stressTensor[1];
    double sigmaZ = stressTensor[2];
    double sigmaYZ = stressTensor[3];
    double sigmaZX = stressTensor[4];
    double sigmaXY = stressTensor[5];

    double fact = 1 / 2 * (vonMisesYield(stressTensor, UTS) + UTS);

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
Eigen::Matrix<double, 6, 6> MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::vonMisesHessian(const VoigtTensor2 &stressTensor,
                                                                                                        const double UTS)
{
    VoigtTensor4 res = VoigtTensor4::Zero();

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
    double sigmaE = vonMisesYield(stressTensor, UTS) + UTS;
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

template< class DataTypes>
void MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::solveDispIncrement(const tangentStiffnessMatrix &tangentStiffness,
                                                                                    Displacement &du,
                                                                                    const plasticNodalForces &residual)
{
    //Solve the linear system K*du = residual, for du

    //First try is with the dense LU decomposition provided by the Eigen library
    //NB: this is not inplace decomposition because we pass a const reference as argument
    Eigen::FullPivLU<tangentStiffnessMatrix> LU(tangentStiffness);
    du = LU.solve(residual);
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::computeStressIncrement(int i,
                                                                                        const VoigtTensor2 &initialStress,
                                                                                        VoigtTensor2 &stressIncrement,
                                                                                        const VoigtTensor2 &strainIncrement,
                                                                                        double &lambdaIncrement)
{
    /** Material point iterations **/
    //NB: we consider that the yield function and the plastic flow are equal (f=g)
    //    This corresponds to an associative flow rule (for plasticity)

    unsigned int nbMaxIterations = 25;
    double threshold = 1; //TO DO: choose coherent value

    const Eigen::Matrix<double, 6, 6>& C = ff->beamsData.getValue()[i]._materialBehaviour; //Matrix D in Krabbenhoft's


    /*****************************/
    /*   Plastic return method   */
    /*****************************/

    ////First we compute the elastic predictor
    //VoigtTensor2 elasticPredictor = C*strainIncrement;

    //VoigtTensor2 currentStressPoint = initialStress + elasticPredictor; //point B in Krabbenhoft's

    //VoigtTensor2 grad, plasticStressIncrement;
    //double denum, dLambda;
    //double fSigma = vonMisesYield(currentStressPoint, _UTS); //for the 1st iteration

    ////We start iterations with point B
    //for (unsigned int i = 0; i < nbMaxIterations; i++)
    //{
    //    grad = vonMisesGradient(currentStressPoint, _UTS);
    //    denum = grad.transpose()*C*grad;
    //    dLambda = fSigma / denum;

    //    plasticStressIncrement = dLambda*C*grad;

    //    currentStressPoint -= plasticStressIncrement;

    //    fSigma = vonMisesYield(currentStressPoint, _UTS);
    //    if (fSigma < iterationThreshold)
    //        break; //Yield condition : the current stress point is on the yield surface
    //}
    //// return the total stress increment
    ////TO DO: return the current stress directly instead?
    //stressIncrement = currentStressPoint - initialStress;

    /*****************************/
    /*      Implicit method      */
    /*****************************/

    //First we compute the elastic predictor
    VoigtTensor2 elasticPredictor = C*strainIncrement;
    VoigtTensor2 currentStressPoint = initialStress + elasticPredictor;

    Eigen::Matrix<double, 7, 1> totalIncrement = Eigen::Matrix<double, 7, 1>::Zero();
    totalIncrement.block<6, 1>(0, 0) = elasticPredictor;

    Eigen::Matrix<double, 7, 1> newIncrement= Eigen::Matrix<double, 7, 1>::Zero();

    //Intermediate variables
    Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
    VoigtTensor2 gradient = vonMisesGradient(currentStressPoint, _UTS);
    VoigtTensor4 hessian;
    double yieldCondition;

    //Initialisation of the Jacobian of the nonlinear system of equations
    Eigen::Matrix<double, 7, 7> J = Eigen::Matrix<double, 7, 7>::Zero();
    J.block<6, 6>(0, 0) = I6; //at first iteration, dLambda = 0
    J.block<6, 1>(0, 6) = C*gradient;
    J.block<1, 6>(6, 0) = gradient.transpose();

    //Initialisation of the second member
    Eigen::Matrix<double, 7, 1> b = Eigen::Matrix<double, 7, 1>::Zero();
    //In the first iteration, the (first) stressIncrement is taken as the elasticPredictor
    //and the lambda increment as 0. Hence b is null

    //solver
    Eigen::FullPivLU<Eigen::Matrix<double, 7, 7> > LU(J.rows(), J.cols());

    for (unsigned int i = 0; i < nbMaxIterations; i++)
    {
        //Computes the new increment
        LU.compute(J);
        newIncrement = LU.solve(-b);

        totalIncrement += newIncrement;

        //Updates J and b for the next iteration
        currentStressPoint = initialStress + totalIncrement.block<6, 1>(0, 0);
        gradient = vonMisesGradient(currentStressPoint, _UTS);
        hessian = vonMisesHessian(currentStressPoint, _UTS);

        J.block<6, 6>(0, 0) = I6 + totalIncrement(6, 0)*C*hessian;
        J.block<6, 1>(0, 6) = C*gradient;
        J.block<1, 6>(6, 0) = gradient.transpose();

        yieldCondition = vonMisesYield(currentStressPoint, _UTS);
        if (yieldCondition < threshold)
            break; //We reach a stress point close enough to the yield surface

        b.block<6, 1>(0, 0) = totalIncrement.block<6, 1>(0, 0) - elasticPredictor + C*totalIncrement(6, 0)*gradient;
        b(6, 0) = yieldCondition;
    }

    // return the total stress increment
    //TO DO: return the current stress directly instead?
    stressIncrement = currentStressPoint - initialStress;
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::accumulateNonLinearForce(VecDeriv& f,
                                                                                          const VecCoord& x,
                                                                                          int i,
                                                                                          Index a, Index b)
{
    //Concrete implementation of addForce
    //Computes f += Kx, assuming that this component is linear
    //All non-linearity has to be handled here (including plasticity)

}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::applyNonLinearStiffness(VecDeriv& f,
                                                                                         const VecDeriv& x,
                                                                                         int i,
                                                                                         Index a, Index b,
                                                                                         double fact = 1.0)
{
    //Concrete implementation of addDForce
}


template< class DataTypes>
void MultiBeamForceField<DataTypes>::MultiBeamPlasticityHandler::updateTangentStiffness(int i,
                                                                                        Index a,
                                                                                        Index b)
{

}
/**************************************************************************/

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_MULTIBEAMFORCEFIELD_INL
