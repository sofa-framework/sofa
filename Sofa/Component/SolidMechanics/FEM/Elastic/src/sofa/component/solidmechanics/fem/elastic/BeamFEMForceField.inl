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
#include <sofa/component/solidmechanics/fem/elastic/BeamFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.inl>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/rmath.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>


namespace sofa::component::solidmechanics::fem::elastic::_beamfemforcefield_
{

using core::objectmodel::BaseContext;
using type::Quat;

template<class DataTypes>
BeamFEMForceField<DataTypes>::BeamFEMForceField()
    : BeamFEMForceField(
        BaseLinearElasticityFEMForceField<DataTypes>::defaultPoissonRatioValue,
        BaseLinearElasticityFEMForceField<DataTypes>::defaultYoungModulusValue,
        0.1, 0.)
{}

template<class DataTypes>
BeamFEMForceField<DataTypes>::BeamFEMForceField(Real poissonRatio, Real youngModulus, Real radius, Real radiusInner)
    : d_beamsData(initData(&d_beamsData, "beamsData", "Internal element data"))
    , m_indexedElements(nullptr)
    , d_radius(initData(&d_radius, radius,"radius","radius of the section"))
    , d_radiusInner(initData(&d_radiusInner, radiusInner,"radiusInner","inner radius of the section for hollow beams"))
    , d_listSegment(initData(&d_listSegment,"listSegment", "apply the forcefield to a subset list of beam segments. If no segment defined, forcefield applies to the whole topology"))
    , d_useSymmetricAssembly(initData(&d_useSymmetricAssembly,false,"useSymmetricAssembly","use symmetric assembly of the matrix K"))
    , m_partialListSegment(false)
    , m_updateStiffnessMatrix(true)
{
    d_radius.setRequired(true);

    this->setPoissonRatio(poissonRatio);
    this->setYoungModulus(youngModulus);
}


template<class DataTypes>
BeamFEMForceField<DataTypes>::~BeamFEMForceField()
{
}


template <class DataTypes>
void BeamFEMForceField<DataTypes>::init()
{
    Inherit1::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    if(this->l_topology->getNbEdges()==0)
    {
        msg_error() << "No edge found in the topology " << this->l_topology.getLinkedPath();
        return;
    }
    m_indexedElements = &this->l_topology->getEdges();
    if (!d_listSegment.getValue().empty())
    {
        msg_info() << "Forcefield named " << this->getName() << " applies to a subset of edges.";
        m_partialListSegment = true;

        for (unsigned int j = 0; j < d_listSegment.getValue().size(); j++)
        {
            const unsigned int i = d_listSegment.getValue()[j];
            if (i >= m_indexedElements->size())
            {
                msg_warning() << "Defined listSegment is not compatible with topology";
                m_partialListSegment = false;
            }
        }
    } else
    {
        msg_info() << "Forcefield named " << this->getName() << " applies to the wholo topo.";
        m_partialListSegment = false;
    }

    d_beamsData.createTopologyHandler(this->l_topology);
    d_beamsData.setCreationCallback([this](Index edgeIndex, BeamInfo& ei,
                                           const core::topology::BaseMeshTopology::Edge& edge,
                                           const sofa::type::vector< Index >& ancestors,
                                           const sofa::type::vector< SReal >& coefs)
    {
        createBeamInfo(edgeIndex, ei, edge, ancestors, coefs);
    });

    reinit();
}

template <class DataTypes>
void BeamFEMForceField<DataTypes>::reinit()
{
    if (!m_indexedElements)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    unsigned int n = m_indexedElements->size();
    m_forces.resize( this->mstate->getSize() );

    initBeams( n );
    for (unsigned int i=0; i<n; ++i)
        reinitBeam(i);
    msg_info() << "Reinit OK, "<<n<<" elements." ;
}

template <class DataTypes>
void BeamFEMForceField<DataTypes>::reinitBeam(Index i)
{
    if (!m_indexedElements)
        return;

    SReal stiffness, length, radius, poisson, radiusInner;
    const auto& [a, b] = (*m_indexedElements)[i].array();

    const VecCoord& x0 = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    stiffness =  this->getYoungModulusInElement(i);

    length = (x0[a].getCenter()-x0[b].getCenter()).norm() ;

    radius = d_radius.getValue() ;
    radiusInner = d_radiusInner.getValue();
    poisson = this->getPoissonRatioInElement(i) ;


    setBeam(i, stiffness, length, poisson, radius, radiusInner);

    computeStiffness(i,a,b);

    initLarge(i,a,b);
}

template< class DataTypes>
void BeamFEMForceField<DataTypes>::createBeamInfo(Index edgeIndex, BeamInfo &ei,
    const core::topology::BaseMeshTopology::Edge &,
    const sofa::type::vector<Index> &,
    const sofa::type::vector<SReal> &)
{
    reinitBeam(edgeIndex);
    ei = d_beamsData.getValue()[edgeIndex];
}

template<class DataTypes>
Quat<SReal>& BeamFEMForceField<DataTypes>::beamQuat(int i)
{
    helper::WriteAccessor<Data<type::vector<BeamInfo> > > bd = d_beamsData;
    return bd[i].quat;
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* mparams,
                                            DataVecDeriv &  dataF,
                                            const DataVecCoord &  dataX ,
                                            const DataVecDeriv &dataV )
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(dataV);

    if (!m_indexedElements)
        return;

    helper::WriteAccessor<Data<VecDeriv> > f = dataF;
    const VecCoord& p=dataX.getValue();
    f.resize(p.size());

    //// First compute each node rotation
    typename VecElement::const_iterator it;

    if (m_partialListSegment)
    {
        for (unsigned int i : d_listSegment.getValue())
        {
            const auto& [a, b] = (*m_indexedElements)[i].array();
            initLarge(i, a, b);
            accumulateForceLarge(f.wref(), p, i, a, b);
        }
    }
    else
    {
        unsigned int i;
        for(it=m_indexedElements->begin(),i=0; it!=m_indexedElements->end(); ++it,++i)
        {
            const auto& [a, b] = it->array();
            initLarge(i, a, b);
            accumulateForceLarge(f.wref(), p, i, a, b);
        }
    }
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams *mparams, DataVecDeriv& datadF , const DataVecDeriv& datadX)
{
    if (!m_indexedElements)
        return;

    helper::WriteAccessor<Data<VecDeriv> > df = datadF;
    const VecDeriv& dx=datadX.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    df.resize(dx.size());

    if (m_partialListSegment)
    {
        for (unsigned int i : d_listSegment.getValue())
        {
            const auto& [a, b] = (*m_indexedElements)[i].array();
            applyStiffnessLarge(df.wref(), dx, i, a, b, kFactor);
        }
    }
    else
    {
        unsigned int i = 0;
        for(auto it = m_indexedElements->begin() ; it != m_indexedElements->end() ; ++it, ++i)
        {
            const auto& [a, b] = it->array();
            applyStiffnessLarge(df.wref(), dx, i, a, b, kFactor);
        }
    }
}

template<class DataTypes>
typename BeamFEMForceField<DataTypes>::Real BeamFEMForceField<DataTypes>::pseudoDeterminantForCoef ( const type::Mat<2, 3, Real>&  M )
{
    return  M(0,1)*M(1,2) - M(1,1)*M(0,2) -  M(0,0)*M(1,2) + M(1,0)*M(0,2) + M(0,0)*M(1,1) - M(1,0)*M(0,1);
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeStiffness(int i, Index , Index )
{
    Real   phiy, phiz;
    Real _L = (Real)d_beamsData.getValue()[i]._L;
    Real _A = (Real)d_beamsData.getValue()[i]._A;
    Real _nu = (Real)d_beamsData.getValue()[i]._nu;
    Real _E = (Real)d_beamsData.getValue()[i]._E;
    Real _Iy = (Real)d_beamsData.getValue()[i]._Iy;
    Real _Iz = (Real)d_beamsData.getValue()[i]._Iz;
    Real _Asy = (Real)d_beamsData.getValue()[i]._Asy;
    Real _Asz = (Real)d_beamsData.getValue()[i]._Asz;
    Real _G = (Real)d_beamsData.getValue()[i]._G;
    Real _J = (Real)d_beamsData.getValue()[i]._J;
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

    helper::WriteAccessor<Data<type::vector<BeamInfo> > > bd = d_beamsData;
    StiffnessMatrix& k_loc = bd[i]._k_loc;

    // Define stiffness matrix 'k' in local coordinates
    k_loc.clear();
    k_loc(6,6)   = k_loc(0,0)   = _E*_A/_L;
    k_loc(7,7)   = k_loc(1,1)   = (Real)(12.0*EIz/(L3*(1.0+phiy)));
    k_loc(8,8)   = k_loc(2,2)   = (Real)(12.0*EIy/(L3*(1.0+phiz)));
    k_loc(9,9)   = k_loc(3,3)   = _G*_J/_L;
    k_loc(10,10) = k_loc(4,4)   = (Real)((4.0+phiz)*EIy/(_L*(1.0+phiz)));
    k_loc(11,11) = k_loc(5,5)   = (Real)((4.0+phiy)*EIz/(_L*(1.0+phiy)));

    k_loc(4,2)   = (Real)(-6.0*EIy/(L2*(1.0+phiz)));
    k_loc(5,1)   = (Real)( 6.0*EIz/(L2*(1.0+phiy)));
    k_loc(6,0)   = -k_loc(0,0);
    k_loc(7,1)   = -k_loc(1,1);
    k_loc(7,5)   = -k_loc(5,1);
    k_loc(8,2)   = -k_loc(2,2);
    k_loc(8,4)   = -k_loc(4,2);
    k_loc(9,3)   = -k_loc(3,3);
    k_loc(10,2)  = k_loc(4,2);
    k_loc(10,4)  = (Real)((2.0-phiz)*EIy/(_L*(1.0+phiz)));
    k_loc(10,8)  = -k_loc(4,2);
    k_loc(11,1)  = k_loc(5,1);
    k_loc(11,5)  = (Real)((2.0-phiy)*EIz/(_L*(1.0+phiy)));
    k_loc(11,7)  = -k_loc(5,1);

    for (int i=0; i<=10; i++)
        for (int j=i+1; j<12; j++)
            k_loc(i,j) = k_loc(j,i);
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
    const type::Quat<SReal> q = b.inverse() * a;
    return q;
}

////////////// large displacements method
template<class DataTypes>
void BeamFEMForceField<DataTypes>::initLarge(int i, Index a, Index b)
{
    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    type::Quat<SReal> quatA, quatB, dQ;
    Vec3 dW;

    quatA = x[a].getOrientation();
    quatB = x[b].getOrientation();

    quatA.normalize();
    quatB.normalize();

    dQ = qDiff(quatB, quatA);
    dQ.normalize();

    // TODO(e.coevoet) remove before v20.12
    // Use of quatToRotationVector instead of toEulerVector: dW = dQ.quatToRotationVector();
    // this is done to keep the old behavior (before the
    // correction of the toEulerVector  function). If the
    // purpose was to obtain the Eulerian vector and not the
    // rotation vector please use the following line instead
    // dW = dQ.toEulerVector();
    dW = dQ.quatToRotationVector();

    const SReal Theta = dW.norm();

    if(Theta>(SReal)0.0000001)
    {
        dW.normalize();

        beamQuat(i) = quatA*dQ.axisToQuat(dW, Theta/2);
        beamQuat(i).normalize();
    }
    else
        beamQuat(i)= quatA;
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::accumulateForceLarge( VecDeriv& f, const VecCoord & x, int i, Index a, Index b )
{
    const VecCoord& x0 = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();

    beamQuat(i)= x[a].getOrientation();
    beamQuat(i).normalize();

    type::Vec<3,Real> u, P1P2, P1P2_0;

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
    type::Quat<SReal> dQ0, dQ;

    dQ0 = qDiff(x0[b].getOrientation(), x0[a].getOrientation());
    dQ =  qDiff(x[b].getOrientation(), x[a].getOrientation());

    dQ0.normalize();
    dQ.normalize();

    type::Quat<SReal> tmpQ = qDiff(dQ,dQ0);
    tmpQ.normalize();

    // TODO(e.coevoet) remove before v20.12
    // Use of quatToRotationVector instead of toEulerVector: u = tmpQ.quatToRotationVector();
    // this is done to keep the old behavior (before the
    // correction of the toEulerVector  function). If the
    // purpose was to obtain the Eulerian vector and not the
    // rotation vector please use the following line instead
    // u = tmpQ.toEulerVector();
    u = tmpQ.quatToRotationVector();

    depl[3] = 0.0; 	depl[4] = 0.0; 	depl[5] = 0.0;
    depl[9] = u[0]; depl[10]= u[1]; depl[11]= u[2];

    // this computation can be optimised: (we know that half of "depl" is null)
    Displacement force = d_beamsData.getValue()[i]._k_loc * depl;


    // Apply lambda transpose (we use the rotation value of point a for the beam)
    const Vec3 fa1 = x[a].getOrientation().rotate(type::Vec3d(force[0],force[1],force[2]));
    const Vec3 fa2 = x[a].getOrientation().rotate(type::Vec3d(force[3],force[4],force[5]));

    const Vec3 fb1 = x[a].getOrientation().rotate(type::Vec3d(force[6],force[7],force[8]));
    const Vec3 fb2 = x[a].getOrientation().rotate(type::Vec3d(force[9],force[10],force[11]));

    f[a] += Deriv(-fa1, -fa2);
    f[b] += Deriv(-fb1, -fb2);

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::applyStiffnessLarge(VecDeriv& df, const VecDeriv& dx, int i, Index a, Index b, SReal fact)
{
    Displacement local_depl;
    type::Vec<3,Real> u;
    type::Quat<SReal>& q = beamQuat(i);
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

    Displacement local_force = d_beamsData.getValue()[i]._k_loc * local_depl;

    const Vec3 fa1 = q.rotate(type::Vec3d(local_force[0],local_force[1] ,local_force[2] ));
    const Vec3 fa2 = q.rotate(type::Vec3d(local_force[3],local_force[4] ,local_force[5] ));
    const Vec3 fb1 = q.rotate(type::Vec3d(local_force[6],local_force[7] ,local_force[8] ));
    const Vec3 fb2 = q.rotate(type::Vec3d(local_force[9],local_force[10],local_force[11]));


    df[a] += Deriv(-fa1,-fa2) * fact;
    df[b] += Deriv(-fb1,-fb2) * fact;
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset)
{
    if (!m_indexedElements)
        return;

    if (m_partialListSegment)
    {

        for (unsigned int i : d_listSegment.getValue())
        {
            const auto& [a, b] = (*m_indexedElements)[i].array();

            type::Quat<SReal>& q = beamQuat(i);
            q.normalize();
            Transformation R,Rt;
            q.toMatrix(R);
            Rt.transpose(R);
            const StiffnessMatrix& K0 = d_beamsData.getValue()[i]._k_loc;
            StiffnessMatrix K;
            for (int x1=0; x1<12; x1+=3)
                for (int y1=0; y1<12; y1+=3)
                {
                    type::Mat<3,3,Real> m;
                    K0.getsub(x1,y1, m);
                    m = R*m*Rt;
                    K.setsub(x1,y1, m);
                }
            int index[12];
            for (int x1=0; x1<6; x1++)
                index[x1] = offset+a*6+x1;
            for (int x1=0; x1<6; x1++)
                index[6+x1] = offset+b*6+x1;
            for (int x1=0; x1<12; ++x1)
                for (int y1=0; y1<12; ++y1)
                    matrix->add(index[x1], index[y1], - K(x1,y1)*kFact);

        }

    }
    else
    {
        unsigned int i {};
        for(auto it = m_indexedElements->begin() ; it != m_indexedElements->end() ; ++it, ++i)
        {
            const auto& [a, b] = it->array();

            type::Quat<SReal>& q = beamQuat(i);
            q.normalize();
            Transformation R,Rt;
            q.toMatrix(R);
            Rt.transpose(R);
            const StiffnessMatrix& K0 = d_beamsData.getValue()[i]._k_loc;
            StiffnessMatrix K;
            const bool exploitSymmetry = d_useSymmetricAssembly.getValue();

            if (exploitSymmetry) {
                for (int x1=0; x1<12; x1+=3) {
                    for (int y1=x1; y1<12; y1+=3)
                    {
                        type::Mat<3,3,Real> m;
                        K0.getsub(x1,y1, m);
                        m = R*m*Rt;

                        for (int i=0; i<3; i++)
                            for (int j=0; j<3; j++) {
                                K(i+x1,j+y1) += m(i,j);
                                K(j+y1,i+x1) += m(i,j);
                            }
                        if (x1 == y1)
                            for (int i=0; i<3; i++)
                                for (int j=0; j<3; j++)
                                    K(i+x1,j+y1) *= SReal(0.5);

                    }
                }
            } else  {
                for (int x1=0; x1<12; x1+=3) {
                    for (int y1=0; y1<12; y1+=3)
                    {
                        type::Mat<3,3,Real> m;
                        K0.getsub(x1,y1, m);
                        m = R*m*Rt;
                        K.setsub(x1,y1, m);
                    }
                }
            }

            int index[12];
            for (int x1=0; x1<6; x1++)
                index[x1] = offset+a*6+x1;
            for (int x1=0; x1<6; x1++)
                index[6+x1] = offset+b*6+x1;
            for (int x1=0; x1<12; ++x1)
                for (int y1=0; y1<12; ++y1)
                    matrix->add(index[x1], index[y1], - K(x1,y1)*kFact);

        }
    }
}

template <class DataTypes>
void BeamFEMForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    if (m_partialListSegment)
    {
        for (unsigned int i : d_listSegment.getValue())
        {
            const auto& [a, b] = (*m_indexedElements)[i].array();

            type::Quat<SReal>& q = beamQuat(i);
            q.normalize();
            Transformation R,Rt;
            q.toMatrix(R);
            Rt.transpose(R);
            const StiffnessMatrix& K0 = d_beamsData.getValue()[i]._k_loc;
            StiffnessMatrix K;
            for (int x1=0; x1<12; x1+=3)
                for (int y1=0; y1<12; y1+=3)
                {
                    type::Mat<3,3,Real> m;
                    K0.getsub(x1,y1, m);
                    m = R*m*Rt;
                    K.setsub(x1,y1, m);
                }
            int index[12];
            for (int x1=0; x1<6; x1++)
                index[x1] = a*6+x1;
            for (int x1=0; x1<6; x1++)
                index[6+x1] = b*6+x1;
            for (int x1=0; x1<12; ++x1)
                for (int y1=0; y1<12; ++y1)
                    dfdx(index[x1], index[y1]) += - K(x1,y1);

        }

    }
    else
    {
        unsigned int i {};
        for(auto it = m_indexedElements->begin() ; it != m_indexedElements->end() ; ++it, ++i)
        {
            const auto& [a, b] = it->array();

            type::Quat<SReal>& q = beamQuat(i);
            q.normalize();
            Transformation R,Rt;
            q.toMatrix(R);
            Rt.transpose(R);
            const StiffnessMatrix& K0 = d_beamsData.getValue()[i]._k_loc;
            StiffnessMatrix K;
            const bool exploitSymmetry = d_useSymmetricAssembly.getValue();

            if (exploitSymmetry) {
                for (int x1=0; x1<12; x1+=3) {
                    for (int y1=x1; y1<12; y1+=3)
                    {
                        type::Mat<3,3,Real> m;
                        K0.getsub(x1,y1, m);
                        m = R*m*Rt;

                        for (int i=0; i<3; i++)
                            for (int j=0; j<3; j++) {
                                K(i+x1,j+y1) += m(i,j);
                                K(j+y1,i+x1) += m(i,j);
                            }
                        if (x1 == y1)
                            for (int i=0; i<3; i++)
                                for (int j=0; j<3; j++)
                                    K(i+x1,j+y1) *= SReal(0.5);

                    }
                }
            } else  {
                for (int x1=0; x1<12; x1+=3) {
                    for (int y1=0; y1<12; y1+=3)
                    {
                        type::Mat<3,3,Real> m;
                        K0.getsub(x1,y1, m);
                        m = R*m*Rt;
                        K.setsub(x1,y1, m);
                    }
                }
            }

            int index[12];
            for (int x1=0; x1<6; x1++)
                index[x1] = a*6+x1;
            for (int x1=0; x1<6; x1++)
                index[6+x1] = b*6+x1;
            for (int x1=0; x1<12; ++x1)
                for (int y1=0; y1<12; ++y1)
                    dfdx(index[x1], index[y1]) += - K(x1,y1);

        }
    }
}

template <class DataTypes>
void BeamFEMForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
SReal BeamFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord&  x) const
{
    SOFA_UNUSED(x);
    SOFA_UNUSED(mparams);
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}


template<class DataTypes>
void BeamFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;
    if (!m_indexedElements)
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    std::vector< type::Vec3 > points[3];

    if (m_partialListSegment)
    {
        for (unsigned int j=0; j<d_listSegment.getValue().size(); j++)
            drawElement(d_listSegment.getValue()[j], points, x);
    }
    else
    {
        for (unsigned int i=0; i<m_indexedElements->size(); ++i)
            drawElement(i, points, x);
    }
    vparams->drawTool()->drawLines(points[0], 1, sofa::type::RGBAColor::red());
    vparams->drawTool()->drawLines(points[1], 1, sofa::type::RGBAColor::green());
    vparams->drawTool()->drawLines(points[2], 1, sofa::type::RGBAColor::blue());


}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( !onlyVisible ) return;

    const VecCoord& p = this->mstate->read(core::vec_id::read_access::position)->getValue();

    type::BoundingBox bbox;
    for (const auto& pt : p )
    {
        bbox.include(pt.getCenter());
    }

    this->f_bbox.setValue(bbox);

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::drawElement(int i, std::vector< type::Vec3 >* points, const VecCoord& x)
{
    const auto& [a, b] = (*m_indexedElements)[i].array();
    const type::Vec3d p = (x[a].getCenter() + x[b].getCenter()) * 0.5;
    type::Vec3d beamVec;
    beamVec[0]= d_beamsData.getValue()[i]._L * 0.5; beamVec[1] = 0.0; beamVec[2] = 0.0;

    const type::Quat<SReal>& q = beamQuat(i);

    // axis X
    points[0].push_back(p - q.rotate(beamVec) );
    points[0].push_back(p + q.rotate(beamVec) );

    // axis Y
    beamVec[0]=0.0; beamVec[1] = d_beamsData.getValue()[i]._r * 0.5;
    points[1].push_back(p );
    points[1].push_back(p + q.rotate(beamVec) );

    // axis Z
    beamVec[1]=0.0; beamVec[2] = d_beamsData.getValue()[i]._r * 0.5;
    points[2].push_back(p);
    points[2].push_back(p + q.rotate(beamVec) );
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::initBeams(std::size_t size)
{
    helper::WriteAccessor<Data<type::vector<BeamInfo> > > bd = d_beamsData;
    bd.resize(size);
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::setUpdateStiffnessMatrix(bool val)
{
    this->m_updateStiffnessMatrix = val;
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::setBeam(Index i, SReal E, SReal L, SReal nu, SReal r, SReal rInner)
{
    helper::WriteAccessor<Data<type::vector<BeamInfo> > > bd = d_beamsData;
    bd[i].init(E,L,nu,r,rInner);
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::BeamInfo::init(SReal E, SReal L, SReal nu, SReal r, SReal rInner)
{
    _E = E;
    _E0 = E;
    _nu = nu;
    _L = L;
    _r = r;
    _rInner = rInner;

    _G=_E/(2.0*(1.0+_nu));
    _Iz = M_PI*(r*r*r*r - rInner*rInner*rInner*rInner)/4.0;

    _Iy = _Iz ;
    _J = _Iz+_Iy;
    _A = M_PI*(r*r - rInner*rInner);


    _Asy = 0.0;
    _Asz = 0.0;
}

} // namespace sofa::component::solidmechanics::fem::elastic::_beamfemforcefield_
