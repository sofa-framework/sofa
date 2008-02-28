#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_INL

#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/component/forcefield/BeamFEMForceField.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/GridTopology.h>
#include <sofa/helper/PolarDecompose.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>
#include <set>
#include <sofa/helper/system/gl.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/tree/GNode.h>
using std::cerr;
using std::endl;
using std::set;


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::core;
using namespace sofa::core::componentmodel;
using namespace sofa::defaulttype;

template <class DataTypes>
void BeamFEMForceField<DataTypes>::init()
{
    this->core::componentmodel::behavior::ForceField<DataTypes>::init();
    sofa::simulation::tree::GNode* context = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
    _topology = context->get< sofa::component::topology::EdgeSetTopology<DataTypes> >();
    topology::MeshTopology* topo2 = context->get< sofa::component::topology::MeshTopology >();
    if (_topology==NULL && topo2==NULL)
    {
        std::cerr << "ERROR(BeamFEMForceField): object must have a EdgeSetTopology or MeshTopology.\n";
        return;
    }
    if (_topology!=NULL)
    {
        topology::EdgeSetTopologyContainer* container = _topology->getEdgeSetTopologyContainer();
        if(container->getEdgeArray().empty())
        {
            std::cerr << "ERROR(BeamFEMForceField): EdgeSetTopology is empty.\n";
            return;
        }
        _indexedElements = & (container->getEdgeArray());
    }
    else if (topo2!=NULL)
    {
        if(topo2->getNbEdges()==0)
        {
            std::cerr << "ERROR(BeamFEMForceField): MeshTopology is empty.\n";
            return;
        }
        //_indexedElements = (const VecElement*) & (topo2->getEdges());
        VecElement* e = new VecElement;
        e->resize(topo2->getNbEdges());
        for (unsigned int i=0; i<e->size(); ++i)
            (*e)[i] = helper::make_array<unsigned int>(topo2->getEdge(i)[0], topo2->getEdge(i)[1]);
        _indexedElements = e;
    }

    if (_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX();
        _initialPoints.setValue(p);
    }

    beamsData.resize(_indexedElements->size());

    typename VecElement::const_iterator it;
    unsigned int i=0;
    for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
    {
        beamsData[i]._E = _youngModulus.getValue();
        beamsData[i]._nu = _poissonRatio.getValue();
        defaulttype::Vec<3,Real> dp; dp = _initialPoints.getValue()[(*it)[0]].getCenter()-_initialPoints.getValue()[(*it)[1]].getCenter();
        beamsData[i]._L = dp.norm();
        beamsData[i]._r = _radius.getValue();
        beamsData[i]._G  = beamsData[i]._E/(2.0*(1.0+beamsData[i]._nu));
        beamsData[i]._Iz = R_PI*beamsData[i]._r*beamsData[i]._r*beamsData[i]._r*beamsData[i]._r/4;
        beamsData[i]._Iy = beamsData[i]._Iz ;
        beamsData[i]._J  = beamsData[i]._Iz+beamsData[i]._Iy;
        beamsData[i]._A  = R_PI*beamsData[i]._r*beamsData[i]._r;

        if (_timoshenko.getValue())
        {
            beamsData[i]._Asy = 10.0/9.0;
            beamsData[i]._Asz = 10.0/9.0;
        }
        else
        {
            beamsData[i]._Asy = 0.0;
            beamsData[i]._Asz = 0.0;
        }
    }
    _stiffnessMatrices.resize(_indexedElements->size() );
    _forces.resize( _initialPoints.getValue().size() );

    //case LARGE :
    {
        _beamQuat.resize( _indexedElements->size() );
        typename VecElement::const_iterator it;
        i=0;
        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];

            computeStiffness(i,a,b);
            initLarge(i,a,b);
        }
        //break;
    }

    std::cout << "BeamFEMForceField: init OK, "<<_indexedElements->size()<<" elements."<<std::endl;
}


template<class DataTypes>
void BeamFEMForceField<DataTypes>::addForce (VecDeriv& f, const VecCoord& p, const VecDeriv& /*v*/)
{
    f.resize(p.size());

    // First compute each node rotation
    unsigned int i;

    _nodeRotations.resize(p.size());
    //Mat3x3d R; R.identity();
    for(i=0; i<p.size(); ++i)
    {
        //R = R * MatrixFromEulerXYZ(p[i][3], p[i][4], p[i][5]);
        //_nodeRotations[i] = R;
        p[i].getOrientation().toMatrix(_nodeRotations[i]);
    }

    typename VecElement::const_iterator it;

    for(it=_indexedElements->begin(),i=0; it!=_indexedElements->end(); ++it,++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];

        accumulateForceLarge( f, p, i, a, b );
        initLarge(i,a,b);
    }

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::addDForce (VecDeriv& df, const VecDeriv& dx)
{
    df.resize(dx.size());
    //if(_assembling) applyStiffnessAssembled(v,x);
    //else
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;

        for(it = _indexedElements->begin() ; it != _indexedElements->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];

            applyStiffnessLarge( df, dx, i, a, b );
        }
    }
}

template<class DataTypes>
typename BeamFEMForceField<DataTypes>::Real BeamFEMForceField<DataTypes>::peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M )
{
    return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::computeStiffness(int i, Index , Index )
{
    double   phiy, phiz;
    double _L = beamsData[i]._L;
    double _A = beamsData[i]._A;
    double _nu = beamsData[i]._nu;
    double _E = beamsData[i]._E;
    double _Iy = beamsData[i]._Iy;
    double _Iz = beamsData[i]._Iz;
    double _Asy = beamsData[i]._Asy;
    double _Asz = beamsData[i]._Asz;
    double _G = beamsData[i]._G;
    double _J = beamsData[i]._J;
    double   L2 = _L * _L;
    double   L3 = L2 * _L;
    double   EIy = _E * _Iy;
    double   EIz = _E * _Iz;

    // Find shear-deformation parameters
    if (_Asy == 0)
        phiy = 0.0;
    else
        phiy = 24.0*(1.0+_nu)*_Iz/(_Asy*L2);

    if (_Asz == 0)
        phiz = 0.0;
    else
        phiz = 24.0*(1.0+_nu)*_Iy/(_Asz*L2);

    StiffnessMatrix& k_loc = _stiffnessMatrices[i];

    // Define stiffness matrix 'k' in local coordinates
    k_loc.clear();
    k_loc[6][6]   = k_loc[0][0]   = _E*_A/_L;
    k_loc[7][7]   = k_loc[1][1]   = 12.0*EIz/(L3*(1.0+phiy));
    k_loc[8][8]   = k_loc[2][2]   = 12.0*EIy/(L3*(1.0+phiz));
    k_loc[9][9]   = k_loc[3][3]   = _G*_J/_L;
    k_loc[10][10] = k_loc[4][4]   = (4.0+phiz)*EIy/(_L*(1.0+phiz));
    k_loc[11][11] = k_loc[5][5]   = (4.0+phiy)*EIz/(_L*(1.0+phiy));

    k_loc[4][2]   = -6.0*EIy/(L2*(1.0+phiz));
    k_loc[5][1]   =  6.0*EIz/(L2*(1.0+phiy));
    k_loc[6][0]   = -k_loc[0][0];
    k_loc[7][1]   = -k_loc[1][1];
    k_loc[7][5]   = -k_loc[5][1];
    k_loc[8][2]   = -k_loc[2][2];
    k_loc[8][4]   = -k_loc[4][2];
    k_loc[9][3]   = -k_loc[3][3];
    k_loc[10][2]  = k_loc[4][2];
    k_loc[10][4]  = (2.0-phiz)*EIy/(_L*(1.0+phiz));
    k_loc[10][8]  = -k_loc[4][2];
    k_loc[11][1]  = k_loc[5][1];
    k_loc[11][5]  = (2.0-phiy)*EIz/(_L*(1.0+phiy));
    k_loc[11][7]  = -k_loc[5][1];

    for (int i=0; i<=10; i++)
        for (int j=i+1; j<12; j++)
            k_loc[i][j] = k_loc[j][i];
}

////////////// large displacements method
template<class DataTypes>
void BeamFEMForceField<DataTypes>::initLarge(int i, Index a, Index b)
{
    behavior::MechanicalState<DataTypes>* mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const VecCoord& x = *mstate->getX();

    Quat quatA, quatB, dQ;
    Vec3d dW;

    quatA = x[a].getOrientation();
    quatB = x[b].getOrientation();


    dQ = quatA.inverse() * quatB;

    dW = dQ.toEulerVector();

    double Theta = dW.norm();

    if(Theta>0.0000001)
    {
        dW.normalize();

        _beamQuat[i] = quatA*dQ.axisToQuat(dW, Theta/2);
    }
    else
        _beamQuat[i]= quatA;

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::accumulateForceLarge( VecDeriv& f, const VecCoord & x, int i, Index a, Index b )
{
    behavior::MechanicalState<DataTypes>* mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const VecCoord& x0 = *mstate->getX0();

    Vec<3,Real> u, P1P2, P1P2_0;
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
    Quat dQ0, dQ;

    // dQ = QA.i * QB ou dQ = QB * QA.i() ??
    dQ0 = x0[a].getOrientation().inverse() * x0[b].getOrientation();
    dQ =  x[a].getOrientation().inverse() * x[b].getOrientation();
    u = dQ.toEulerVector() - dQ0.toEulerVector();

    depl[3] = 0.0; 	depl[4] = 0.0; 	depl[5] = 0.0;
    depl[9] = u[0]; depl[10]= u[1]; depl[11]= u[2];

    // this computation can be optimised: (we know that half of "depl" is null)
    Displacement force = _stiffnessMatrices[i] * depl;


    // Apply lambda transpose (we use the rotation value of point a for the beam)

    Vec3d fa1 = x[a].getOrientation().rotate(Vec3d(force[0],force[1],force[2]));
    Vec3d fa2 = x[a].getOrientation().rotate(Vec3d(force[3],force[4],force[5]));

    Vec3d fb1 = x[a].getOrientation().rotate(Vec3d(force[6],force[7],force[8]));
    Vec3d fb2 = x[a].getOrientation().rotate(Vec3d(force[9],force[10],force[11]));


    f[a] += Deriv(-fa1,-fa2);
    f[b] += Deriv(-fb1,-fb2);

}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::applyStiffnessLarge( VecDeriv& df, const VecDeriv& dx, int i, Index a, Index b )
{
    behavior::MechanicalState<DataTypes>* mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const VecCoord& x = *mstate->getX();

    Displacement local_depl;
    Vec3d u;
    const Quat& q = x[a].getOrientation();

    u = q.inverseRotate(dx[a].getVCenter());
    local_depl[0] = u[0];
    local_depl[1] = u[1];
    local_depl[2] = u[2];

    u = q.inverseRotate(dx[a].getVOrientation());
    local_depl[3] = u[0];
    local_depl[4] = u[1];
    local_depl[5] = u[2];


    u = q.inverseRotate(dx[b].getVCenter());
    local_depl[6] = u[0];
    local_depl[7] = u[1];
    local_depl[8] = u[2];

    u = q.inverseRotate(dx[b].getVOrientation());
    local_depl[9] = u[0];
    local_depl[10] = u[1];
    local_depl[11] = u[2];

    Displacement local_force = _stiffnessMatrices[i] * local_depl;

    Vec3d fa1 = q.rotate(Vec3d(local_force[0],local_force[1] ,local_force[2] ));
    Vec3d fa2 = q.rotate(Vec3d(local_force[3],local_force[4] ,local_force[5] ));
    Vec3d fb1 = q.rotate(Vec3d(local_force[6],local_force[7] ,local_force[8] ));
    Vec3d fb2 = q.rotate(Vec3d(local_force[9],local_force[10],local_force[11]));

    df[a] += Deriv(-fa1,-fa2);
    df[b] += Deriv(-fb1,-fb2);
}

template<class DataTypes>
void BeamFEMForceField<DataTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);
    typename VecElement::const_iterator it;
    int i;
    for(it = _indexedElements->begin(), i = 0 ; it != _indexedElements->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        defaulttype::Vec3d p; p = (x[a].getCenter()+x[b].getCenter())*0.5;
        Vec3d beamVec;
        beamVec[0]=beamsData[i]._L*0.5; beamVec[1] = 0.0; beamVec[2] = 0.0;

        const Quat& q = _beamQuat[i];
        // axis X
        glColor3f(1,0,0);
        helper::gl::glVertexT(p - q.rotate(beamVec) );
        helper::gl::glVertexT(p + q.rotate(beamVec) );
        // axis Y
        beamVec[0]=0.0; beamVec[1] = beamsData[i]._L*0.5;
        glColor3f(0,1,0);
        helper::gl::glVertexT(p); // - R.col(1)*len);
        helper::gl::glVertexT(p + q.rotate(beamVec) );
        // axis Z
        beamVec[1]=0.0; beamVec[2] = beamsData[i]._L*0.5;
        glColor3f(0,0,1);
        helper::gl::glVertexT(p); // - R.col(2)*len);
        helper::gl::glVertexT(p + q.rotate(beamVec) );
    }
    glEnd();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
