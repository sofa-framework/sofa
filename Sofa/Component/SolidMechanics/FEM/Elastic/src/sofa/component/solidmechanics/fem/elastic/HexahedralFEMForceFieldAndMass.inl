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
#include <sofa/component/solidmechanics/fem/elastic/HexahedralFEMForceFieldAndMass.h>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h> 

#include <sofa/core/topology/TopologyData.inl>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
HexahedralFEMForceFieldAndMass<DataTypes>::HexahedralFEMForceFieldAndMass()
    : MassT()
    , HexahedralFEMForceFieldT()
    , d_density(initData(&d_density, (Real)1.0, "density", "density == volumetric mass in english (kg.m-3)"))
    , d_useLumpedMass(initData(&d_useLumpedMass, (bool)false, "lumpedMass", "Does it use lumped masses?"))
    , d_elementMasses(initData(&d_elementMasses, "massMatrices", "Mass matrices per element (M_i)", false))
    , d_elementTotalMass(initData(&d_elementTotalMass, "totalMass", "Total mass per element", false))
    , d_particleMasses(initData(&d_particleMasses, "particleMasses", "Mass per particle", false))
    , d_lumpedMasses(initData(&d_lumpedMasses, "lumpedMasses", "Lumped masses", false))
{
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::init( )
{
    BaseLinearElasticityFEMForceField<DataTypes>::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    this->reinit();
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::reinit( )
{
    HexahedralFEMForceFieldT::reinit();
    MassT::reinit();

    computeElementMasses();
    computeParticleMasses();
    computeLumpedMasses();
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeParticleMasses(  )
{
    unsigned int numPoints = this->l_topology->getNbPoints();
    const VecElement& hexahedra = this->l_topology->getHexahedra();

    type::vector<Real>&	particleMasses = *this->d_particleMasses.beginEdit();

    particleMasses.clear();
    particleMasses.resize( numPoints );

    for(unsigned int i=0; i<hexahedra.size(); ++i)
    {
        // mass of a particle...
        Real mass = d_elementTotalMass.getValue()[i] * (Real) 0.125;

        // ... is added to each particle of the element
        for(int w=0; w<8; ++w)
            particleMasses[ hexahedra[i][w] ] += mass;
    }

    this->d_particleMasses.endEdit();
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeLumpedMasses(  )
{
    unsigned int numPoints = this->l_topology->getNbPoints();
    const VecElement& hexahedra = this->l_topology->getHexahedra();

    if( d_useLumpedMass.getValue() )
    {
        type::vector<Coord>&	lumpedMasses = *this->d_lumpedMasses.beginEdit();

        lumpedMasses.clear();
        lumpedMasses.resize( numPoints, Coord(0.0, 0.0, 0.0) );

        for(unsigned int i=0; i<hexahedra.size(); ++i)
        {
            const ElementMass& mass = this->d_elementMasses.getValue()[i];

            for(int w=0; w<8; ++w)
            {
                for(int j=0; j<8*3; ++j)
                {
                    lumpedMasses[ hexahedra[i][w] ][0] += mass(w*3  ,j);
                    lumpedMasses[ hexahedra[i][w] ][1] += mass(w*3+1,j);
                    lumpedMasses[ hexahedra[i][w] ][2] += mass(w*3+2,j);
                }
            }
        }

        this->d_lumpedMasses.endEdit();
    }
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeElementMasses(  )
{
    const VecCoord& initialPoints = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();

    const VecElement& hexahedra = this->l_topology->getHexahedra();

    type::vector<ElementMass>& elementMasses = *this->d_elementMasses.beginEdit();
    type::vector<Real>& elementTotalMass = *this->d_elementTotalMass.beginEdit();

    elementMasses.resize( hexahedra.size() );
    elementTotalMass.resize( hexahedra.size() );

    for(unsigned int i=0; i<hexahedra.size(); ++i)
    {
        type::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = initialPoints[hexahedra[i][w]];

        computeElementMass( elementMasses[i], elementTotalMass[i],
                this->d_hexahedronInfo.getValue()[i].rotatedInitialElements);
    }

    this->d_elementTotalMass.endEdit();
    this->d_elementMasses.endEdit();
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeElementMass( ElementMass &Mass, Real& totalMass,
        const type::fixed_array<Coord,8> &nodes)
{
    // volume of a element
    Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

    // total element mass
    totalMass = volume * d_density.getValue();

    Coord l = nodes[6] - nodes[0];

    Mass.clear();

    for(int i=0; i<8; ++i)
    {
        Real mass = totalMass * integrateVolume(this->_coef(i,0),
                this->_coef(i,1),
                this->_coef(i,2),
                2.0f/l[0],
                2.0f/l[1],
                2.0f/l[2]);

        Mass(i*3,i*3) += mass;
        Mass(i*3+1,i*3+1) += mass;
        Mass(i*3+2,i*3+2) += mass;

        for(int j=i+1; j<8; ++j)
        {
            Real mass = totalMass * integrateVolume(this->_coef(i,0),
                    this->_coef(i,1),
                    this->_coef(i,2),
                    2.0f/l[0],
                    2.0f/l[1],
                    2.0f/l[2]);

            Mass(i*3,j*3) += mass;
            Mass(i*3+1,j*3+1) += mass;
            Mass(i*3+2,j*3+2) += mass;
        }
    }

    for(int i=0; i<24; ++i)
        for(int j=i+1; j<24; ++j)
        {
            Mass(j,i) = Mass(i,j);
        }
}


template<class DataTypes>
typename HexahedralFEMForceFieldAndMass<DataTypes>::Real HexahedralFEMForceFieldAndMass<DataTypes>::integrateVolume(  int signx, int signy, int signz,Real /*l0*/,Real /*l1*/,Real /*l2*/  )
{
    Real t1 = (Real)(signx*signx);
    Real t2 = (Real)(signy*signy);
    Real t3 = (Real)(signz*signz);
    Real t9 = (Real)(t1*t2);

    return (Real)(t1*t3/72.0+t2*t3/72.0+t9*t3/216.0+t3/24.0+1.0/8.0+t9/72.0+t1/24.0+t2/24.0);
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor)
{
    helper::WriteAccessor< DataVecDeriv > _f = f;
    const VecDeriv& _dx = dx.getValue();

    if( ! d_useLumpedMass.getValue() )
    {
        const VecElement& hexahedra = this->l_topology->getHexahedra();
        for(unsigned int i=0; i<hexahedra.size(); ++i)
        {
            type::Vec<24, Real> actualDx, actualF;

            for(int k=0 ; k<8 ; ++k )
            {
                const int index = k*3;
                for(int j=0 ; j<3 ; ++j )
                    actualDx[index+j] = _dx[hexahedra[i][k]][j];
            }

            actualF = d_elementMasses.getValue()[i] * actualDx;


            for(unsigned int w=0; w<8; ++w)
                _f[hexahedra[i][w]] += Deriv( actualF[w*3],  actualF[w*3+1],   actualF[w*3+2]  ) * factor;
        }
    }
    else // lumped matrices
    {
        for(unsigned int i=0; i < d_lumpedMasses.getValue().size(); ++i)
            for(unsigned int j=0; j<3; ++j)
                _f[i][j] += (Real)(d_lumpedMasses.getValue()[i][j] * _dx[i][j] * factor);
    }
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addMToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal mFact, unsigned int &offset)
{
    // Build Matrix Block for this ForceField
    int i, j, n1, n2;
    int node1, node2;

    const VecElement& hexahedra = this->l_topology->getHexahedra();

    for(unsigned int e=0; e<hexahedra.size(); ++e)
    {
        const ElementMass &Me = d_elementMasses.getValue()[e];

        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            node1 = hexahedra[e][n1];
            n2 = n1; /////////// WARNING Changed to compute only diag elements

            // find index of node 2
            //for (n2=0; n2<8; n2++) /////////// WARNING Changed to compute only diag elements
            {
                node2 = hexahedra[e][n2];

                Mat33 tmp = Mat33(Coord(Me(3*n1+0,3*n2+0),Me(3*n1+0,3*n2+1),Me(3*n1+0,3*n2+2)),
                        Coord(Me(3*n1+1,3*n2+0),Me(3*n1+1,3*n2+1),Me(3*n1+1,3*n2+2)),
                        Coord(Me(3*n1+2,3*n2+0),Me(3*n1+2,3*n2+1),Me(3*n1+2,3*n2+2)));
                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        mat->add(offset+3*node1+i, offset+3*node2+j, tmp(i,j)*mFact);
            }
        }
    }
}

///// WARNING this method only add diagonal elements in the given matrix !
template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2, e;

    //typename VecElement::const_iterator it;
    typename type::vector<HexahedronInformation>::const_iterator it;

    Index node1, node2;
    const VecElement& hexahedra = this->l_topology->getHexahedra();

    for(it = this->d_hexahedronInfo.getValue().begin(), e=0 ; it != this->d_hexahedronInfo.getValue().end() ; ++it,++e)
    {
        const Element hexa = hexahedra[e];
        const ElementStiffness &Ke = it->stiffness;

        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            n2 = n1; /////////// WARNING Changed to compute only diag elements
            node1 = hexa[n1];

            // find index of node 2
            //for (n2=0; n2<8; n2++) /////////// WARNING Changed to compute only diag elements
            {
                node2 = hexa[n2];

                Mat33 tmp = it->rotation.multTranspose( Mat33(Coord(Ke(3*n1+0,3*n2+0),Ke(3*n1+0,3*n2+1),Ke(3*n1+0,3*n2+2)),
                        Coord(Ke(3*n1+1,3*n2+0),Ke(3*n1+1,3*n2+1),Ke(3*n1+1,3*n2+2)),
                        Coord(Ke(3*n1+2,3*n2+0),Ke(3*n1+2,3*n2+1),Ke(3*n1+2,3*n2+2))) ) * it->rotation;
                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        matrix->add(offset+3*node1+i, offset+3*node2+j, - tmp(i,j)*kFact);
            }
        }
    }
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addMBKToMatrix (const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    int i, j, n1, n2;
    Index node1, node2;

    const VecElement& hexahedra = this->l_topology->getHexahedra();

    if (this->d_hexahedronInfo.getValue().size() != hexahedra.size())
    {
        msg_error() << "HexahedronInformation vector and Topology's Hexahedron vector should have the same size.";
        return;
    }

    typename type::vector<HexahedronInformation>::const_iterator it;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);

    unsigned int e = 0;
    for ( it = this->d_hexahedronInfo.getValue().begin() ; it != this->d_hexahedronInfo.getValue().end() ; ++it, ++e )
    {
        const ElementMass &Me = d_elementMasses.getValue() [e];
        const Element hexa = hexahedra[e];
        const ElementStiffness &Ke = it->stiffness;

        // find index of node 1
        Real mFactor = (Real)sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(mparams, this->rayleighMass.getValue());
        Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
        for ( n1 = 0; n1 < 8; n1++ )
        {
            n2 = n1; /////////// WARNING Changed to compute only diag elements
            node1 = hexa[n1];

            // find index of node 2
            //for (n2=0; n2<8; n2++) /////////// WARNING Changed to compute only diag elements
            {
                node2 = hexa[n2];

                // add M to matrix
                Mat33 tmp = Mat33 ( Coord ( Me(3*n1+0,3*n2+0), Me(3*n1+0,3*n2+1), Me(3*n1+0,3*n2+2) ),
                        Coord ( Me(3*n1+1,3*n2+0), Me(3*n1+1,3*n2+1), Me(3*n1+1,3*n2+2) ),
                        Coord ( Me(3*n1+2,3*n2+0), Me(3*n1+2,3*n2+1), Me(3*n1+2,3*n2+2) ) );
                for ( i = 0; i < 3; i++ )
                    for ( j = 0; j < 3; j++ )
                        r.matrix->add ( r.offset + 3*node1 + i, r.offset + 3*node2 + j, tmp(i,j)*mFactor);

                // add K to matrix
                tmp = it->rotation.multTranspose ( Mat33 ( Coord ( Ke(3*n1+0,3*n2+0), Ke(3*n1+0,3*n2+1), Ke(3*n1+0,3*n2+2) ),
                        Coord ( Ke(3*n1+1,3*n2+0), Ke(3*n1+1,3*n2+1), Ke(3*n1+1,3*n2+2) ),
                        Coord ( Ke(3*n1+2,3*n2+0), Ke(3*n1+2,3*n2+1), Ke(3*n1+2,3*n2+2) ) ) ) * it->rotation;
                for ( i = 0; i < 3; i++ )
                    for ( j = 0; j < 3; j++ )
                        r.matrix->add ( r.offset + 3*node1 + i, r.offset + 3*node2 + j, - tmp(i,j)*kFactor);
            }
        }
    }
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::accFromF(const core::MechanicalParams*, DataVecDeriv& /*a*/, const DataVecDeriv& /*f*/)
{
    msg_error() << "HexahedralFEMForceFieldAndMass<DataTypes>::accFromF not yet implemented";
    // need to built the big global mass matrix and to inverse it...
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        SReal _dt = sofa::core::mechanicalparams::dt(mparams);

        for (unsigned int i=0; i < d_particleMasses.getValue().size(); i++)
        {
            v[i] +=this->getContext()->getGravity()*_dt;
        }
        d_v.beginEdit();
    }
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    HexahedralFEMForceFieldT::addForce(mparams, f,x,v);

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if (this->m_separateGravity.getValue())
        return;

    helper::WriteAccessor< DataVecDeriv > _f = f;
    for (unsigned int i=0; i < d_particleMasses.getValue().size(); i++)
    {
        _f[i] += this->getContext()->getGravity() * d_particleMasses.getValue()[i];
    }
}



template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    HexahedralFEMForceFieldT::addDForce(mparams, df, dx);
}


template<class DataTypes>
SReal HexahedralFEMForceFieldAndMass<DataTypes>::getElementMass(sofa::Index /*index*/) const
{
    msg_error() << "HexahedralFEMForceFieldAndMass<DataTypes>::getElementMass not yet implemented";
    return 0.0;
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    HexahedralFEMForceFieldT::draw(vparams);

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    std::vector<type::Vec3> pos;
    pos.reserve(x.size());

    std::transform(x.begin(), x.end(), std::back_inserter(pos),
        [](const auto& e){ return DataTypes::getCPos(e); });

    vparams->drawTool()->drawPoints(pos,2.0f, sofa::type::RGBAColor::white());
}

} // namespace sofa::component::solidmechanics::fem::elastic
