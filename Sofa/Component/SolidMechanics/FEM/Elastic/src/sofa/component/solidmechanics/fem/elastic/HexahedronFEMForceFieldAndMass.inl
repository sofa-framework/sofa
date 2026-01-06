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

#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceFieldAndMass.h>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.inl>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
HexahedronFEMForceFieldAndMass<DataTypes>::HexahedronFEMForceFieldAndMass()
    : MassT()
    , HexahedronFEMForceFieldT()
    , d_elementMasses(initData(&d_elementMasses,"massMatrices", "Mass matrices per element (M_i)"))
    , d_density(initData(&d_density,(Real)1.0,"density","density == volumetric mass in english (kg.m-3)"))
    , d_lumpedMass(initData(&d_lumpedMass,(bool)false,"lumpedMass","Does it use lumped masses?"))
{
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::init( )
{
    if(this->_alreadyInit)return;
    HexahedronFEMForceFieldT::init();
    MassT::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    _particleMasses.resize( this->d_initialPoints.getValue().size() );

    int i=0;
    for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
    {
        type::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = this->d_initialPoints.getValue()[(*it)[w]];

        // volume of a element
        Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

        if( this->_sparseGrid ) // if sparseGrid -> the filling ratio is taken into account
            volume *= this->_sparseGrid->getMassCoef(i);

        // mass of a particle...
        Real mass = Real (( volume * d_density.getValue() ) / 8.0 );

        // ... is added to each particle of the element
        for(int w=0; w<8; ++w)
            _particleMasses[ (*it)[w] ] += mass;
    }



    if( d_lumpedMass.getValue() )
    {
        _lumpedMasses.resize( this->d_initialPoints.getValue().size() );
        i=0;
        for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {

            const ElementMass& mass=d_elementMasses.getValue()[i];

            for(int w=0; w<8; ++w)
            {
                for(int j=0; j<8*3; ++j)
                {
                    _lumpedMasses[ (*it)[w] ][0] += mass(w*3  ,j);
                    _lumpedMasses[ (*it)[w] ][1] += mass(w*3+1,j);
                    _lumpedMasses[ (*it)[w] ][2] += mass(w*3+2,j);
                }
            }
        }
    }
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::reinit( )
{
    HexahedronFEMForceFieldT::reinit();
    computeElementMasses();
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMasses(  )
{
    int i=0;
    typename VecElement::const_iterator it;
    for(it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
    {
        type::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = this->d_initialPoints.getValue()[(*it)[w]];

        if( d_elementMasses.getValue().size() <= (unsigned)i )
        {
            d_elementMasses.beginEdit()->resize( d_elementMasses.getValue().size()+1 );
            computeElementMass( (*d_elementMasses.beginEdit())[i], this->_rotatedInitialElements[i],i,	this->_sparseGrid?this->_sparseGrid->getMassCoef(i):1.0 );
        }
    }
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMass( ElementMass &Mass, const type::fixed_array<Coord,8> &nodes, const Index /*elementIndice*/, SReal stiffnessFactor)
{
    Real vol = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

    Coord l = nodes[6] - nodes[0];

    Mass.clear();

    for(int i=0; i<8; ++i)
    {
        Real mass = vol * integrateMass(this->_coef(i,0), this->_coef(i,1),this->_coef(i,2), 2.0f/l[0], 2.0f/l[1], 2.0f/l[2]);

        Mass(i*3,i*3) += mass;
        Mass(i*3+1,i*3+1) += mass;
        Mass(i*3+2,i*3+2) += mass;



        for(int j=i+1; j<8; ++j)
        {
            Real mass = vol * integrateMass(this->_coef(i,0), this->_coef(i,1),this->_coef(i,2), 2.0f/l[0], 2.0f/l[1], 2.0f/l[2]);

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

    Mass *= (Real)stiffnessFactor;
}


template<class DataTypes>
typename HexahedronFEMForceFieldAndMass<DataTypes>::Real HexahedronFEMForceFieldAndMass<DataTypes>::integrateMass(  int signx, int signy, int signz,Real /*l0*/,Real /*l1*/,Real /*l2*/  )
{
    Real t1 = (Real)(signx*signx);
    Real t2 = (Real)(signy*signy);
    Real t3 = (Real)(signz*signz);
    Real t9 = (Real)(t1*t2);

    return (Real)(t1*t3/72.0+t2*t3/72.0+t9*t3/216.0+t3/24.0+1.0/8.0+t9/72.0+t1/24.0+t2/24.0)*d_density.getValue();
}

template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor)
{
    helper::WriteAccessor< DataVecDeriv > _f = f;
    helper::ReadAccessor< DataVecDeriv > _dx = dx;
    if( ! d_lumpedMass.getValue() )
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;

        for(it=this->getIndexedElements()->begin(); it!=this->getIndexedElements()->end(); ++it,++i)
        {

            type::Vec<24, Real> actualDx, actualF;

            for(int k=0 ; k<8 ; ++k )
            {
                const int index = k*3;
                for(int j=0 ; j<3 ; ++j )
                    actualDx[index+j] = _dx[(*it)[k]][j];

            }

            actualF = d_elementMasses.getValue()[i] * actualDx;


            for(int w=0; w<8; ++w)
                _f[(*it)[w]] += Deriv( actualF[w*3],  actualF[w*3+1],   actualF[w*3+2]  ) * factor;

        }
    }
    else // lumped matrices
    {
        for(unsigned i=0; i<_lumpedMasses.size(); ++i)
            for(int j=0; j<3; ++j)
                _f[i][j] += (Real)(_lumpedMasses[i][j] * _dx[i][j] *factor);
    }
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addMToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal mFact, unsigned int &offset)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2, e;

    typename VecElement::const_iterator it;

    int node1, node2;

    for(it = this->getIndexedElements()->begin(), e=0 ; it != this->getIndexedElements()->end() ; ++it,++e)
    {
        const ElementMass &Me = d_elementMasses.getValue()[e];

        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            node1 = (*it)[n1];

            // find index of node 2
            for (n2=0; n2<8; n2++)
            {
                node2 = (*it)[n2];

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

template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    HexahedronFEMForceFieldT::buildStiffnessMatrix(matrix);
}

template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    int e = 0;
    for(auto it = this->getIndexedElements()->begin(); it != this->getIndexedElements()->end() ; ++it, ++e)
    {
        const ElementMass &Me = d_elementMasses.getValue()[e];

        // find index of node 1
        for (int n1 = 0; n1<8; n1++)
        {
            const int node1 = (*it)[n1];

            // find index of node 2
            for (int n2 = 0; n2<8; n2++)
            {
                const int node2 = (*it)[n2];

                const Mat33 tmp = Mat33(Coord(Me(3*n1+0,3*n2+0),Me(3*n1+0,3*n2+1),Me(3*n1+0,3*n2+2)),
                        Coord(Me(3*n1+1,3*n2+0),Me(3*n1+1,3*n2+1),Me(3*n1+1,3*n2+2)),
                        Coord(Me(3*n1+2,3*n2+0),Me(3*n1+2,3*n2+1),Me(3*n1+2,3*n2+2)));
                matrices->add(3 * node1, 3 * node2, tmp);
            }
        }
    }
}

template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*a*/, const DataVecDeriv& /*f*/)
{
    msg_warning()<<"HexahedronFEMForceFieldAndMass<DataTypes>::accFromF not yet implemented"<<msgendl;
    // need to built the big global mass matrix and to inverse it...
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();
        SReal dt = sofa::core::mechanicalparams::dt(mparams);
        for (unsigned int i=0; i<_particleMasses.size(); i++)
        {
            v[i] +=this->getContext()->getGravity()*dt;
        }
        d_v.beginEdit();
    }
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    HexahedronFEMForceFieldT::addForce(mparams, f,x,v);

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if (this->m_separateGravity.getValue())
        return;

    helper::WriteAccessor< DataVecDeriv > _f = f;
    for (unsigned int i=0; i<_particleMasses.size(); i++)
    {
        _f[i] += this->getContext()->getGravity()*_particleMasses[i];
    }
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    HexahedronFEMForceFieldT::addDForce(mparams, df, dx);
}


template<class DataTypes>
SReal HexahedronFEMForceFieldAndMass<DataTypes>::getElementMass(sofa::Index /*index*/) const
{
    msg_warning()<<"HexahedronFEMForceFieldAndMass<DataTypes>::getElementMass not yet implemented"<<msgendl; return 0.0;
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    HexahedronFEMForceFieldT::draw(vparams);

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
    // since drawTool requires a std::vector<Vec3> we have to convert x in an ugly way
    std::vector<type::Vec3> pos;
    pos.resize(x.size());
    auto posIT = pos.begin();
    typename VecCoord::const_iterator xIT = x.begin();
    for(; posIT != pos.end() ; ++posIT, ++xIT)
    {
        *posIT = *xIT;
    }
    vparams->drawTool()->drawPoints(pos,2.0f, sofa::type::RGBAColor::white());
}

} // namespace sofa::component::solidmechanics::fem::elastic
