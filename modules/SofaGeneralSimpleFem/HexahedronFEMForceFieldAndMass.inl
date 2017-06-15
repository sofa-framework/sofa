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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONANDMASSFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONANDMASSFEMFORCEFIELD_INL


#include <SofaGeneralSimpleFem/HexahedronFEMForceFieldAndMass.h>
#include <SofaSimpleFem/HexahedronFEMForceField.inl>
#include <SofaBaseTopology/SparseGridTopology.h>


namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
HexahedronFEMForceFieldAndMass<DataTypes>::HexahedronFEMForceFieldAndMass()
    : MassT()
    , HexahedronFEMForceFieldT()
    ,_elementMasses(initData(&_elementMasses,"massMatrices", "Mass matrices per element (M_i)"))
    , _density(initData(&_density,(Real)1.0,"density","density == volumetric mass in english (kg.m-3)"))
    , _lumpedMass(initData(&_lumpedMass,(bool)false,"lumpedMass","Does it use lumped masses?"))
{
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::init( )
{
    if(this->_alreadyInit)return;

    // 		  serr<<"HexahedronFEMForceFieldAndMass<DataTypes>::init( ) "<<this->getName()<<sendl;
    HexahedronFEMForceFieldT::init();
    MassT::init();

    //         computeElementMasses();

    // 		_particleMasses.clear();
    _particleMasses.resize( this->_initialPoints.getValue().size() );

    int i=0;
    for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
    {
        defaulttype::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
            nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];
#else
            nodes[w] = this->_initialPoints.getValue()[(*it)[w]];
#endif

        // volume of a element
        Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

        if( this->_sparseGrid ) // if sparseGrid -> the filling ratio is taken into account
            volume *= this->_sparseGrid->getMassCoef(i);
        // 				volume *= (Real) (this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5:1.0);

        // mass of a particle...
        Real mass = Real (( volume * _density.getValue() ) / 8.0 );

        // ... is added to each particle of the element
        for(int w=0; w<8; ++w)
            _particleMasses[ (*it)[w] ] += mass;
    }



    if( _lumpedMass.getValue() )
    {
        _lumpedMasses.resize( this->_initialPoints.getValue().size() );
        i=0;
        for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {

            const ElementMass& mass=_elementMasses.getValue()[i];

            for(int w=0; w<8; ++w)
            {
                for(int j=0; j<8*3; ++j)
                {
                    _lumpedMasses[ (*it)[w] ][0] += mass[w*3  ][j];
                    _lumpedMasses[ (*it)[w] ][1] += mass[w*3+1][j];
                    _lumpedMasses[ (*it)[w] ][2] += mass[w*3+2][j];
                }
            }
        }
    }



    // 		Real totalmass = 0.0;
    // 		for( unsigned i=0;i<_particleMasses.size();++i)
    // 		{
    // 			totalmass+=_particleMasses[i];
    // 		}
    // 		serr<<"TOTAL MASS = "<<totalmass<<sendl;
}



template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::reinit( )
{
    // 		  serr<<"HexahedronFEMForceFieldAndMass<DataTypes>::reinit( )"<<sendl;
    HexahedronFEMForceFieldT::reinit();
    //         Mass::reinit();

    computeElementMasses();
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMasses(  )
{
    // 		  _elementMasses.resize( this->_elementStiffnesses.getValue().size() );

    int i=0;
    typename VecElement::const_iterator it;
    for(it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
    {
        defaulttype::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
            nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];
#else
            nodes[w] = this->_initialPoints.getValue()[(*it)[w]];
#endif

        if( _elementMasses.getValue().size() <= (unsigned)i )
        {
            _elementMasses.beginEdit()->resize( _elementMasses.getValue().size()+1 );
            computeElementMass( (*_elementMasses.beginEdit())[i], this->_rotatedInitialElements[i],i,	this->_sparseGrid?this->_sparseGrid->getMassCoef(i):1.0 );
        }


    }
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::computeElementMass( ElementMass &Mass, const helper::fixed_array<Coord,8> &nodes, const int /*elementIndice*/, SReal stiffnessFactor)
{
    Real vol = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

    Coord l = nodes[6] - nodes[0];

    Mass.clear();

    for(int i=0; i<8; ++i)
    {
        Real mass = vol * integrateMass(this->_coef[i][0], this->_coef[i][1],this->_coef[i][2], 2.0f/l[0], 2.0f/l[1], 2.0f/l[2]);

        Mass[i*3][i*3] += mass;
        Mass[i*3+1][i*3+1] += mass;
        Mass[i*3+2][i*3+2] += mass;



        for(int j=i+1; j<8; ++j)
        {
            Real mass = vol * integrateMass(this->_coef[i][0], this->_coef[i][1],this->_coef[i][2], 2.0f/l[0], 2.0f/l[1], 2.0f/l[2]);

            Mass[i*3][j*3] += mass;
            Mass[i*3+1][j*3+1] += mass;
            Mass[i*3+2][j*3+2] += mass;
        }
    }

    for(int i=0; i<24; ++i)
        for(int j=i+1; j<24; ++j)
        {
            Mass[j][i] = Mass[i][j];
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

    return (Real)(t1*t3/72.0+t2*t3/72.0+t9*t3/216.0+t3/24.0+1.0/8.0+t9/72.0+t1/24.0+t2/24.0)*_density.getValue();


    // 		  Real t1 = l0*l0;
    // 		  Real t2 = t1*signx;
    // 		  Real t3 = signz*signx;
    // 		  Real t7 = t1*signy;
    // 		  return t2*t3*signz/72.0+t7*signz*signy*signz/72.0+t2*signy*t3*signy*
    // 				  signz/216.0+t1*signz*signz/24.0+t2*signy*signx*signy/72.0+t1/8.0+t2*signx/
    // 				  24.0+t7*signy/24.0 *_density.getValue() /(l0*l1*l2);

}


template<class DataTypes>
std::string HexahedronFEMForceFieldAndMass<DataTypes>::getTemplateName() const
{
    return HexahedronFEMForceFieldT::getTemplateName();
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor)
{
    helper::WriteAccessor< DataVecDeriv > _f = f;
    helper::ReadAccessor< DataVecDeriv > _dx = dx;
    if( ! _lumpedMass.getValue() )
    {
        unsigned int i=0;
        typename VecElement::const_iterator it;

        for(it=this->getIndexedElements()->begin(); it!=this->getIndexedElements()->end(); ++it,++i)
        {

            defaulttype::Vec<24, Real> actualDx, actualF;

            for(int k=0 ; k<8 ; ++k )
            {
                int indice = k*3;
                for(int j=0 ; j<3 ; ++j )
#ifndef SOFA_NEW_HEXA
                    actualDx[indice+j] = _dx[(*it)[this->_indices[k]]][j];
#else
                    actualDx[indice+j] = _dx[(*it)[k]][j];
#endif

            }

            actualF = _elementMasses.getValue()[i] * actualDx;


            for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
                _f[(*it)[this->_indices[w]]] += Deriv( actualF[w*3],  actualF[w*3+1],   actualF[w*3+2]  ) * factor;
#else
                _f[(*it)[w]] += Deriv( actualF[w*3],  actualF[w*3+1],   actualF[w*3+2]  ) * factor;
#endif

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
void HexahedronFEMForceFieldAndMass<DataTypes>::addMToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2, e;

    typename VecElement::const_iterator it;

    int node1, node2;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);

    for(it = this->getIndexedElements()->begin(), e=0 ; it != this->getIndexedElements()->end() ; ++it,++e)
    {
        const ElementMass &Me = _elementMasses.getValue()[e];

        Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
#ifndef SOFA_NEW_HEXA
            node1 = (*it)[_indices[n1]];
#else
            node1 = (*it)[n1];
#endif
            // find index of node 2
            for (n2=0; n2<8; n2++)
            {
#ifndef SOFA_NEW_HEXA
                node2 = (*it)[_indices[n2]];
#else
                node2 = (*it)[n2];
#endif
                Mat33 tmp = Mat33(Coord(Me[3*n1+0][3*n2+0],Me[3*n1+0][3*n2+1],Me[3*n1+0][3*n2+2]),
                        Coord(Me[3*n1+1][3*n2+0],Me[3*n1+1][3*n2+1],Me[3*n1+1][3*n2+2]),
                        Coord(Me[3*n1+2][3*n2+0],Me[3*n1+2][3*n2+1],Me[3*n1+2][3*n2+2]));
                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        r.matrix->add(r.offset+3*node1+i, r.offset+3*node2+j, tmp[i][j]*mFactor);
            }
        }
    }
}

template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*a*/, const DataVecDeriv& /*f*/)
{
    serr<<"HexahedronFEMForceFieldAndMass<DataTypes>::accFromF not yet implemented"<<sendl;
    // need to built the big global mass matrix and to inverse it...
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();
        SReal dt = mparams->dt();
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

    // gravity
    // 		Vec3d g ( this->getContext()->getGravity() );
    // 		Deriv theGravity;
    // 		DataTypes::set ( theGravity, g[0], g[1], g[2]);

    helper::WriteAccessor< DataVecDeriv > _f = f;
#ifdef SOFA_SUPPORT_MOVING_FRAMES

    helper::ReadAccessor< DataVecDeriv > _v = v;
    helper::ReadAccessor< DataVecDeriv > _x = x;
    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector( aframe );

    // add weight and inertia force
    for (unsigned int i=0; i<_particleMasses.size(); i++)
    {
        _f[i] += this->getContext()->getGravity()*_particleMasses[i] + core::behavior::inertiaForce(vframe,aframe,_particleMasses[i],_x[i],_v[i]);
    }
#else
    for (unsigned int i=0; i<_particleMasses.size(); i++)
    {
        _f[i] += this->getContext()->getGravity()*_particleMasses[i];
    }
#endif
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    //if (mparams->kFactor() != 1.0)
    //{
    //	helper::ReadAccessor< DataVecDeriv> _dx = dx;
    //	DataVecDeriv kdx;// = dx * kFactor;
    //	helper::WriteAccessor< DataVecDeriv > _kdx = kdx;
    //	_kdx.resize(_dx.size());
    //	Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    //	for(unsigned i=0;i<_dx.size();++i)
    //		_kdx[i]=_dx[i]*kFactor;
    //	HexahedronFEMForceFieldT::addDForce(mparams, df,kdx);
    //}
    //else
    //{
    HexahedronFEMForceFieldT::addDForce(mparams, df, dx);
    //}
}


template<class DataTypes>
SReal HexahedronFEMForceFieldAndMass<DataTypes>::getElementMass(unsigned int /*index*/) const
{
    serr<<"HexahedronFEMForceFieldAndMass<DataTypes>::getElementMass not yet implemented"<<sendl; return 0.0;
}


template<class DataTypes>
void HexahedronFEMForceFieldAndMass<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    // 		  serr<<"HexahedronFEMForceFieldAndMass<DataTypes>::draw()  "<<this->getIndexedElements()->size()<<""<<sendl;
    HexahedronFEMForceFieldT::draw(vparams);

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    // since drawTool requires a std::vector<Vector3> we have to convert x in an ugly way
    std::vector<defaulttype::Vector3> pos;
    pos.resize(x.size());
    std::vector<defaulttype::Vector3>::iterator posIT = pos.begin();
    typename VecCoord::const_iterator xIT = x.begin();
    for(; posIT != pos.end() ; ++posIT, ++xIT)
    {
        *posIT = *xIT;
    }
    vparams->drawTool()->drawPoints(pos,2.0f, defaulttype::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONANDMASSFEMFORCEFIELD_INL
