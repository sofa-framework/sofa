/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_INL
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_INL


#include "HexahedralFEMForceFieldAndMass.h"
#include <sofa/core/visual/VisualParams.h>
#include "HexahedralFEMForceField.inl"

#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
HexahedralFEMForceFieldAndMass<DataTypes>::HexahedralFEMForceFieldAndMass()
    : MassT()
    , HexahedralFEMForceFieldT()
    , _density(initData(&_density,(Real)1.0,"density","density == volumetric mass in english (kg.m-3)"))
    , _useLumpedMass(initData(&_useLumpedMass, (bool)false, "lumpedMass", "Does it use lumped masses?"))
    , _elementMasses(initData(&_elementMasses,"massMatrices", "Mass matrices per element (M_i)",false))
    , _elementTotalMass(initData(&_elementTotalMass,"totalMass", "Total mass per element",false))
    , _particleMasses(initData(&_particleMasses, "particleMasses", "Mass per particle",false))
    , _lumpedMasses(initData(&_lumpedMasses, "lumpedMasses", "Lumped masses",false))
{
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::init( )
{
    this->core::behavior::ForceField<DataTypes>::init();

    this->getContext()->get(this->_topology);

    if(this->_topology == NULL)
    {
        serr << "ERROR(HexahedralFEMForceField): object must have a HexahedronSetTopology."<<sendl;
        return;
    }

    this->reinit();
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

/*
template <class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::handleTopologyChange(core::topology::Topology* t)
{
	if(t != this->_topology)
		return;

	HexahedralFEMForceFieldT::handleTopologyChange();

	std::list<const TopologyChange *>::const_iterator itBegin=this->_topology->beginChange();
	std::list<const TopologyChange *>::const_iterator itEnd=this->_topology->endChange();
#ifdef TODOTOPO
	// handle point events
	_particleMasses.handleTopologyEvents(itBegin,itEnd);

	if( _useLumpedMass.getValue() )
		_lumpedMasses.handleTopologyEvents(itBegin,itEnd);

	// handle hexa events
	_elementMasses.handleTopologyEvents(itBegin,itEnd);
	_elementTotalMass.handleTopologyEvents(itBegin,itEnd);
#endif

	for(std::list<const TopologyChange *>::const_iterator iter = itBegin;
		iter != itEnd; ++iter)
	{
		switch((*iter)->getChangeType())
		{
		// for added elements:
		// compute ElementMasses and TotalMass
		// add particle masses and lumped masses of adjacent particles
		case HEXAHEDRAADDED:
			{
				const VecElement& hexahedra = this->_topology->getHexahedra();
				const sofa::helper::vector<unsigned int> &hexaModif = (static_cast< const HexahedraAdded *> (*iter))->hexahedronIndexArray;

				const VecCoord& initialPoints = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

				helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
				helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();

				for(unsigned int i=0; i<hexaModif.size(); ++i)
				{
					const unsigned int hexaId = hexaModif[i];

					Vec<8,Coord> nodes;
					for(int w=0;w<8;++w)
						nodes[w] = initialPoints[hexahedra[hexaId][w]];

					computeElementMass( elementMasses[hexaId], elementTotalMass[hexaId],
										this->hexahedronInfo.getValue()[hexaId].rotatedInitialElements);
				}

				this->_elementTotalMass.endEdit();
				this->_elementMasses.endEdit();


				helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

				for(unsigned int i=0; i<hexaModif.size(); ++i)
				{
					const unsigned int hexaId = hexaModif[i];

					Real mass = _elementTotalMass.getValue()[hexaId] * (Real) 0.125;

					for(int w=0; w<8; ++w)
						particleMasses[ hexahedra[hexaId][w] ] += mass;
				}

				this->_particleMasses.endEdit();

				if( _useLumpedMass.getValue() )
				{
					helper::vector<Coord>&	lumpedMasses = *this->_lumpedMasses.beginEdit();

					for(unsigned int i=0; i<hexaModif.size(); ++i)
					{
						const unsigned int hexaId = hexaModif[i];
						const ElementMass& mass = this->_elementMasses.getValue()[hexaId];

						for(int w=0;w<8;++w)
						{
							for(int j=0;j<8*3;++j)
							{
								lumpedMasses[ hexahedra[hexaId][w] ][0] += mass[w*3  ][j];
								lumpedMasses[ hexahedra[hexaId][w] ][1] += mass[w*3+1][j];
								lumpedMasses[ hexahedra[hexaId][w] ][2] += mass[w*3+2][j];
							}
						}
					}

					this->_lumpedMasses.endEdit();
				}

			}
			break;

		// for removed elements:
		// subttract particle masses and lumped masses of adjacent particles
		case HEXAHEDRAREMOVED:
			{
				const VecElement& hexahedra = this->_topology->getHexahedra();
				const sofa::helper::vector<unsigned int> &hexaModif = (static_cast< const HexahedraRemoved *> (*iter))->getArray();

				helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

				for(unsigned int i=0; i<hexaModif.size(); ++i)
				{
					const unsigned int hexaId = hexaModif[i];

					Real mass = _elementTotalMass.getValue()[hexaId] * (Real) 0.125;

					for(int w=0; w<8; ++w)
						particleMasses[ hexahedra[hexaId][w] ] -= mass;
				}

				this->_particleMasses.endEdit();

				if( _useLumpedMass.getValue() )
				{
					helper::vector<Coord>&	lumpedMasses = *this->_lumpedMasses.beginEdit();

					for(unsigned int i=0; i<hexaModif.size(); ++i)
					{
						const unsigned int hexaId = hexaModif[i];
						const ElementMass& mass = this->_elementMasses.getValue()[hexaId];

						for(int w=0;w<8;++w)
						{
							for(int j=0;j<8*3;++j)
							{
								lumpedMasses[ hexahedra[hexaId][w] ][0] -= mass[w*3  ][j];
								lumpedMasses[ hexahedra[hexaId][w] ][1] -= mass[w*3+1][j];
								lumpedMasses[ hexahedra[hexaId][w] ][2] -= mass[w*3+2][j];
							}
						}
					}

					this->_lumpedMasses.endEdit();
				}
			}
			break;
		default:
			break;
		}
	}
}
*/

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeParticleMasses(  )
{
    unsigned int numPoints = this->_topology->getNbPoints();
    const VecElement& hexahedra = this->_topology->getHexahedra();

    helper::vector<Real>&	particleMasses = *this->_particleMasses.beginEdit();

    particleMasses.clear();
    particleMasses.resize( numPoints );

    for(unsigned int i=0; i<hexahedra.size(); ++i)
    {
        // mass of a particle...
        Real mass = _elementTotalMass.getValue()[i] * (Real) 0.125;

        // ... is added to each particle of the element
        for(int w=0; w<8; ++w)
            particleMasses[ hexahedra[i][w] ] += mass;
    }

    this->_particleMasses.endEdit();
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeLumpedMasses(  )
{
    unsigned int numPoints = this->_topology->getNbPoints();
    const VecElement& hexahedra = this->_topology->getHexahedra();

    if( _useLumpedMass.getValue() )
    {
        helper::vector<Coord>&	lumpedMasses = *this->_lumpedMasses.beginEdit();

        lumpedMasses.clear();
        lumpedMasses.resize( numPoints, Coord(0.0, 0.0, 0.0) );

        for(unsigned int i=0; i<hexahedra.size(); ++i)
        {
            const ElementMass& mass = this->_elementMasses.getValue()[i];

            for(int w=0; w<8; ++w)
            {
                for(int j=0; j<8*3; ++j)
                {
                    lumpedMasses[ hexahedra[i][w] ][0] += mass[w*3  ][j];
                    lumpedMasses[ hexahedra[i][w] ][1] += mass[w*3+1][j];
                    lumpedMasses[ hexahedra[i][w] ][2] += mass[w*3+2][j];
                }
            }
        }

        this->_lumpedMasses.endEdit();
    }
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeElementMasses(  )
{
    const VecCoord& initialPoints = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    const VecElement& hexahedra = this->_topology->getHexahedra();

    helper::vector<ElementMass>& elementMasses = *this->_elementMasses.beginEdit();
    helper::vector<Real>& elementTotalMass = *this->_elementTotalMass.beginEdit();

    elementMasses.resize( hexahedra.size() );
    elementTotalMass.resize( hexahedra.size() );

    for(unsigned int i=0; i<hexahedra.size(); ++i)
    {
        defaulttype::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = initialPoints[hexahedra[i][w]];

        computeElementMass( elementMasses[i], elementTotalMass[i],
                this->hexahedronInfo.getValue()[i].rotatedInitialElements);
    }

    this->_elementTotalMass.endEdit();
    this->_elementMasses.endEdit();
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::computeElementMass( ElementMass &Mass, Real& totalMass,
        const helper::fixed_array<Coord,8> &nodes)
{
    // volume of a element
    Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

    // total element mass
    totalMass = volume * _density.getValue();

    Coord l = nodes[6] - nodes[0];

    Mass.clear();

    for(int i=0; i<8; ++i)
    {
        Real mass = totalMass * integrateVolume(this->_coef[i][0],
                this->_coef[i][1],
                this->_coef[i][2],
                2.0f/l[0],
                2.0f/l[1],
                2.0f/l[2]);

        Mass[i*3][i*3] += mass;
        Mass[i*3+1][i*3+1] += mass;
        Mass[i*3+2][i*3+2] += mass;

        for(int j=i+1; j<8; ++j)
        {
            Real mass = totalMass * integrateVolume(this->_coef[i][0],
                    this->_coef[i][1],
                    this->_coef[i][2],
                    2.0f/l[0],
                    2.0f/l[1],
                    2.0f/l[2]);

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

    if( ! _useLumpedMass.getValue() )
    {
        const VecElement& hexahedra = this->_topology->getHexahedra();
        for(unsigned int i=0; i<hexahedra.size(); ++i)
        {
            defaulttype::Vec<24, Real> actualDx, actualF;

            for(int k=0 ; k<8 ; ++k )
            {
                int indice = k*3;
                for(int j=0 ; j<3 ; ++j )
                    actualDx[indice+j] = _dx[hexahedra[i][k]][j];
            }

            actualF = _elementMasses.getValue()[i] * actualDx;


            for(unsigned int w=0; w<8; ++w)
                _f[hexahedra[i][w]] += Deriv( actualF[w*3],  actualF[w*3+1],   actualF[w*3+2]  ) * factor;
        }
    }
    else // lumped matrices
    {
        for(unsigned int i=0; i<_lumpedMasses.getValue().size(); ++i)
            for(unsigned int j=0; j<3; ++j)
                _f[i][j] += (Real)(_lumpedMasses.getValue()[i][j] * _dx[i][j] *factor);
    }
}

template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addMToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    // Build Matrix Block for this ForceField
    int i, j, n1, n2;
    int node1, node2;

    const VecElement& hexahedra = this->_topology->getHexahedra();

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);

    for(unsigned int e=0; e<hexahedra.size(); ++e)
    {
        const ElementMass &Me = _elementMasses.getValue()[e];

        Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            node1 = hexahedra[e][n1];
            n2 = n1; /////////// WARNING Changed to compute only diag elements

            // find index of node 2
            //for (n2=0; n2<8; n2++) /////////// WARNING Changed to compute only diag elements
            {
                node2 = hexahedra[e][n2];

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

///// WARNING this method only add diagonal elements in the given matrix !
template<class DataTypes>
// void HexahedralFEMForceFieldAndMass<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *mat, SReal k, unsigned int &offset)
void HexahedralFEMForceFieldAndMass<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    // Build Matrix Block for this ForceField
    int i,j,n1, n2, e;

    //typename VecElement::const_iterator it;
    typename helper::vector<HexahedronInformation>::const_iterator it;

    Index node1, node2;
    const VecElement& hexahedra = this->_topology->getHexahedra();

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);

    for(it = this->hexahedronInfo.getValue().begin(), e=0 ; it != this->hexahedronInfo.getValue().end() ; ++it,++e)
    {
        const Element hexa = hexahedra[e];
        const ElementStiffness &Ke = it->stiffness;

        Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
        // find index of node 1
        for (n1=0; n1<8; n1++)
        {
            n2 = n1; /////////// WARNING Changed to compute only diag elements
#ifndef SOFA_NEW_HEXA
            node1 = hexa[_indices[n1]];
#else
            node1 = hexa[n1];
#endif
            // find index of node 2
            //for (n2=0; n2<8; n2++) /////////// WARNING Changed to compute only diag elements
            {
#ifndef SOFA_NEW_HEXA
                node2 = hexa[_indices[n2]];
#else
                node2 = hexa[n2];
#endif
                Mat33 tmp = it->rotation.multTranspose( Mat33(Coord(Ke[3*n1+0][3*n2+0],Ke[3*n1+0][3*n2+1],Ke[3*n1+0][3*n2+2]),
                        Coord(Ke[3*n1+1][3*n2+0],Ke[3*n1+1][3*n2+1],Ke[3*n1+1][3*n2+2]),
                        Coord(Ke[3*n1+2][3*n2+0],Ke[3*n1+2][3*n2+1],Ke[3*n1+2][3*n2+2])) ) * it->rotation;
                for(i=0; i<3; i++)
                    for (j=0; j<3; j++)
                        r.matrix->add(r.offset+3*node1+i, r.offset+3*node2+j, - tmp[i][j]*kFactor);
            }
        }
    }
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addMBKToMatrix (const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
// void HexahedralFEMForceFieldAndMass<DataTypes>::addMBKToMatrix ( sofa::defaulttype::BaseMatrix * matrix,
// double mFact, double /*bFact*/, double kFact, unsigned int &offset )
{
    int i, j, n1, n2;
    Index node1, node2;

    const VecElement& hexahedra = this->_topology->getHexahedra();

    //typename VecElement::const_iterator it;
    typename helper::vector<HexahedronInformation>::const_iterator it;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);

    for ( unsigned int e = 0; e < hexahedra.size(); ++e )
    {
        const ElementMass &Me = _elementMasses.getValue() [e];
        const Element hexa = hexahedra[e];
        const ElementStiffness &Ke = it->stiffness;

        // find index of node 1

        Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
        Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
        for ( n1 = 0; n1 < 8; n1++ )
        {
            n2 = n1; /////////// WARNING Changed to compute only diag elements
#ifndef SOFA_NEW_HEXA
            node1 = hexa[_indices[n1]];
#else
            node1 = hexa[n1];
#endif
            // find index of node 2
            //for (n2=0; n2<8; n2++) /////////// WARNING Changed to compute only diag elements
            {
#ifndef SOFA_NEW_HEXA
                node2 = hexa[_indices[n2]];
#else
                node2 = hexa[n2];
#endif
                // add M to matrix
                Mat33 tmp = Mat33 ( Coord ( Me[3*n1+0][3*n2+0], Me[3*n1+0][3*n2+1], Me[3*n1+0][3*n2+2] ),
                        Coord ( Me[3*n1+1][3*n2+0], Me[3*n1+1][3*n2+1], Me[3*n1+1][3*n2+2] ),
                        Coord ( Me[3*n1+2][3*n2+0], Me[3*n1+2][3*n2+1], Me[3*n1+2][3*n2+2] ) );
                for ( i = 0; i < 3; i++ )
                    for ( j = 0; j < 3; j++ )
                        r.matrix->add ( r.offset + 3*node1 + i, r.offset + 3*node2 + j, tmp[i][j]*mFactor);

                // add K to matrix
                tmp = it->rotation.multTranspose ( Mat33 ( Coord ( Ke[3*n1+0][3*n2+0], Ke[3*n1+0][3*n2+1], Ke[3*n1+0][3*n2+2] ),
                        Coord ( Ke[3*n1+1][3*n2+0], Ke[3*n1+1][3*n2+1], Ke[3*n1+1][3*n2+2] ),
                        Coord ( Ke[3*n1+2][3*n2+0], Ke[3*n1+2][3*n2+1], Ke[3*n1+2][3*n2+2] ) ) ) * it->rotation;
                for ( i = 0; i < 3; i++ )
                    for ( j = 0; j < 3; j++ )
                        r.matrix->add ( r.offset + 3*node1 + i, r.offset + 3*node2 + j, - tmp[i][j]*kFactor);
            }
        }
    }
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::accFromF(const core::MechanicalParams*, DataVecDeriv& /*a*/, const DataVecDeriv& /*f*/)
{
    serr<<"HexahedralFEMForceFieldAndMass<DataTypes>::accFromF not yet implemented"<<sendl;
    // need to built the big global mass matrix and to inverse it...
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        SReal _dt = mparams->dt();

        for (unsigned int i=0; i<_particleMasses.getValue().size(); i++)
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
    for (unsigned int i=0; i<_particleMasses.getValue().size(); i++)
    {
        _f[i] += this->getContext()->getGravity()*_particleMasses.getValue()[i];
    }
}



template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    //if (mparams->kFactor() != 1.0)
    //{
    //	helper::ReadAccessor< DataVecDeriv > _dx = dx;
    //	DataVecDeriv kdx;// = dx * kFactor;
    //	helper::WriteAccessor< DataVecDeriv > _kdx = kdx;
    //	_kdx.resize(_dx.size());
    //	Real _kFactor = (Real)mparams->kFactor();
    //	for(unsigned i=0;i<_dx.size();++i)
    //		_kdx[i]=_dx[i]*_kFactor;
    //	HexahedralFEMForceFieldT::addDForce(mparams, df,kdx);
    //}
    //else
    //{
    HexahedralFEMForceFieldT::addDForce(mparams, df, dx);
    //}
}


template<class DataTypes>
SReal HexahedralFEMForceFieldAndMass<DataTypes>::getElementMass(unsigned int /*index*/) const
{
    serr<<"HexahedralFEMForceFieldAndMass<DataTypes>::getElementMass not yet implemented"<<sendl; return 0.0;
}


template<class DataTypes>
void HexahedralFEMForceFieldAndMass<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    HexahedralFEMForceFieldT::draw(vparams);

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


#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELDANDMASS_INL
