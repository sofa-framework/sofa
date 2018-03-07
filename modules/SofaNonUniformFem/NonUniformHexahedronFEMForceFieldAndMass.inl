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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELD_INL

#include <SofaNonUniformFem/NonUniformHexahedronFEMForceFieldAndMass.h>
#include <sofa/core/visual/VisualParams.h>




#include <SofaNonUniformFem/SparseGridMultipleTopology.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template <class DataTypes>
void NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::init()
{
    if(this->_alreadyInit)return;
    else this->_alreadyInit=true;


    this->core::behavior::ForceField<DataTypes>::init();


    if( this->getContext()->getMeshTopology()==NULL )
    {
        serr << "ERROR(NonUniformHexahedronFEMForceFieldDensity): object must have a Topology."<<sendl;
        return;
    }

    this->_mesh = this->getContext()->getMeshTopology();
    if ( this->_mesh==NULL)
    {
        serr << "ERROR(NonUniformHexahedronFEMForceFieldDensity): object must have a MeshTopology."<<sendl;
        return;
    }
#ifdef SOFA_NEW_HEXA
    else if( this->_mesh->getNbHexahedra()<=0 )
#else
    else if( this->_mesh->getNbCubes()<=0 )
#endif
    {
        serr << "ERROR(NonUniformHexahedronFEMForceFieldDensity): object must have a hexahedric MeshTopology."<<sendl;
        serr << this->_mesh->getName()<<sendl;
        serr << this->_mesh->getTypeName()<<sendl;
        serr<<this->_mesh->getNbPoints()<<sendl;
        return;
    }

    this->_sparseGrid = dynamic_cast<topology::SparseGridTopology*>(this->_mesh);



    if (this->_initialPoints.getValue().size() == 0)
    {
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
        this->_initialPoints.setValue(p);
    }

    this->_materialsStiffnesses.resize(this->getIndexedElements()->size() );
    this->_rotations.resize( this->getIndexedElements()->size() );
    this->_rotatedInitialElements.resize(this->getIndexedElements()->size());


    // verify if it is wanted and possible to compute non-uniform stiffness
    if( !this->_nbVirtualFinerLevels.getValue() )
    {
        serr<<"ForceField "<<this->getName()<<" need 0 VirtualFinerLevels -> classical uniform properties are used." << sendl;
    }
    else if( !this->_sparseGrid )
    {
        this->_nbVirtualFinerLevels.setValue(0);
        serr<<"ForceField "<<this->getName()<<" must be used with a SparseGrid in order to handle VirtualFinerLevels -> classical uniform properties are used.." << sendl;
    }
    else if( this->_sparseGrid->getNbVirtualFinerLevels() < this->_nbVirtualFinerLevels.getValue()  )
    {
        this->_nbVirtualFinerLevels.setValue(0);
        serr<<"Conflict in nb of virtual levels between ForceField "<<this->getName()<<" and SparseGrid "<<this->_sparseGrid->getName()<<" -> classical uniform properties are used" << sendl;
    }



    this->_elementStiffnesses.beginEdit()->resize(this->getIndexedElements()->size());
    this->_elementMasses.beginEdit()->resize(this->getIndexedElements()->size());



    //////////////////////


    if (this->f_method.getValue() == "large")
        this->setMethod(HexahedronFEMForceFieldT::LARGE);
    else if (this->f_method.getValue() == "polar")
        this->setMethod(HexahedronFEMForceFieldT::POLAR);

    for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
    {
        sofa::defaulttype::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
            nodes[w] = this->_initialPoints.getValue()[(*this->getIndexedElements())[i][this->_indices[w]]];
#else
            nodes[w] = this->_initialPoints.getValue()[(*this->getIndexedElements())[i][w]];
#endif


        // compute initial configuration in order to compute corotationnal deformations
        if( this->method == HexahedronFEMForceFieldT::LARGE )
        {
            Coord horizontal;
            horizontal = (nodes[1]-nodes[0] + nodes[2]-nodes[3] + nodes[5]-nodes[4] + nodes[6]-nodes[7])*.25;
            Coord vertical;
            vertical = (nodes[3]-nodes[0] + nodes[2]-nodes[1] + nodes[7]-nodes[4] + nodes[6]-nodes[5])*.25;
            this->computeRotationLarge( this->_rotations[i], horizontal,vertical);
        }
        else
            this->computeRotationPolar( this->_rotations[i], nodes);
        for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
            this->_rotatedInitialElements[i][w] = this->_rotations[i]*this->_initialPoints.getValue()[(*this->getIndexedElements())[i][this->_indices[w]]];
#else
            this->_rotatedInitialElements[i][w] = this->_rotations[i]*this->_initialPoints.getValue()[(*this->getIndexedElements())[i][w]];
#endif
    }
    //////////////////////


    // compute mechanichal matrices (mass and stiffness) by condensating from _nbVirtualFinerLevels
    computeMechanicalMatricesByCondensation( );


    // 	post-traitement of non-uniform stiffness
    //         if( this->_nbVirtualFinerLevels.getValue() )
    //         {
    //           this->_sparseGrid->setNbVirtualFinerLevels(0);
    //           //delete undesirable sparsegrids and hexa
    //           for(int i=0;i<this->_sparseGrid->getNbVirtualFinerLevels();++i)
    //             delete this->_sparseGrid->_virtualFinerLevels[i];
    //           this->_sparseGrid->_virtualFinerLevels.resize(0);
    //         }




    // hack to use true mass matrices or masses concentrated in particules
    if(_useMass.getValue() )
    {

        MassT::init();
        this->_particleMasses.resize( this->_initialPoints.getValue().size() );


        int i=0;
        for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {
            sofa::defaulttype::Vec<8,Coord> nodes;
            for(int w=0; w<8; ++w)
#ifndef SOFA_NEW_HEXA
                nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];
#else
                nodes[w] = this->_initialPoints.getValue()[(*it)[w]];
#endif

            // volume of a element
            Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

            volume *= (Real) (this->_sparseGrid->getMassCoef(i));


            // mass of a particle...
            Real mass = Real (( volume * this->_density.getValue() ) / 8.0 );

            // ... is added to each particle of the element
            for(int w=0; w<8; ++w)
                this->_particleMasses[ (*it)[w] ] += mass;
        }





        if( this->_lumpedMass.getValue() )
        {
            this->_lumpedMasses.resize( this->_initialPoints.getValue().size() );
            i=0;
            for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
            {

                const ElementMass& mass=this->_elementMasses.getValue()[i];

                for(int w=0; w<8; ++w)
                {
                    for(int j=0; j<8*3; ++j)
                    {
                        this->_lumpedMasses[ (*it)[w] ][0] += mass[w*3  ][j];
                        this->_lumpedMasses[ (*it)[w] ][1] += mass[w*3+1][j];
                        this->_lumpedMasses[ (*it)[w] ][2] += mass[w*3+2][j];
                    }
                }
            }

            for(unsigned j=0; j<this->_lumpedMasses.size(); ++j)
            {
                for(int k=0; k<3; ++k)
                    if( this->_lumpedMasses[j][k] < 0 )
                    {
                        // 					  serr<<"WARNING lumped mass"<<sendl;
                        this->_lumpedMasses[ j ][k] = -this->_lumpedMasses[ j ][k];
                    }
            }
        }


    }
    else
    {
        this->_particleMasses.resize( this->_initialPoints.getValue().size() );
        // 		int nbboundary=0;
        // 		int i=0;
        // 		for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        // 		{
        // 			if( this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY ) ++nbboundary;
        // 		}
        // 		Real semielemmass = _totalMass.getValue() / Real( 2*this->getIndexedElements()->size() - nbboundary);
        //
        // 		for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        // 		{
        // 			for(int w=0;w<8;++w)
        // 				if( this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY )
        // 					this->_particleMasses[ (*it)[w] ] += semielemmass;
        // 				else
        // 					this->_particleMasses[ (*it)[w] ] += 2.0*semielemmass;
        // 		}

        Real mass = _totalMass.getValue() / Real(this->getIndexedElements()->size());
        for(unsigned i=0; i<this->_particleMasses.size(); ++i)
            this->_particleMasses[ i ] = mass;
    }






}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation( )
{
    for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
    {
        computeMechanicalMatricesByCondensation( (*this->_elementStiffnesses.beginEdit())[i],
                (*this->_elementMasses.beginEdit())[i],i,0);
    }
}

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level)
{

    if (level == this->_nbVirtualFinerLevels.getValue())
        computeClassicalMechanicalMatrices(K,M,elementIndice,this->_sparseGrid->getNbVirtualFinerLevels()-level);
    else
    {
        helper::fixed_array<int,8> finerChildren;
        if (level == 0)
        {
            finerChildren = this->_sparseGrid->_hierarchicalCubeMap[elementIndice];
        }
        else
        {
            finerChildren = this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level]->_hierarchicalCubeMap[elementIndice];
        }

        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            if (finerChildren[i] != -1)
            {
                ElementStiffness finerK;
                ElementMass finerM;
                computeMechanicalMatricesByCondensation(finerK, finerM, finerChildren[i], level+1);
                this->addFineToCoarse(K, finerK, i);
                this->addFineToCoarse(M, finerM, i);
            }
        }
    }
}




template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeClassicalMechanicalMatrices( ElementStiffness &K, ElementMass &M, const int elementIndice, int level)
{
    //Get the 8 indices of the coarser Hexa
    const helper::fixed_array<unsigned int,8>& points = this->_sparseGrid->_virtualFinerLevels[level]->getHexahedra()[elementIndice];
    //Get the 8 points of the coarser Hexa
    helper::fixed_array<Coord,8> nodes;

#ifndef SOFA_NEW_HEXA
    //           for (unsigned int k=0;k<8;++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[level]->getPointPos(points[this->_indices[k]]);
    for (unsigned int k=0; k<8; ++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[level]->getPointPos(points[this->_indices[k]]).linearProduct(this->mstate->getScale());
#else
    //           for (unsigned int k=0;k<8;++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[level]->getPointPos(points[k]);
    for (unsigned int k=0; k<8; ++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[level]->getPointPos(points[k]).linearProduct(this->mstate->getScale());
#endif

    //       //given an elementIndice, find the 8 others from the sparse grid
    //       //compute MaterialStiffness
    MaterialStiffness material;
    computeMaterialStiffness(material, this->f_youngModulus.getValue(),this->f_poissonRatio.getValue());

    //Nodes are found using Sparse Grid
    Real stiffnessCoef = this->_sparseGrid->_virtualFinerLevels[level]->getStiffnessCoef(elementIndice);
    Real massCoef = this->_sparseGrid->_virtualFinerLevels[level]->getStiffnessCoef(elementIndice);
    // 		  if (SparseGridMultipleTopology* sgmt =  dynamic_cast<SparseGridMultipleTopology*>( this->_sparseGrid->_virtualFinerLevels[level] ) )
    // 			  stiffnessCoef = sgmt->getStiffnessCoef( elementIndice );
    // 		  else if( this->_sparseGrid->_virtualFinerLevels[level]->getType(elementIndice)==topology::SparseGridTopology::BOUNDARY )
    // 			  stiffnessCoef = .5;


    HexahedronFEMForceFieldAndMassT::computeElementStiffness(K,material,nodes,elementIndice, stiffnessCoef); // classical stiffness

    HexahedronFEMForceFieldAndMassT::computeElementMass(M,nodes,elementIndice,massCoef);
}




template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addFineToCoarse( ElementStiffness& coarse, const ElementStiffness& fine, int indice )
{
    ElementStiffness A;
    for(int i=0; i<24; i++)
        for(int j=0; j<24; j++)
        {
            A[i][j] = j%3==0 ? fine[i][0] *(Real) FINE_TO_COARSE[indice][0][j/3] : Real(0.0);
            for(int k=1; k<24; k++)
                A[i][j] += j%3==k%3  ? fine[i][k] * (Real)FINE_TO_COARSE[indice][k/3][j/3] : Real(0.0);
        }

    for(int i=0; i<24; i++)
        for(int j=0; j<24; j++)
            for(int k=0; k<24; k++)
                coarse[i][j] += i%3==k%3  ? (Real)FINE_TO_COARSE[indice][k/3][i/3] * A[k][j] : Real(0.0);   // FINE_TO_COARSE[indice] transposed
}








template<class T>
const float NonUniformHexahedronFEMForceFieldAndMass<T>::FINE_TO_COARSE[8][8][8]=
{
    {
        {1,0,0,0,0,0,0,0},
        {0.5,0.5,0,0,0,0,0,0},
        {0.25,0.25,0.25,0.25,0,0,0,0},
        {0.5,0,0,0.5,0,0,0,0},
        {0.5,0,0,0,0.5,0,0,0},
        {0.25,0.25,0,0,0.25,0.25,0,0},
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
        {0.25,0,0,0.25,0.25,0,0,0.25}
    },
    {
        {0.5,0.5,0,0,0,0,0,0},
        {0,1,0,0,0,0,0,0},
        {0,0.5,0.5,0,0,0,0,0},
        {0.25,0.25,0.25,0.25,0,0,0,0},
        {0.25,0.25,0,0,0.25,0.25,0,0},
        {0,0.5,0,0,0,0.5,0,0},
        {0,0.25,0.25,0,0,0.25,0.25,0},
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125}
    },
    {
        {0.5,0,0,0.5,0,0,0,0},
        {0.25,0.25,0.25,0.25,0,0,0,0},
        {0,0,0.5,0.5,0,0,0,0},
        {0,0,0,1,0,0,0,0},
        {0.25,0,0,0.25,0.25,0,0,0.25},
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
        {0,0,0.25,0.25,0,0,0.25,0.25},
        {0,0,0,0.5,0,0,0,0.5}
    },
    {
        {0.25,0.25,0.25,0.25,0,0,0,0},
        {0,0.5,0.5,0,0,0,0,0},
        {0,0,1,0,0,0,0,0},
        {0,0,0.5,0.5,0,0,0,0},
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
        {0,0.25,0.25,0,0,0.25,0.25,0},
        {0,0,0.5,0,0,0,0.5,0},
        {0,0,0.25,0.25,0,0,0.25,0.25}
    },
    {
        {0.5,0,0,0,0.5,0,0,0},
        {0.25,0.25,0,0,0.25,0.25,0,0},
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
        {0.25,0,0,0.25,0.25,0,0,0.25},
        {0,0,0,0,1,0,0,0},
        {0,0,0,0,0.5,0.5,0,0},
        {0,0,0,0,0.25,0.25,0.25,0.25},
        {0,0,0,0,0.5,0,0,0.5}
    },
    {
        {0.25,0.25,0,0,0.25,0.25,0,0},
        {0,0.5,0,0,0,0.5,0,0},
        {0,0.25,0.25,0,0,0.25,0.25,0},
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
        {0,0,0,0,0.5,0.5,0,0},
        {0,0,0,0,0,1,0,0},
        {0,0,0,0,0,0.5,0.5,0},
        {0,0,0,0,0.25,0.25,0.25,0.25}
    },
    {
        {0.25,0,0,0.25,0.25,0,0,0.25},
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
        {0,0,0.25,0.25,0,0,0.25,0.25},
        {0,0,0,0.5,0,0,0,0.5},
        {0,0,0,0,0.5,0,0,0.5},
        {0,0,0,0,0.25,0.25,0.25,0.25},
        {0,0,0,0,0,0,0.5,0.5},
        {0,0,0,0,0,0,0,1}
    },
    {
        {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
        {0,0.25,0.25,0,0,0.25,0.25,0},
        {0,0,0.5,0,0,0,0.5,0},
        {0,0,0.25,0.25,0,0,0.25,0.25},
        {0,0,0,0,0.25,0.25,0.25,0.25},
        {0,0,0,0,0,0.5,0.5,0},
        {0,0,0,0,0,0,1,0},
        {0,0,0,0,0,0,0.5,0.5}
    }

};



template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio)
{
    m[0][0] = m[1][1] = m[2][2] = 1;
    m[0][1] = m[0][2] = m[1][0]= m[1][2] = m[2][0] =  m[2][1] = (Real)(poissonRatio/(1-poissonRatio));
    m[0][3] = m[0][4] =	m[0][5] = 0;
    m[1][3] = m[1][4] =	m[1][5] = 0;
    m[2][3] = m[2][4] =	m[2][5] = 0;
    m[3][0] = m[3][1] = m[3][2] = m[3][4] =	m[3][5] = 0;
    m[4][0] = m[4][1] = m[4][2] = m[4][3] =	m[4][5] = 0;
    m[5][0] = m[5][1] = m[5][2] = m[5][3] =	m[5][4] = 0;
    m[3][3] = m[4][4] = m[5][5] = (Real)((1-2*poissonRatio)/(2*(1-poissonRatio)));
    m *= (Real)((youngModulus*(1-poissonRatio))/((1+poissonRatio)*(1-2*poissonRatio)));
}



/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor)
{
    if(_useMass.getValue())
        HexahedronFEMForceFieldAndMassT::addMDx(mparams, f,dx,factor);
    else
    {
        helper::WriteAccessor<DataVecDeriv> _f = f;
        helper::ReadAccessor<DataVecDeriv> _dx = dx;

        for (unsigned int i=0; i<_dx.size(); i++)
        {
            _f[i] += _dx[i] * this->_particleMasses[i] * (Real)factor;
        }
    }
}

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(_useMass.getValue())
        HexahedronFEMForceFieldAndMassT::addGravityToV(mparams, d_v);
    else
    {
        if(mparams)
        {
            VecDeriv& v = *d_v.beginEdit();

            const SReal* g = this->getContext()->getGravity().ptr();
            Deriv theGravity;
            T::set( theGravity, (Real)g[0], (Real)g[1], (Real)g[2]);
            Deriv hg = theGravity * (mparams->dt());
            for (unsigned int i=0; i<v.size(); i++)
            {
                v[i] += hg;
            }
            d_v.endEdit();
        }
    }
}

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    if(_useMass.getValue())
        HexahedronFEMForceFieldAndMassT::addForce(mparams, f,x,v);
    else
    {
        HexahedronFEMForceFieldT::addForce(mparams, f,x,v);

        helper::WriteAccessor<DataVecDeriv> _f = f;

        const SReal* g = this->getContext()->getGravity().ptr();
        Deriv theGravity;
        T::set( theGravity, g[0], g[1], g[2]);

        for (unsigned int i=0; i<_f.size(); i++)
        {
            _f[i] += theGravity * this->_particleMasses[i];
        }
    }
}



} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELD_INL
