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

#include <sofa/component/solidmechanics/fem/nonuniform/NonUniformHexahedronFEMForceFieldAndMass.h>
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceFieldAndMass.inl>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::solidmechanics::fem::nonuniform
{

template <class DataTypes>
NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::NonUniformHexahedronFEMForceFieldAndMass()
    : HexahedronFEMForceFieldAndMassT()
    , d_nbVirtualFinerLevels(initData(&d_nbVirtualFinerLevels,0,"nbVirtualFinerLevels","use virtual finer levels, in order to compte non-uniform stiffness"))
    , d_useMass(initData(&d_useMass,true,"useMass","Using this ForceField like a Mass? (rather than using a separated Mass)"))
    , d_totalMass(initData(&d_totalMass,(Real)0.0,"totalMass",""))
{
}

template <class DataTypes>
void NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::init()
{
    if(this->_alreadyInit)return;
    else this->_alreadyInit=true;


    elastic::BaseLinearElasticityFEMForceField<DataTypes>::init();

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    if(this->l_topology->getNbHexahedra()<=0 )
    {
        msg_error() << "NonUniformHexahedronFEMForceFieldDensity: object must have a hexahedric MeshTopology.\n"
                    << this->l_topology->getName() << "\n"
                    << this->l_topology->getTypeName() << "\n"
                    << this->l_topology->getNbPoints() << "\n";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    this->_sparseGrid = dynamic_cast<topology::container::grid::SparseGridTopology*>(this->l_topology.get());



    if (this->d_initialPoints.getValue().size() == 0)
    {
        const VecCoord& p = this->mstate->read(core::vec_id::read_access::position)->getValue();
        this->d_initialPoints.setValue(p);
    }

    this->_materialsStiffnesses.resize(this->getIndexedElements()->size() );
    this->_rotations.resize( this->getIndexedElements()->size() );
    this->_rotatedInitialElements.resize(this->getIndexedElements()->size());


    // verify if it is wanted and possible to compute non-uniform stiffness
    if( !this->d_nbVirtualFinerLevels.getValue() )
    {
        msg_error()<<"ForceField "<<this->getName()<<" need 0 VirtualFinerLevels -> classical uniform properties are used." ;
    }
    else if( !this->_sparseGrid )
    {
        this->d_nbVirtualFinerLevels.setValue(0);
        msg_error()<<"ForceField "<<this->getName()<<" must be used with a SparseGrid in order to handle VirtualFinerLevels -> classical uniform properties are used..";
    }
    else if( this->_sparseGrid->getNbVirtualFinerLevels() < this->d_nbVirtualFinerLevels.getValue()  )
    {
        this->d_nbVirtualFinerLevels.setValue(0);
        msg_error()<<"Conflict in nb of virtual levels between ForceField "<<this->getName()<<" and SparseGrid "<<this->_sparseGrid->getName()<<" -> classical uniform properties are used";
    }



    this->d_elementStiffnesses.beginEdit()->resize(this->getIndexedElements()->size());
    this->d_elementMasses.beginEdit()->resize(this->getIndexedElements()->size());



    //////////////////////


    if (this->d_method.getValue() == "large")
        this->setMethod(HexahedronFEMForceFieldT::LARGE);
    else if (this->d_method.getValue() == "polar")
        this->setMethod(HexahedronFEMForceFieldT::POLAR);

    for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
    {
        sofa::type::Vec<8,Coord> nodes;
        for(int w=0; w<8; ++w)
            nodes[w] = this->d_initialPoints.getValue()[(*this->getIndexedElements())[i][w]];


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
            this->_rotatedInitialElements[i][w] = this->_rotations[i]*this->d_initialPoints.getValue()[(*this->getIndexedElements())[i][w]];
    }
    //////////////////////


    // compute mechanichal matrices (mass and stiffness) by condensating from d_nbVirtualFinerLevels
    computeMechanicalMatricesByCondensation( );
    // hack to use true mass matrices or masses concentrated in particules
    if(d_useMass.getValue() )
    {

        MassT::init();
        this->_particleMasses.resize( this->d_initialPoints.getValue().size() );


        int i=0;
        for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
        {
            sofa::type::Vec<8,Coord> nodes;
            for(int w=0; w<8; ++w)
                nodes[w] = this->d_initialPoints.getValue()[(*it)[w]];

            // volume of a element
            Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

            volume *= (Real) (this->_sparseGrid->getMassCoef(i));


            // mass of a particle...
            Real mass = Real (( volume * this->d_density.getValue() ) / 8.0 );

            // ... is added to each particle of the element
            for(int w=0; w<8; ++w)
                this->_particleMasses[ (*it)[w] ] += mass;
        }





        if( this->d_lumpedMass.getValue() )
        {
            this->_lumpedMasses.resize( this->d_initialPoints.getValue().size() );
            i=0;
            for(typename VecElement::const_iterator it = this->getIndexedElements()->begin() ; it != this->getIndexedElements()->end() ; ++it, ++i)
            {

                const ElementMass& mass=this->d_elementMasses.getValue()[i];

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
                        this->_lumpedMasses[ j ][k] = -this->_lumpedMasses[ j ][k];
                    }
            }
        }
    }
    else
    {
        this->_particleMasses.resize( this->d_initialPoints.getValue().size() );
        Real mass = d_totalMass.getValue() / Real(this->getIndexedElements()->size());
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
        computeMechanicalMatricesByCondensation( (*this->d_elementStiffnesses.beginEdit())[i],
                (*this->d_elementMasses.beginEdit())[i],i,0);
    }
}

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level)
{

    if (level == this->d_nbVirtualFinerLevels.getValue())
        computeClassicalMechanicalMatrices(K,M,elementIndice,this->_sparseGrid->getNbVirtualFinerLevels()-level);
    else
    {
        type::fixed_array<Index, 8> finerChildren;
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
            if (finerChildren[i] != sofa::InvalidID)
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
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeClassicalMechanicalMatrices( ElementStiffness &K, ElementMass &M, const Index elementIndice, int level)
{
    //Get the 8 indices of the coarser Hexa
    const auto& points = this->_sparseGrid->_virtualFinerLevels[level]->getHexahedra()[elementIndice];
    //Get the 8 points of the coarser Hexa
    type::fixed_array<Coord,8> nodes;

    //           for (unsigned int k=0;k<8;++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[level]->getPointPos(points[k]);
    for (unsigned int k=0; k<8; ++k) nodes[k] =  this->_sparseGrid->_virtualFinerLevels[level]->getPointPos(points[k]).linearProduct(this->mstate->getScale());

    //       //given an elementIndice, find the 8 others from the sparse grid
    //       //compute MaterialStiffness
    MaterialStiffness material;
    computeMaterialStiffness(material, this->getYoungModulusInElement(0), this->getPoissonRatioInElement(0));

    //Nodes are found using Sparse Grid
    Real stiffnessCoef = this->_sparseGrid->_virtualFinerLevels[level]->getStiffnessCoef(elementIndice);
    Real massCoef = this->_sparseGrid->_virtualFinerLevels[level]->getStiffnessCoef(elementIndice);

    HexahedronFEMForceFieldAndMassT::computeElementStiffness(K,material,nodes,elementIndice, stiffnessCoef); // classical stiffness

    HexahedronFEMForceFieldAndMassT::computeElementMass(M,nodes,elementIndice,massCoef);
}




template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addFineToCoarse( ElementStiffness& coarse, const ElementStiffness& fine, Index indice )
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
    if(d_useMass.getValue())
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
    if(d_useMass.getValue())
        HexahedronFEMForceFieldAndMassT::addGravityToV(mparams, d_v);
    else
    {
        if(mparams)
        {
            VecDeriv& v = *d_v.beginEdit();

            const SReal* g = this->getContext()->getGravity().ptr();
            Deriv theGravity;
            T::set( theGravity, (Real)g[0], (Real)g[1], (Real)g[2]);
            Deriv hg = theGravity * (sofa::core::mechanicalparams::dt(mparams));
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
    if(d_useMass.getValue())
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

} // namespace sofa::component::solidmechanics::fem::nonuniform

