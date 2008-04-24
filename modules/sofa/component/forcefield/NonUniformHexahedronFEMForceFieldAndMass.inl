/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELD_INL

#include <sofa/component/forcefield/NonUniformHexahedronFEMForceFieldAndMass.h>

using std::cerr;
using std::endl;
using std::set;





namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;



template <class DataTypes>
void NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::init()
{
// 	cerr<<"NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::init()\n";


    if(this->_alreadyInit)return;
    else this->_alreadyInit=true;

    this->core::componentmodel::behavior::ForceField<DataTypes>::init();
    if( this->getContext()->getTopology()==NULL )
    {
        std::cerr << "ERROR(NonUniformHexahedronFEMForceFieldAndMass): object must have a Topology.\n";
        return;
    }

    this->_mesh = dynamic_cast<sofa::component::topology::MeshTopology*>(this->getContext()->getTopology());
    if ( this->_mesh==NULL)
    {
        std::cerr << "ERROR(NonUniformHexahedronFEMForceFieldAndMass): object must have a MeshTopology.\n";
        return;
    }
    else if( this->_mesh->getNbCubes()<=0 )
    {
        std::cerr << "ERROR(NonUniformHexahedronFEMForceFieldAndMass): object must have a hexahedric MeshTopology.\n";
        std::cerr << this->_mesh->getName()<<std::endl;
        std::cerr << this->_mesh->getTypeName()<<std::endl;
        cerr<<this->_mesh->getNbPoints()<<endl;
        return;
    }

    this->_indexedElements = & (this->_mesh->getCubes());


    this->_sparseGrid = dynamic_cast<topology::SparseGridTopology*>(this->_mesh);



    if (this->_initialPoints.getValue().size() == 0)
    {
        VecCoord& p = *this->mstate->getX();
        this->_initialPoints.setValue(p);
    }

    this->_materialsStiffnesses.resize(this->_indexedElements->size() );
    this->_rotations.resize( this->_indexedElements->size() );
    this->_rotatedInitialElements.resize(this->_indexedElements->size());


    // verify if it is wanted and possible to compute non-uniform stiffness
    if( !_nbVirtualFinerLevels.getValue() || !this->_sparseGrid || this->_sparseGrid->getNbVirtualFinerLevels() < _nbVirtualFinerLevels.getValue()  )
    {
        _nbVirtualFinerLevels = 0;
        cerr<<"WARNING: NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::init    conflict in nb of virtual levels between ForceField "<<this->getName()<<" and SparseGrid "<<this->_sparseGrid->getName()<<" -> classical uniform properties are used"<<endl;
    }
    else
    {
// 		this->f_updateStiffnessMatrix.setValue(false);
        //build virtual finer levels
        for(int i=0; i<_nbVirtualFinerLevels.getValue(); ++i)
        {
            _virtualFinerLevels.push_back( new NonUniformHexahedronFEMForceFieldAndMass<DataTypes>() );

            NonUniformHexahedronFEMForceFieldAndMass<DataTypes>* finer = _virtualFinerLevels.back();

            finer->_sparseGrid = this->_sparseGrid->_virtualFinerLevels[	this->_sparseGrid->getNbVirtualFinerLevels()-_nbVirtualFinerLevels.getValue() + i];
// 			finer->setName( "virtual" );


            finer->_indexedElements = & (finer->_sparseGrid->getCubes());
// 			cerr<<finer->_indexedElements->size()<<endl;





            if( i==0 ) //finest
            {
                finer->setYoungModulus( this->f_youngModulus.getValue() );
                finer->setPoissonRatio( this->f_poissonRatio.getValue() );


                finer->setMethod( this->method );

                finer->_materialsStiffnesses.resize(finer->_indexedElements->size() );
                finer->_rotations.resize( finer->_indexedElements->size() );
                finer->_rotatedInitialElements.resize(finer->_indexedElements->size());

                Real scale(0.0);
                if( MechanicalObject<DataTypes>*mo=dynamic_cast<MechanicalObject<DataTypes>*>(this->mstate)) scale=mo->getScale();

                int nbp = finer->_sparseGrid->getNbPoints();
                finer->_initialPoints.beginEdit()->resize(nbp);
                for (int j=0; j<nbp; j++)
                {
                    (*finer->_initialPoints.beginEdit())[j] = Coord( finer->_sparseGrid->getPX(j)*scale, finer->_sparseGrid->getPY(j)*scale, finer->_sparseGrid->getPZ(j)*scale);
                }
// 				cerr<<finer->_initialPoints.getValue()<<endl;
                finer->reinit();
            }
            else
            {
                finer->_finerLevel = _virtualFinerLevels[i-1];

                unsigned int j=0;
                typename VecElement::const_iterator it;
                for(it = finer->_indexedElements->begin() ; it != finer->_indexedElements->end() ; ++it, ++j)
                {
                    finer->_elementStiffnesses.beginEdit()->resize( finer->_elementStiffnesses.getValue().size()+1 );
                    finer->computeElementStiffnessFromFiner( (*finer->_elementStiffnesses.beginEdit())[j],j);

                    if( _useMass.getValue())
                    {
                        finer->_elementMasses.beginEdit()->resize( finer->_elementMasses.getValue().size()+1 );
                        finer->computeElementMassFromFiner( (*finer->_elementMasses.beginEdit())[j],j);
                    }
                }

            }

        }
        _finerLevel = _virtualFinerLevels.back();
    }



    this->reinit();











    // post-traitement of non-uniform stiffness
    if( _nbVirtualFinerLevels.getValue() )
    {
        this->_sparseGrid->setNbVirtualFinerLevels(0);
        //delete undesirable sparsegrids and hexa
        for(int i=0; i<this->_sparseGrid->getNbVirtualFinerLevels(); ++i)
            delete this->_sparseGrid->_virtualFinerLevels[i];
        this->_sparseGrid->_virtualFinerLevels.resize(0);
        for(int i=0; i<_nbVirtualFinerLevels.getValue(); ++i)
            delete _virtualFinerLevels[i];
        _virtualFinerLevels.resize(0);
        _finerLevel=NULL;
    }



    if(_useMass.getValue() )
    {

        MassT::init();
        this->_particleMasses.resize( this->_initialPoints.getValue().size() );


        int i=0;
        for(typename VecElement::const_iterator it = this->_indexedElements->begin() ; it != this->_indexedElements->end() ; ++it, ++i)
        {
            Vec<8,Coord> nodes;
            for(int w=0; w<8; ++w)
                nodes[w] = this->_initialPoints.getValue()[(*it)[this->_indices[w]]];

            // volume of a element
            Real volume = (nodes[1]-nodes[0]).norm()*(nodes[3]-nodes[0]).norm()*(nodes[4]-nodes[0]).norm();

            volume *= (Real) (this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5:1.0);

            // mass of a particle...
            Real mass = Real (( volume * this->_density.getValue() ) / 8.0 );

            // ... is added to each particle of the element
            for(int w=0; w<8; ++w)
                this->_particleMasses[ (*it)[w] ] += mass;
        }
    }
    else
    {
        this->_particleMasses.resize( this->_initialPoints.getValue().size() );
// 		int nbboundary=0;
// 		int i=0;
// 		for(typename VecElement::const_iterator it = this->_indexedElements->begin() ; it != this->_indexedElements->end() ; ++it, ++i)
// 		{
// 			if( this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY ) ++nbboundary;
// 		}
// 		Real semielemmass = _totalMass.getValue() / Real( 2*this->_indexedElements->size() - nbboundary);
//
// 		for(typename VecElement::const_iterator it = this->_indexedElements->begin() ; it != this->_indexedElements->end() ; ++it, ++i)
// 		{
// 			for(int w=0;w<8;++w)
// 				if( this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY )
// 					this->_particleMasses[ (*it)[w] ] += semielemmass;
// 				else
// 					this->_particleMasses[ (*it)[w] ] += 2.0*semielemmass;
// 		}

        Real mass = _totalMass.getValue() / Real(this->_indexedElements->size());
        for(unsigned i=0; i<this->_particleMasses.size(); ++i)
            this->_particleMasses[ i ] = mass;
    }


}


// template <class DataTypes>
// 		void NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::reinit()
// {
// 	cerr<<"NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::reinit() "<<this->_sparseGrid->getNbCubes()<<endl;
// // 	this->computeElementMasses();
//
// 	this->f_updateStiffnessMatrix.setValue(0);
//
// 	if( !_useMass.getValue())
// 	{
// 		Real mass = _totalMass.getValue() / Real(this->_indexedElements->size());
// 		for(unsigned i=0;i<this->_particleMasses.size();++i)
// 			this->_particleMasses[ i ] = mass;
// 	}
//
// }

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////



template<class DataTypes>
void NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const helper::fixed_array<Coord,8> &nodes, const int elementIndice)
{
    if( _finerLevel )
        computeElementStiffnessFromFiner(K,elementIndice); // non-uniform stiffness
    else
    {
        HexahedronFEMForceFieldT::computeElementStiffness(K,M,nodes,elementIndice); // classical stiffness
    }
}

template<class DataTypes>
void NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::computeElementStiffnessFromFiner( ElementStiffness &K, const int elementIndice)
{
// 	cerr<<"NonUniformHexahedronFEMForceFieldAndMass<DataTypes>::computeElementStiffnessFromFiner\n";

    helper::fixed_array<int,8>& children = this->_sparseGrid->_hierarchicalCubeMap[elementIndice];
    K.fill(0.0);

    for(int i=0; i<8; ++i)
    {

        if( children[i] == -1 ) continue; // outside == void

        const ElementStiffness &Kchild = _finerLevel->_elementStiffnesses.getValue()[children[i]];

        addFineToCoarse(K, Kchild, i);
    }


}



template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeElementMass( ElementMass &Mass, const helper::fixed_array<Coord,8> &nodes, const int elementIndice)
{
    if(_useMass.getValue() )
    {
// 		cerr<<"NonUniformHexahedronFEMForceFieldAndMass<T>::computeElementMass "<<_finerLevel<<endl;
        if( _finerLevel )
            computeElementMassFromFiner(Mass,elementIndice); // non-uniform stiffness
        else
            HexahedronFEMForceFieldAndMassT::computeElementMass(Mass,nodes,elementIndice); // classical stiffness
    }
}


template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::computeElementMassFromFiner( ElementMass &Mass, const int elementIndice)
{
// 	cerr<<"NonUniformHexahedronFEMForceFieldAndMass<T>::computeElementMassFromFiner\n";
    helper::fixed_array<int,8>& children = this->_sparseGrid->_hierarchicalCubeMap[elementIndice];
    Mass.fill(0.0);

    for(int i=0; i<8; ++i)
    {
        if( children[i] == -1 ) continue; // outside

        const ElementMass &Mchild = _finerLevel->_elementMasses.getValue()[children[i]];

        addFineToCoarse(Mass, Mchild, i);
    }
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

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addMDx(VecDeriv& f, const VecDeriv& dx, Real_Sofa factor)
{
    if(_useMass.getValue())
        HexahedronFEMForceFieldAndMassT::addMDx(f,dx,factor);
    else
    {
        for (unsigned int i=0; i<dx.size(); i++)
        {
            f[i] += dx[i] * this->_particleMasses[i] * (Real)factor;
        }
    }
}

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addGravityToV(double dt)
{
    if(_useMass.getValue())
        HexahedronFEMForceFieldAndMassT::addGravityToV(dt);
    else
    {
        VecDeriv& v = *this->mstate->getV();
        const Real_Sofa* g = this->getContext()->getLocalGravity().ptr();
        Deriv theGravity;
        T::set( theGravity, (Real)g[0], (Real)g[1], (Real)g[2]);
        Deriv hg = theGravity * dt;
        for (unsigned int i=0; i<v.size(); i++)
        {
            v[i] += hg;
        }
    }
}

template<class T>
void NonUniformHexahedronFEMForceFieldAndMass<T>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if(_useMass.getValue())
        HexahedronFEMForceFieldAndMassT::addForce(f,x,v);
    else
    {
        HexahedronFEMForceFieldT::addForce(f,x,v);

        const Real_Sofa* g = this->getContext()->getLocalGravity().ptr();
        Deriv theGravity;
        T::set( theGravity, g[0], g[1], g[2]);

        for (unsigned int i=0; i<f.size(); i++)
        {
            f[i] += theGravity * this->_particleMasses[i];
        }
    }
}



} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
