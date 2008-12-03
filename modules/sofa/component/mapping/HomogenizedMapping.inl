/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_HomogenizedMAPPING_INL
#define SOFA_COMPONENT_MAPPING_HomogenizedMAPPING_INL

#include <sofa/component/mapping/HomogenizedMapping.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <string>
#include <iostream>


using std::cerr;
using std::endl;


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;




template <class BasicMapping>
void HomogenizedMapping<BasicMapping>::init()
{
    if(_alreadyInit) return;
    _alreadyInit=true;


    _sparseGrid = dynamic_cast<SparseGridTopologyT*> (this->fromModel->getContext()->getTopology());
    if(!_sparseGrid)
    {
        cerr<<"HomogenizedMapping can only be used with a SparseGridTopology\n";
        exit(0);
    }


    this->fromModel->getContext()->get(_forcefield);
    if(!_forcefield)
    {
        cerr<<"HomogenizedMapping can only be used with a HomogenizedHexahedronFEMForceFieldAndMass\n";
        exit(0);
    }


    _finestSparseGrid = _sparseGrid->_virtualFinerLevels[_sparseGrid->getNbVirtualFinerLevels()-_forcefield->_nbVirtualFinerLevels.getValue()];



    // in order to take into account initial translation
    InCoord translation0 = (*this->fromModel->getX())[0] - _sparseGrid->getPointPos(0);


    for(unsigned i=0; i<this->toModel->getX()->size(); ++i)
    {
        _p0.push_back( (*this->toModel->getX())[i] );
        (*this->toModel->getX())[i] += translation0;
    }

    for(unsigned i=0; i<this->fromModel->getX()->size(); ++i) // par construction de la sparse grid, pas de rotation initiale
        _qCoarse0.push_back( (*this->fromModel->getX())[i] - translation0 );






    _finestBarycentricCoord.resize(_p0.size());

    _rotations.resize( _sparseGrid->getNbHexas() );


    for (unsigned int i=0; i<_p0.size(); i++)
    {
        Vec3d coefs;
        int elementIdx = _sparseGrid->findCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
        if (elementIdx==-1)
        {
            elementIdx = _sparseGrid->findNearestCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
        }


        if (elementIdx!=-1)
        {
// 			_pointsCorrespondingToElem[elementIdx].push_back( i );


            // find barycentric coordinate in the finest element
            elementIdx = _finestSparseGrid->findCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
            if (elementIdx==-1)
            {
                elementIdx = _finestSparseGrid->findNearestCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
            }

            if( elementIdx!=-1)
            {
                helper::fixed_array<Real,8> baryCoefs;
                baryCoefs[0] = (1-coefs[0]) * (1-coefs[1]) * (1-coefs[2]);
                baryCoefs[1] = (coefs[0]) * (1-coefs[1]) * (1-coefs[2]);
                baryCoefs[2] = (coefs[0]) * (coefs[1]) * (1-coefs[2]);
                baryCoefs[3] = (1-coefs[0]) * (coefs[1]) * (1-coefs[2]);
                baryCoefs[4] = (1-coefs[0]) * (1-coefs[1]) * (coefs[2]);
                baryCoefs[5] = (coefs[0]) * (1-coefs[1]) * (coefs[2]);
                baryCoefs[6] = (coefs[0]) * (coefs[1]) * (coefs[2]);
                baryCoefs[7] = (1-coefs[0]) * (coefs[1]) * (coefs[2]);

                _finestBarycentricCoord[i] = std::pair<int,helper::fixed_array<Real,8> >(elementIdx, baryCoefs);
            }
            else
                cerr<<"HomogenizedMapping::init()   error finding the corresponding finest cube of vertex "<<_p0[i]<<endl;
        }
        else
            cerr<<"HomogenizedMapping::init()   error finding the corresponding coarse cube of vertex "<<_p0[i]<<endl;
    }

    cerr<<"HomogenizedMapping::init() before \n";



    _finestWeights.resize(_finestSparseGrid->getNbPoints());
    helper::vector<Real> sumCoefs(_finestSparseGrid->getNbPoints());

    for (unsigned int i=0; i<_forcefield->_finalWeights.size(); i++)
    {
        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexa(i);
// 		const SparseGridTopologyT::Hexa& coarsehexa = _sparseGrid->getHexa(_forcefield->_finalWeights[i].first);

// 		if( _finestSparseGrid->getType(i) != SparseGridTopologyT::BOUNDARY ) continue; // optimisation : regarde un element fin que si boundary == contient un triangle

        for(int w=0; w<8; ++w)
        {
            int fineNode = finehexa[w];

            helper::fixed_array<InCoord,8> coefs;
            for(int v=0; v<8; ++v)
            {
                for(int t=0; t<3; ++t)
                    coefs[v][t] = _forcefield->_finalWeights[i].second[ w*3+t ][ v*3+t ];
            }

            _finestWeights[ fineNode ].push_back( std::pair<int, helper::fixed_array<InCoord ,8> >(_forcefield->_finalWeights[i].first,coefs) );
            ++sumCoefs[fineNode];
        }

    }

    for (unsigned int i=0; i<_finestWeights.size(); i++)
    {
        for (unsigned int j=0; j<_finestWeights[i].size(); j++)
        {
            for( int k=0; k<8; ++k )
                _finestWeights[i][j].second[k] /= sumCoefs[i];
        }
    }
}



template <class BasicMapping>
void HomogenizedMapping<BasicMapping>::apply ( OutVecCoord& out, const InVecCoord& in )
{
    for(int i=0; i<_sparseGrid->getNbHexas(); ++i)
    {
        _rotations[i] = &_forcefield->getRotation(i);
    }

    InVecCoord coarseDisplacements(in.size() );
    for(unsigned i=0; i<in.size(); ++i)
    {
        coarseDisplacements[i] = in[i] - _qCoarse0[i]; // WARNING: ok for positions but not for velocities (can cause problem is initial velocities are not null or maybe when using
    }

    applyJ( out, coarseDisplacements );

    for(unsigned i=0; i<out.size(); ++i)
    {
        out[i] += _p0[i];
    }
}


template <class BasicMapping>
void HomogenizedMapping<BasicMapping>::applyJ ( OutVecDeriv& out, const InVecDeriv& in )
{
    // les deplacements des noeuds grossiers
    helper::vector< helper::fixed_array< InCoord, 8 >  > coarseDisplacements( _sparseGrid->getNbHexas() );
    for(int i=0; i<_sparseGrid->getNbHexas(); ++i)
    {
        const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexa(i);

        for(int w=0; w<8; ++w)
        {
            coarseDisplacements[i][w] =  (*_rotations[i]) * in[ hexa[w] ];
        }
    }


    // les déplacements des noeuds fins
    helper::vector< OutCoord > fineDisplacements( _finestWeights.size() );




    for (unsigned int i=0; i<_finestWeights.size(); i++) // fine nodes
    {
        for (unsigned int j=0; j<_finestWeights[i].size(); j++) // coarse elem
        {
            int coarseElem = _finestWeights[i][j].first;

            const Transformation& rotation = (*_rotations[ coarseElem ]);

            InCoord Wuc;
            for( int k=0; k<8; ++k ) // coarse nodes
            {
                for( int t=0; t<3; ++t ) // dimensions
                {
                    Wuc[t] += _finestWeights[i][j].second[k][t] * coarseDisplacements[ coarseElem ][k][t];
                }
            }

            fineDisplacements[i] += rotation.multTranspose( Wuc );
        }
    }


    // les déplacements des points mappés
    for(unsigned i=0; i<_p0.size(); ++i)
    {
        out[i] = OutCoord();


        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexa( _finestBarycentricCoord[i].first );

        for(int w=0; w<8; ++w)
        {
            out[i] += (fineDisplacements[ finehexa[w] ]  * _finestBarycentricCoord[i].second[w] );
        }
    }
}


template <class BasicMapping>
void HomogenizedMapping<BasicMapping>::applyJT ( InVecDeriv& out, const OutVecDeriv& in )
{
    // les forces des noeuds fins
    helper::vector< InDeriv > fineForces( _finestWeights.size() );


    for(unsigned i=0; i<_p0.size(); ++i)
    {
        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexa( _finestBarycentricCoord[i].first );

        for(int w=0; w<8; ++w)
        {
            fineForces[ finehexa[w] ] += in[i] *  _finestBarycentricCoord[i].second[w];
        }
    }


    // les forces des noeuds grossier
    for(unsigned i=0; i<fineForces.size(); ++i)
    {
        for( unsigned j=0; j<_finestWeights[i].size(); ++j) // coarse elem
        {
            int coarseElem = _finestWeights[i][j].first;
            const SparseGridTopologyT::Hexa& coarsehexa = _sparseGrid->getHexa( coarseElem );

            const Transformation& rotation = (*_rotations[ coarseElem ]);

            helper::fixed_array< InCoord, 8 > df;
            for(int w=0; w<8; ++w)
            {
                df[ w ] += rotation * fineForces[i];

                for(int t=0; t<3; ++t)
                {
                    df[ w ][t] *=  _finestWeights[i][j].second[w][t];
                }

                out[ coarsehexa[ w ] ] += rotation.multTranspose( df[ w ] );
            }
        }
    }
}



template <class BasicMapping>
void HomogenizedMapping<BasicMapping>::draw()
{
    if (!this->getShow()) return;
    glDisable (GL_LIGHTING);
// 	glPointSize(2);
// 		glColor4f (.2,1,0,1);

// 	const typename Out::VecCoord& out = *this->toModel->getX();
// 	glBegin (GL_POINTS);


// 	for(unsigned int i=0;i<_pointsCorrespondingToElem.size();i++)
// 	{
// 		for(unsigned j=0;j<_pointsCorrespondingToElem[i].size();++j)
// 		{
// 			for(int w=0;w<8;++w)
// 			{
// 				helper::gl::glVertexT(out[_pointsCorrespondingToElem[i][j]]);
// 			}
// 		}
// 	}

// 	for(unsigned int i=0;i<_finePos.size();i++)
// 	{
// 		helper::gl::glVertexT(_finePos[i]);
// 	}
//
//
//
// 	glEnd();
// 	glPointSize(1);

// 	const typename In::VecCoord& in = *this->fromModel->getX();
//
// 	const SparseGridTopologyT::SeqHexas& cubes = this->_sparseGrid->getHexas();
// 	glBegin (GL_LINES);
// 	{
// 		for(unsigned int i=0;i<_pointsCorrespondingToElem.size();i++)
// 		{
// 			for(unsigned j=0;j<_pointsCorrespondingToElem[i].size();++j)
// 			{
// 				for(int w=0;w<8;++w)
// 				{
// 					helper::gl::glVertexT(out[_pointsCorrespondingToElem[i][j]]);
// 					helper::gl::glVertexT(in[cubes[i][w]]);
// 				}
// 			}
// 		}
// 	}
// 	glEnd();


}


// template <class BasicMapping>
// typename HomogenizedMapping<BasicMapping>::InCoord HomogenizedMapping<BasicMapping>::computeTranslation( const SparseGridTopologyT::Hexa& hexa, unsigned idx )
// {
// 	InCoord bary;
// 	for(int j=0;j<8;++j)
// 		bary += _sparseGrid->getPointPos( hexa[j] );
// 	return (bary / 8.0) - _baycenters0[idx];
// }





} // namespace mapping

} // namespace component

} // namespace sofa

#endif
