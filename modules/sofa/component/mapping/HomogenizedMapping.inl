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




    for(unsigned i=0; i<this->toModel->getX()->size(); ++i)
        _p0.push_back( (*this->toModel->getX())[i] );

    for(unsigned i=0; i<this->fromModel->getX()->size(); ++i) // par construction de la sparse grid, pas de rotation initiale
        _qCoarse0.push_back( (*this->fromModel->getX())[i] );

    InCoord translation0 = (*this->fromModel->getX())[0] - _sparseGrid->getPointPos(0);

    for(int i=0; i<_finestSparseGrid->getNbPoints(); ++i)
        _qFine0.push_back( _finestSparseGrid->getPointPos(i)+translation0 );



    cerr<<_qCoarse0[0]<<endl;

// 	for(int i=0;i<_qFine0.size();++i)
// 		cerr<<i<<" : "<<_qFine0[i]<<endl;


// 	_pointsCorrespondingToElem.resize(_sparseGrid->getNbHexas());



// 	_baycenters0.resize(_sparseGrid->getNbHexas());
// 	for(int i=0;i<_sparseGrid->getNbHexas();++i)
// 	{
// 		const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexa(i);
// 		for(int j=0;j<8;++j)
// 			_baycenters0[i] += _sparseGrid->getPointPos( hexa[j] );
// 		_baycenters0[i] /= 8.0;
// 	}




    _finestBarycentricCoord.resize(_p0.size());
    _finestWeights.resize(_finestSparseGrid->getNbPoints());

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
// 	cerr<<_forcefield->_finalWeights.size()<<" "<<_finestSparseGrid->getNbHexas()<<endl;

    helper::vector<bool> deja(_finestSparseGrid->getNbPoints(),false);


    for (unsigned int i=0; i<_forcefield->_finalWeights.size(); i++)
    {
        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexa(i);

// 		if( _finestSparseGrid->getType(i) != SparseGridTopologyT::BOUNDARY ) continue; // optimisation : regarde un element fin que si boundary == contient un triangle

        for(int w=0; w<8; ++w)
        {
            Weight W;
            W[0] = _forcefield->_finalWeights[i].second[ w*3   ];
            W[1] = _forcefield->_finalWeights[i].second[ w*3+1 ];
            W[2] = _forcefield->_finalWeights[i].second[ w*3+2 ];


// 			if( deja[ finehexa[w] ])
// 				if( (W[0] - _finestWeights[ finehexa[w] ][0].second[0]).norm() > 1.0e-3 ) cerr<<"hum "<<(W[0] - _finestWeights[ finehexa[w] ][0].second[0]).norm()<<"\n";
// 			else cerr<<"tout pareil\n";


// 			if( !deja[ finehexa[w] ])
            {
                _finestWeights[ finehexa[w] ].push_back( std::pair<int,Weight>( _forcefield->_finalWeights[i].first ,  W  ) );

// 			if( finehexa[w]==5 ) cerr<<finehexa[w]<<" "<<_forcefield->_finalWeights[i].first<<" "<<i<<" "<<w<<endl;
// 			if( finehexa[w]==5 ) cerr<<finehexa[w]<<" "<<W<<endl;
// 			if( finehexa[w]==23 ) cerr<<finehexa[w]<<" "<<W<<endl;

// 			if( i==0 && w==3 ) cerr<<finehexa[w]<<endl;
// 			if( i==1 && w==2 ) cerr<<finehexa[w]<<endl;

                deja[ finehexa[w] ] = true;
            }
        }
    }

// 	cerr<<_finestWeights[17].size( )<<endl;
// 	for(int i=0;i<_finestWeights[17].size( );++i)
// 		cerr<<_finestWeights[17][i].second<<"\n";



// 	for (unsigned int i=0;i<_forcefield->_finalWeights.size();i++)
// 	{
// 		const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexa(i);
// 		cerr<<_finestWeights[ finehexa[i] ].size()<<endl;
// 		for(int w=0;w<_finestWeights[ finehexa[w] ].size();++w)
// 		{
// 			cerr<< _finestWeights[ finehexa[w] ][w].first<<endl;
// 			cerr<< _finestWeights[ finehexa[w] ][w].second<<endl;
// 			cerr<<"-------\n";
// 		}
// 		cerr<<"****************\n";
// 	}


    // non necessary memory optimisation
// 	_forcefield->_finalWeights.resize(0);

}



template <class BasicMapping>
void HomogenizedMapping<BasicMapping>::apply ( OutVecCoord& out, const InVecCoord& in )
{
    // les deplacements des noeuds grossiers
    helper::vector< Vec< 24 >  > coarseDisplacements( _sparseGrid->getNbHexas() );
    for(int i=0; i<_sparseGrid->getNbHexas(); ++i)
    {
        const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexa(i);
// 		InCoord translation = computeTranslation( hexa, i );

// 		const Transformation& rotation = _forcefield->getRotation(i);
// 		_rotations[i].fromMatrix( rotation );

        _rotations[i] = _forcefield->getRotation(i);


        for(int w=0; w<8; ++w)
        {
            InCoord u = _rotations[i] * in[ hexa[w] ] /*-translation*/ - _qCoarse0[hexa[w]];

            coarseDisplacements[i][w*3  ] = u[0];
            coarseDisplacements[i][w*3+1] = u[1];
            coarseDisplacements[i][w*3+2] = u[2];
        }
    }

// 	cerr<<"coarseDisplacements: "<<coarseDisplacements<<endl;





    // les déplacements des noeuds fins
    helper::vector< OutCoord > fineDisplacements( _finestWeights.size() );
// 	helper::vector< Transformation > meanRotations( _finestWeights.size() );

    for(unsigned i=0; i<_finestWeights.size(); ++i)
    {

// 		helper::Quater<Real> meanRotation;
        for( unsigned j=0; j<_finestWeights[i].size(); ++j)
        {
// 			meanRotation += _rotations[ _finestWeights[i][j].first ];
            Transformation& rotation = _rotations[ _finestWeights[i][j].first ];

// 			cerr<<rotation<<endl;

            fineDisplacements[i] += rotation.multTranspose( _qFine0[i] + _finestWeights[i][j].second * coarseDisplacements[ _finestWeights[i][j].first ] );
        }
// 		meanRotation /= _finestWeights[i].size(); meanRotation.toMatrix( meanRotations[i] );
        fineDisplacements[i] /= _finestWeights[i].size();

// 		fineDisplacements[i] += _qFine0[i];
    }


// 	cerr<<in<<" ==> "<<coarseDisplacements<<" ==> "<<fineDisplacements[5]<<endl;
    cerr<<fineDisplacements[5]<<endl;
    cerr<<fineDisplacements[23]<<endl;

// 	cerr<<"fineDisplacements: "<<fineDisplacements<<endl;
// 	_finePos.resize( fineDisplacements.size());
// 	_finePos=fineDisplacements;






    // les déplacements des points mappés
    for(unsigned i=0; i<_p0.size(); ++i)
    {
        out[i] = OutCoord(); //_p0[i] /*+ translation*/;


        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexa( _finestBarycentricCoord[i].first );

        for(int w=0; w<8; ++w)
        {
            out[i] += /*meanRotations[ finehexa[w] ].multTranspose*/(fineDisplacements[ finehexa[w] ]  * _finestBarycentricCoord[i].second[w] );
        }

    }






// 	// les déplacements des noeuds fins décomposés par element grossier
// 	helper::vector< helper::vector< OutCoord > > fineDisplacements( _finestWeights.size() );
// 	for(unsigned i=0;i<_finestWeights.size();++i)
// 	{
// 		fineDisplacements[i].resize(_finestWeights[i].size());
//
//
// 		for( unsigned j=0;j<_finestWeights[i].size();++j)
// 		{
// 			const Transformation& rotation = _forcefield->getRotation(_finestWeights[i][j].first);
// 			fineDisplacements[i][j] = (/*_qFine0[i] +*/ rotation.multTranspose( _finestWeights[i][j].second * coarseDisplacements[ _finestWeights[i][j].first ] ) ) / _finestWeights[i].size();
// 		}
// 	}
//
//
// 	// les déplacements des points mappés
// 	for(unsigned i=0;i<_p0.size();++i)
// 	{
// 		//out[i] = OutCoord();
// 		out[i] = _p0[i] /*+ translation*/;
//
// 		const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexa( _finestBarycentricCoord[i].first );
//
// 		for(int w=0;w<8;++w)
// 		{
// 			for( unsigned j=0;j<fineDisplacements[finehexa[w]].size();++j)
// 			{
// 				out[i] += fineDisplacements[ finehexa[w] ][j]  * _finestBarycentricCoord[i].second[w];
// 			}
// 		}
//
// 	}
}


template <class BasicMapping>
void HomogenizedMapping<BasicMapping>::applyJ ( OutVecDeriv& out, const InVecDeriv& in )
{
    // les deplacements des noeuds grossiers
    helper::vector< Vec< 24 >  > coarseDisplacements( _sparseGrid->getNbHexas() );
    for(int i=0; i<_sparseGrid->getNbHexas(); ++i)
    {
        const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexa(i);

        for(int w=0; w<8; ++w)
        {
            InCoord u = _rotations[i] * in[ hexa[w] ];

            coarseDisplacements[i][w*3  ] = u[0];
            coarseDisplacements[i][w*3+1] = u[1];
            coarseDisplacements[i][w*3+2] = u[2];
        }
    }


    // les déplacements des noeuds fins
    helper::vector< OutCoord > fineDisplacements( _finestWeights.size() );

    for(unsigned i=0; i<_finestWeights.size(); ++i)
    {
        for( unsigned j=0; j<_finestWeights[i].size(); ++j)
        {
            Transformation& rotation = _rotations[ _finestWeights[i][j].first ];

            fineDisplacements[i] += rotation.multTranspose( _finestWeights[i][j].second * coarseDisplacements[ _finestWeights[i][j].first ] );
        }
        fineDisplacements[i] /= _finestWeights[i].size();
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

        for( unsigned j=0; j<_finestWeights[i].size(); ++j)
        {
            Transformation& rotation = _rotations[ _finestWeights[i][j].first];

            Vec< 24 > dfplat = _finestWeights[i][j].second.multTranspose( rotation * fineForces[i] ) / _finestWeights[i].size();

            const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexa( _finestWeights[i][j].first );
            for(int w=0; w<8; ++w)
            {
                out[ hexa[ w ] ] += rotation.multTranspose( InCoord( dfplat[w*3],dfplat[w*3+1],dfplat[w*3+2]));
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
