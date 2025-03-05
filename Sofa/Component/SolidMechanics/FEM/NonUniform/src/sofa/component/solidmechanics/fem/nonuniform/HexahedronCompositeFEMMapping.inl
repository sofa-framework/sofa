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

#include <sofa/component/solidmechanics/fem/nonuniform/HexahedronCompositeFEMMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <string>
#include <iostream>


namespace sofa::component::solidmechanics::fem::nonuniform
{

template <class BasicMapping>
void HexahedronCompositeFEMMapping<BasicMapping>::init()
{
    using namespace sofa::defaulttype;

    if(_alreadyInit) return;
    _alreadyInit=true;


    _sparseGrid = dynamic_cast<SparseGridTopologyT*> (this->fromModel->getContext()->getTopology());
    if(!_sparseGrid)
    {
        msg_error() << "HexahedronCompositeFEMMapping can only be used with a SparseGridTopology";
        exit(EXIT_FAILURE);
    }


    this->fromModel->getContext()->get(_forcefield);
    if(!_forcefield)
    {
        msg_error() << "HexahedronCompositeFEMMapping can only be used with a HexahedronCompositeFEMForceFieldAndMass";
        exit(EXIT_FAILURE);
    }


    _finestSparseGrid = _sparseGrid->_virtualFinerLevels[_sparseGrid->getNbVirtualFinerLevels() -_forcefield->d_nbVirtualFinerLevels.getValue()];


    for(unsigned i=0; i<(unsigned)this->toModel->getSize(); ++i)
        _p0.push_back( this->toModel->read(core::vec_id::read_access::position)->getValue()[i] );

    for(unsigned i=0; i<(unsigned)this->fromModel->getSize(); ++i) // par construction de la sparse grid, pas de rotation initiale
        _qCoarse0.push_back( this->fromModel->read(core::vec_id::read_access::position)->getValue()[i] );

    InCoord translation0 = this->fromModel->read(core::vec_id::read_access::position)->getValue()[0] - _sparseGrid->getPointPos(0);

    for(Size i=0; i<_finestSparseGrid->getNbPoints(); ++i)
        _qFine0.push_back( _finestSparseGrid->getPointPos(i)+translation0 );

    _qFine = _qFine0;

// 	_pointsCorrespondingToElem.resize(_sparseGrid->getNbHexahedra());



// 	_baycenters0.resize(_sparseGrid->getNbHexahedra());
// 	for(int i=0;i<_sparseGrid->getNbHexahedra();++i)
// 	{
// 		const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexahedron(i);
// 		for(int j=0;j<8;++j)
// 			_baycenters0[i] += _sparseGrid->getPointPos( hexa[j] );
// 		_baycenters0[i] /= 8.0;
// 	}




    _finestBarycentricCoord.resize(_p0.size());
    _finestWeights.resize(_finestSparseGrid->getNbPoints());

    _rotations.resize( _sparseGrid->getNbHexahedra() );



    for (unsigned int i=0; i<_p0.size(); i++)
    {
        sofa::type::Vec3 coefs;
// 		int elementIdx = _sparseGrid->findCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
// 		if (elementIdx==-1)
// 		{
// 			elementIdx = _sparseGrid->findNearestCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
// 		}


// 		if (elementIdx!=-1)
        {
// 			_pointsCorrespondingToElem[elementIdx].push_back( i );


            // find barycentric coordinate in the finest element
            auto elementIdx = _finestSparseGrid->findCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
            if (elementIdx== sofa::InvalidID)
            {
                elementIdx = _finestSparseGrid->findNearestCube( _p0[i] , coefs[0], coefs[1], coefs[2] );
            }

            if (elementIdx != sofa::InvalidID)
            {
                type::fixed_array<Real, 8> baryCoefs;
                baryCoefs[0] = (Real)((1 - coefs[0]) * (1 - coefs[1]) * (1 - coefs[2]));
                baryCoefs[1] = (Real)((coefs[0]) * (1 - coefs[1]) * (1 - coefs[2]));
                baryCoefs[2] = (Real)((coefs[0]) * (coefs[1]) * (1 - coefs[2]));
                baryCoefs[3] = (Real)((1 - coefs[0]) * (coefs[1]) * (1 - coefs[2]));
                baryCoefs[4] = (Real)((1 - coefs[0]) * (1 - coefs[1]) * (coefs[2]));
                baryCoefs[5] = (Real)((coefs[0]) * (1 - coefs[1]) * (coefs[2]));
                baryCoefs[6] = (Real)((coefs[0]) * (coefs[1]) * (coefs[2]));
                baryCoefs[7] = (Real)((1 - coefs[0]) * (coefs[1]) * (coefs[2]));

                _finestBarycentricCoord[i] = std::pair<Index, type::fixed_array<Real, 8> >(elementIdx, baryCoefs);
            }
            else
                msg_error() << "HexahedronCompositeFEMMapping::init()   error finding the corresponding finest cube of vertex " << _p0[i];
        }
    }


    for (unsigned int i=0; i<_forcefield->_finalWeights.size(); i++)
    {
// 		if( _finestSparseGrid->getType(i) != SparseGridTopologyT::BOUNDARY ) continue; // optimisation : regarde un element fin que si boundary == contient un triangle


        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexahedron(i);

        for(int w=0; w<8; ++w)
        {
            Weight W;
            W[0] = _forcefield->_finalWeights[i].second[ w*3   ];
            W[1] = _forcefield->_finalWeights[i].second[ w*3+1 ];
            W[2] = _forcefield->_finalWeights[i].second[ w*3+2 ];

            _finestWeights[ finehexa[w] ][_forcefield->_finalWeights[i].first] =  W ;
        }
    }

    // non necessary memory optimisation
// 	_forcefield->_finalWeights.resize(0);



// 	for(unsigned i=0;i<_finestWeights.size();++i)
// 	{
// 		std::map< int, Weight >::iterator it = _finestWeights[i].begin();
//
// 		SparseGridTopologyT::Hexa& coarsehexa = _sparseGrid.getHexahedron( (*it).first );
//
//
// 		for(int j=0;j<8;++j)
// 			if( fabs( (*it).second[0][j*3] ) > 1.0e-5 || fabs( (*it).second[1][j*3+1] ) > 1.0e-5 || fabs( (*it).second[2][j*3+2] ) > 1.0e-5 )
//
//
// 	}


    Inherit::init();

}



template <class BasicMapping>
void HexahedronCompositeFEMMapping<BasicMapping>::apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& outData, const InDataVecCoord& inData)
{
    SOFA_UNUSED(mparams);

    OutVecCoord& out = *outData.beginEdit();
    const InVecCoord& in = inData.getValue();

    // les deplacements des noeuds grossiers
    type::vector< sofa::type::Vec< 24 >  > coarseDisplacements( _sparseGrid->getNbHexahedra() );
    for(sofa::Size i = 0; i<_sparseGrid->getNbHexahedra(); ++i)
    {
        const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexahedron(i);
// 		InCoord translation = computeTranslation( hexa, i );

// 		const Transformation& rotation = _forcefield->getRotation(i);
// 		_rotations[i].fromMatrix( rotation );

        _rotations[i] = _forcefield->getElementRotation(i);


        for(int w=0; w<8; ++w)
        {
            InCoord u = _rotations[i] * in[ hexa[w] ] /*-translation*/ - _qCoarse0[hexa[w]];

            coarseDisplacements[i][w*3  ] = (float)u[0];
            coarseDisplacements[i][w*3+1] = (float)u[1];
            coarseDisplacements[i][w*3+2] = (float)u[2];
        }
    }





    // les d�placements des noeuds fins
// 	type::vector< OutCoord > fineDisplacements( _finestWeights.size() );
// 	type::vector< Transformation > meanRotations( _finestWeights.size() );

    for(unsigned i=0; i<_finestWeights.size(); ++i)
    {
        _qFine[i] = InCoord();
// 		type::Quat<Real> meanRotation;
        for(std::map< int, Weight >::iterator it = _finestWeights[i].begin(); it!=_finestWeights[i].end(); ++it)
        {
// 			meanRotation += _rotations[ _finestWeights[i][j].first ];
            Transformation& rotation = _rotations[(*it).first ];

            _qFine[i] += rotation.multTranspose( _qFine0[i] + (*it).second * coarseDisplacements[ (*it).first ] );
        }
// 		meanRotation /= _finestWeights[i].size(); meanRotation.toMatrix( meanRotations[i] );
        _qFine[i] /= _finestWeights[i].size();

    }



    // les d�placements des points mapp�s
    for(unsigned i=0; i<_p0.size(); ++i)
    {
        out[i] = OutCoord(); //_p0[i] /*+ translation*/;


        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexahedron( _finestBarycentricCoord[i].first );

        for(int w=0; w<8; ++w)
        {
            out[i] += /*meanRotations[ finehexa[w] ].multTranspose*/(_qFine[ finehexa[w] ]  * _finestBarycentricCoord[i].second[w] );
        }

    }

    outData.endEdit();



}


template <class BasicMapping>
void HexahedronCompositeFEMMapping<BasicMapping>::applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& outData, const InDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    OutVecDeriv& out = *outData.beginEdit();
    const InVecDeriv& in = inData.getValue();

    // les deplacements des noeuds grossiers
    type::vector< sofa::type::Vec< 24 >  > coarseDisplacements( _sparseGrid->getNbHexahedra() );
    for(sofa::Size i=0; i<_sparseGrid->getNbHexahedra(); ++i)
    {
        const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexahedron(i);

        for(int w=0; w<8; ++w)
        {
            InCoord u = _rotations[i] * in[ hexa[w] ];

            coarseDisplacements[i][w*3  ] = (float)u[0];
            coarseDisplacements[i][w*3+1] = (float)u[1];
            coarseDisplacements[i][w*3+2] = (float)u[2];
        }
    }


    // les d�placements des noeuds fins
    type::vector< OutCoord > fineDisplacements( _finestWeights.size() );

    for(unsigned i=0; i<_finestWeights.size(); ++i)
    {
        for(std::map< int, Weight >::iterator it = _finestWeights[i].begin(); it!=_finestWeights[i].end(); ++it)
        {
            Transformation& rotation = _rotations[ (*it).first ];

            fineDisplacements[i] += rotation.multTranspose((*it).second * coarseDisplacements[ (*it).first ] );
        }
        fineDisplacements[i] /= _finestWeights[i].size();
    }


    // les d�placements des points mapp�s
    for(unsigned i=0; i<_p0.size(); ++i)
    {
        out[i] = OutCoord();


        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexahedron( _finestBarycentricCoord[i].first );

        for(int w=0; w<8; ++w)
        {
            out[i] += (fineDisplacements[ finehexa[w] ]  * _finestBarycentricCoord[i].second[w] );
        }
    }
    outData.endEdit();
}


template <class BasicMapping>
void HexahedronCompositeFEMMapping<BasicMapping>::applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& outData, const OutDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    InVecDeriv& out = *outData.beginEdit();
    const OutVecDeriv& in = inData.getValue();

    // les forces des noeuds fins
    type::vector< InDeriv > fineForces( _finestWeights.size() );


    for(unsigned i=0; i<_p0.size(); ++i)
    {
        const SparseGridTopologyT::Hexa& finehexa = _finestSparseGrid->getHexahedron( _finestBarycentricCoord[i].first );

        for(int w=0; w<8; ++w)
        {
            fineForces[ finehexa[w] ] += in[i] *  _finestBarycentricCoord[i].second[w];
        }
    }


    // les forces des noeuds grossier
    for(unsigned i=0; i<fineForces.size(); ++i)
    {

        for(std::map< int, Weight >::iterator it = _finestWeights[i].begin(); it!=_finestWeights[i].end(); ++it)
        {
            Transformation& rotation = _rotations[ (*it).first];

            sofa::type::Vec< 24 > dfplat = (*it).second.multTranspose( rotation * fineForces[i] ) / _finestWeights[i].size();

            const SparseGridTopologyT::Hexa& hexa = _sparseGrid->getHexahedron( (*it).first );
            for(int w=0; w<8; ++w)
            {
                out[ hexa[ w ] ] += rotation.multTranspose( InCoord( dfplat[w*3],dfplat[w*3+1],dfplat[w*3+2]));
            }
        }
    }

    outData.endEdit();
}



template <class BasicMapping>
void HexahedronCompositeFEMMapping<BasicMapping>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings()) return;

    std::vector< sofa::type::Vec3 > points;
    sofa::type::Vec3 point;

    for(unsigned int i=0; i<_qFine.size(); i++)
    {
        point = OutDataTypes::getCPos(_qFine[i]);
        points.push_back(point);
    }


    vparams->drawTool()->drawPoints(points, 7, sofa::type::RGBAColor(0.2f,1.0f,0.0f,1.0f));
}

} // namespace sofa::component::solidmechanics::fem::nonuniform
