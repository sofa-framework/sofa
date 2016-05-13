/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMEdgeBasedMapping_INL
#define SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMEdgeBasedMapping_INL

#include <sofa/component/mapping/HexahedronCompositeFEMEdgeBasedMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/behavior/MechanicalMapping.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <string>
#include <iostream>



namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class BasicMapping>
const int HexahedronCompositeFEMEdgeBasedMapping<BasicMapping>::EDGES[12][3]=
{
    {0,1,0},
    {3,2,0},
    {4,5,0},
    {7,6,0},
    {0,3,1},
    {1,2,1},
    {4,7,1},
    {5,6,1},
    {0,4,2},
    {1,5,2},
    {3,7,2},
    {2,6,2}
};




template <class BasicMapping>
void HexahedronCompositeFEMEdgeBasedMapping<BasicMapping>::init()
{
    if(this->_alreadyInit) return;

    Inherit::init();



    for(int i=0; i<this->_finestSparseGrid->getNbPoints(); ++i)
        this->_qFine0[i] = this->_finestSparseGrid->getPointPos(i);



    _coarseBarycentricCoord.resize( this->_qFine0.size());
    for (unsigned int i=0; i<this->_qFine0.size(); i++)
    {
        Vector3 coefs;
        int elementIdx = this->_sparseGrid->findCube( this->_qFine0[i] , coefs[0], coefs[1], coefs[2] );
        if (elementIdx==-1)
        {
            elementIdx = this->_sparseGrid->findNearestCube( this->_qFine0[i] , coefs[0], coefs[1], coefs[2] );
// 			cerr<<"a cote : " <<elementIdx<<"     "<<coefs<<sendl;
// 			cerr<<i<<sendl;
        }



        if( elementIdx!=-1)
        {
            const topology::SparseGridTopology::Hexa& coarsehexa = this->_sparseGrid->getHexahedron( elementIdx );
            _coarseBarycentricCoord[i][coarsehexa[0]] = (Real)((1-coefs[0]) * (1-coefs[1]) * (1-coefs[2]));
            _coarseBarycentricCoord[i][coarsehexa[1]] = (Real)((coefs[0]) * (1-coefs[1]) * (1-coefs[2]));
            _coarseBarycentricCoord[i][coarsehexa[2]] = (Real)((coefs[0]) * (coefs[1]) * (1-coefs[2]));
            _coarseBarycentricCoord[i][coarsehexa[3]] = (Real)((1-coefs[0]) * (coefs[1]) * (1-coefs[2]));
            _coarseBarycentricCoord[i][coarsehexa[4]] = (Real)((1-coefs[0]) * (1-coefs[1]) * (coefs[2]));
            _coarseBarycentricCoord[i][coarsehexa[5]] = (Real)((coefs[0]) * (1-coefs[1]) * (coefs[2]));
            _coarseBarycentricCoord[i][coarsehexa[6]] = (Real)((coefs[0]) * (coefs[1]) * (coefs[2]));
            _coarseBarycentricCoord[i][coarsehexa[7]] = (Real)((1-coefs[0]) * (coefs[1]) * (coefs[2]));
        }
        else
            cerr<<"HexahedronCompositeFEMEdgeBasedMapping::init()   error finding the corresponding coarest cube of vertex "<<this->_qFine0[i]<<sendl;
    }



    _weightsEdge.resize( this->_finestWeights.size());
    int nbedges=1;
    std::map<Edge,int > edgesmap;
    for(unsigned i=0; i<this->_finestWeights.size(); ++i)
    {
        for(typename std::map< int, Weight >::iterator it = this->_finestWeights[i].begin(); it!=this->_finestWeights[i].end(); ++it)
        {
            const topology::SparseGridTopology::Hexa& coarsehexa = this->_sparseGrid->getHexahedron( it->first );

            for(int j=0; j<12; ++j) //edges
            {
                float& coef0 = (*it).second[ EDGES[j][2] ][ EDGES[j][0]*3+EDGES[j][2]];
                float& coef1 = (*it).second[ EDGES[j][2] ][ EDGES[j][1]*3+EDGES[j][2]];

                Real& barycoef0 = _coarseBarycentricCoord[i][ coarsehexa[ EDGES[j][0] ] ];
                Real& barycoef1 = _coarseBarycentricCoord[i][ coarsehexa[ EDGES[j][1] ] ];


                if( /*fabs( coef0 )>1.0e-5 && fabs( coef1 )>1.0e-5 &&*/ barycoef0>0 && barycoef1>0)
                {


                    Real coef = coef0/(coef0+coef1);
                    Real barycoef = barycoef0/(barycoef0+barycoef1);

// 					Real edgecoef = 1-2*(coef0-barycoef0)/(coef0+coef1);
                    Real edgecoef = barycoef-coef;



// 					cerr<<coef0<<" "<<coef1<<" "<<barycoef0<<sendl;

// 					cerr<<edgecoef<<sendl;


                    if( fabs(edgecoef) > 1.0e-5 )
                    {

                        Edge e( coarsehexa[ EDGES[j][0] ], coarsehexa[ EDGES[j][1]], coarsehexa[ EDGES[j][2]] );
                        if(!edgesmap[ e ] )
                        {
                            edgesmap[ e ]=nbedges;
                            ++nbedges;
                            _edges.push_back( e );
                        }

                        _weightsEdge[ i ][ edgesmap[ e ]-1 ] += edgecoef;
                    }

                }
            }
        }
    }

    _size0[0] = (InReal)(this->_sparseGrid->_regularGrid.getDx()[0]);
    _size0[1] = (InReal)(this->_sparseGrid->_regularGrid.getDy()[1]);
    _size0[2] = (InReal)(this->_sparseGrid->_regularGrid.getDz()[2]);
}



template <class BasicMapping>
void HexahedronCompositeFEMEdgeBasedMapping<BasicMapping>::apply ( OutVecCoord& out, const InVecCoord& in )
{
    for(int i=0; i<this->_sparseGrid->getNbHexahedra(); ++i)
        this->_rotations[i] = this->_forcefield->getRotation(i);


    helper::vector<Real> elongations(_edges.size());
    helper::vector<InCoord> directions(_edges.size());
    for(unsigned i=0; i<_edges.size(); ++i)
    {
        InCoord e = in[ _edges[i][1] ] - in[ _edges[i][0] ];
        Real n = (Real)e.norm();
        if( n==0.0 ) serr<<"HexahedronCompositeFEMEdgeBasedMapping apply div 0"<<sendl;
        elongations[i] = (Real)((n -_size0[_edges[i][2]])/2.0f);
        directions[i] = e/n;
    }


    for(unsigned i=0; i<_weightsEdge.size(); ++i)
    {
// 		cerr<<i<<":"<<sendl;
        this->_qFine[i] = InCoord(); // TODO interpola baryc
        for( typename std::map< int, Real >::iterator it=_coarseBarycentricCoord[i].begin(); it!=_coarseBarycentricCoord[i].end(); ++it)
        {
            this->_qFine[i] += in[it->first]*it->second;
        }

        for( typename std::map<int,Real>::iterator it = _weightsEdge[i].begin(); it!=_weightsEdge[i].end(); ++it)
        {
            this->_qFine[i] += directions[it->first]*(elongations[ it->first ] * it->second);
// 			cerr<<it->first<<" "<< it->second<<"     "<<elongations[ it->first ]<<sendl;
        }
// 		cerr<<sendl;

    }


    // les d�placements des points mapp�s
    for(unsigned i=0; i<this->_p0.size(); ++i)
    {
        out[i] = OutCoord();


        const topology::SparseGridTopology::Hexa& finehexa = this->_finestSparseGrid->getHexahedron(this->_finestBarycentricCoord[i].first );

        for(int w=0; w<8; ++w)
        {
            out[i] += this->_qFine[ finehexa[w] ]  * this->_finestBarycentricCoord[i].second[w];
        }

    }


}




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
