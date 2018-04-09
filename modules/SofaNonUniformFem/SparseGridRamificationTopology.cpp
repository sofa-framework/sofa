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

#include <sofa/core/ObjectFactory.h>

#include "SparseGridRamificationTopology.h"

// #include <sofa/helper/proximity.h>

namespace sofa
{

namespace component
{

namespace topology
{



SOFA_DECL_CLASS(SparseGridRamificationTopology)

int SparseGridRamificationTopologyClass = core::RegisterObject("Sparse grid in 3D (modified)")
        .addAlias("SparseGridRamification")
        .add< SparseGridRamificationTopology >()
        ;

SparseGridRamificationTopology::SparseGridRamificationTopology(bool isVirtual)
    : SparseGridTopology(isVirtual)
    , _finestConnectivity( initData(&_finestConnectivity,true,"finestConnectivity","Test for connectivity at the finest level? (more precise but slower by testing all intersections between the model mesh and the faces between boundary cubes)"))
{
}

SparseGridRamificationTopology::~SparseGridRamificationTopology()
{
    for( unsigned i=0; i<_connexions.size(); ++i)
        for( unsigned j=0; j<_connexions[i].size(); ++j)
            delete _connexions[i][j];
}


void SparseGridRamificationTopology::init()
{
    SparseGridTopology::init();

    if( this->isVirtual || _nbVirtualFinerLevels.getValue() > 0)
        findCoarsestParents(); // in order to compute findCube by beginning by the finnest, by going up and give the coarsest parent
}

void SparseGridRamificationTopology::buildAsFinest()
{
// 				serr<<"SparseGridRamificationTopology::buildAsFinest()"<<sendl;

    SparseGridTopology::buildAsFinest();


    if( _finestConnectivity.getValue() || this->isVirtual || _nbVirtualFinerLevels.getValue() > 0 )
    {
        // find the connexion graph between the finest hexahedra
        findConnexionsAtFinestLevel();
    }

    if( _finestConnectivity.getValue() )
    {
        buildRamifiedFinestLevel();
    }
}


void SparseGridRamificationTopology::findConnexionsAtFinestLevel()
{

    _connexions.resize( getNbHexahedra() );
    for( unsigned i=0; i<_connexions.size(); ++i)
        _connexions[i].push_back( new Connexion() ); // at the finest level, each hexa corresponds exatly to one connexion

    helper::io::Mesh* mesh = NULL;

    // Finest level is asked
    if (_finestConnectivity.getValue())
    {
        std::string filename = this->fileTopology.getValue();
        if (!filename.empty()) // file given, try to load it.
        {
            mesh = helper::io::Mesh::Create(filename.c_str());
            if (!mesh)
            {
                serr<<"Warning: SparseGridRamificationTopology::findConnexionsAtFinestLevel Can't create mesh from file=\""<< fileTopology.getValue()<<"\" is valid)"<<sendl;
                return;
            }
        }
        if(filename.empty() && vertices.getValue().empty()) // No vertices buffer set, nor mesh file => impossible to create mesh
        {
            serr<<"Warning: SparseGridRamificationTopology::findConnexionsAtFinestLevel -- mesh is NULL (check if fileTopology=\""<< fileTopology.getValue()<<"\" is valid)"<<sendl;
            return;
        }
        else if(filename.empty() && !vertices.getValue().empty()) // no file given but vertex buffer. We can rebuild the mesh
        {
            mesh = new helper::io::Mesh();
            for (unsigned int i = 0; i<vertices.getValue().size(); ++i)
                mesh->getVertices().push_back(vertices.getValue()[i]);
            const helper::vector < helper::vector <int> >& facets = this->facets.getValue();
            const SeqTriangles& triangles = this->input_triangles.getValue();
            const SeqQuads& quads = this->input_quads.getValue();
            mesh->getFacets().resize(facets.size() + triangles.size() + quads.size());
            for (unsigned int i = 0; i<facets.size(); ++i)
                mesh->getFacets()[i].push_back(facets[i]);
            for (unsigned int i0 = facets.size(), i = 0; i<triangles.size(); ++i)
            {
                mesh->getFacets()[i0 + i].resize(1);
                mesh->getFacets()[i0 + i][0].resize(3);
                mesh->getFacets()[i0 + i][0][0] = triangles[i][0];
                mesh->getFacets()[i0 + i][0][1] = triangles[i][1];
                mesh->getFacets()[i0 + i][0][2] = triangles[i][2];
            }
            for (unsigned int i0 = facets.size() + triangles.size(), i = 0; i<quads.size(); ++i)
            {
                mesh->getFacets()[i0 + i].resize(1);
                mesh->getFacets()[i0 + i][0].resize(4);
                mesh->getFacets()[i0 + i][0][0] = quads[i][0];
                mesh->getFacets()[i0 + i][0][1] = quads[i][1];
                mesh->getFacets()[i0 + i][0][2] = quads[i][2];
                mesh->getFacets()[i0 + i][0][3] = quads[i][3];
            }
        }

    }

    // loop on every cubes
    for(int z=0; z<getNz()-1; ++z)
        for(int y=0; y<getNy()-1; ++y)
            for(int x=0; x<getNx()-1; ++x)
            {
                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                if(cubeIdx != -1) // if existing in SG (ie not outside)
                {

                    Connexion* actualConnexion = _connexions[cubeIdx][0];

                    // find all neighbors in 3 directions

                    if(x>0)
                    {
                        // left neighbor
                        const int neighborIdxRG = _regularGrid->cube(x-1,y,z);
                        const int neighborIdx = _indicesOfRegularCubeInSparseGrid[neighborIdxRG];

                        if(_indicesOfRegularCubeInSparseGrid[neighborIdxRG] != -1 && sharingTriangle( mesh, cubeIdx, neighborIdx, LEFT ))
                        {
                            Connexion* neighbor = _connexions[neighborIdx][0];
                            actualConnexion->_neighbors[LEFT].insert(neighbor);
                            neighbor->_neighbors[RIGHT].insert(actualConnexion);
                        }
                    }
                    if(y>0)
                    {
                        // lower neighbor
                        const int neighborIdxRG = _regularGrid->cube(x,y-1,z);
                        const int neighborIdx = _indicesOfRegularCubeInSparseGrid[neighborIdxRG];

                        if(_indicesOfRegularCubeInSparseGrid[neighborIdxRG] != -1 && sharingTriangle( mesh, cubeIdx, neighborIdx, DOWN ))
                        {
                            Connexion* neighbor = _connexions[neighborIdx][0];
                            actualConnexion->_neighbors[DOWN].insert(neighbor);
                            neighbor->_neighbors[UP].insert(actualConnexion);
                        }

                    }
                    if(z>0)
                    {
                        // back neighbor
                        const int neighborIdxRG = _regularGrid->cube(x,y,z-1);
                        const int neighborIdx = _indicesOfRegularCubeInSparseGrid[neighborIdxRG];

                        if(_indicesOfRegularCubeInSparseGrid[neighborIdxRG] != -1 && sharingTriangle( mesh, cubeIdx, neighborIdx, BEFORE ))
                        {
                            Connexion* neighbor = _connexions[neighborIdx][0];
                            actualConnexion->_neighbors[BEFORE].insert(neighbor);
                            neighbor->_neighbors[BEHIND].insert(actualConnexion);
                        }
                    }
                }
            }





    // HACK: delete the connexions with no-neighbours
// 				for(unsigned i=0 ; i<_connexions.size();++i)
// 				{
// 					int nb=0;
// 					for(int j=0;j<NUM_CONNECTED_NODES;++j)
// 						nb += _connexions[i][0]->_neighbors[j].size();
// 					if( nb == 0 )
// 						_connexions[i].resize(0);
// 				}
    // end HACK


    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        if( !_connexions[i].empty() )
        {
            _connexions[i][0]->_hexaIdx = i;
            _connexions[i][0]->_nonRamifiedHexaIdx = i;
            _mapHexa_Connexion[ i ] = std::pair<helper::vector<Connexion*>,int>(_connexions[i],0);
        }
    }


    delete mesh;


// 				printNeighborhood();




// 				serr<<"END SparseGridRamificationTopology::buildAsFinest()"<<sendl;

}


bool SparseGridRamificationTopology::sharingTriangle(helper::io::Mesh* mesh, int cubeIdx, int neighborIdx, unsigned where )
{
    if(!_finestConnectivity.getValue() || mesh==NULL )
        return true;

    // it is not necessary to analyse connectivity between non-boundary cells
    if( getType(cubeIdx)!=BOUNDARY || getType(neighborIdx)!=BOUNDARY)
        return true;


// 				serr<<"SparseGridRamificationTopology::sharingTriangle "<<this->fileTopology.getValue()<<""<<sendl;

    const Hexa& hexa=getHexahedron( cubeIdx );

    Vector3 a,b,c,d;//P,Q;

    //trouver la face commune
    switch(where)
    {
    case LEFT:
        a = getPointPos( hexa[0] );
        b = getPointPos( hexa[4] );
        c = getPointPos( hexa[7] );
        d = getPointPos( hexa[3] );
        break;
    case DOWN:
        a = getPointPos( hexa[0] );
        b = getPointPos( hexa[1] );
        c = getPointPos( hexa[5] );
        d = getPointPos( hexa[4] );
        break;
    case BEFORE:
        a = getPointPos( hexa[0] );
        b = getPointPos( hexa[1] );
        c = getPointPos( hexa[2] );
        d = getPointPos( hexa[3] );
        break;
    }


    const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();
    const helper::vector<Vector3>& vertices = mesh->getVertices();
    for (unsigned int f=0; f<facets.size(); f++)
    {
        const helper::vector<int>& facet = facets[f][0];

        for (unsigned int j=1; j<facet.size(); j++) // Triangularize
        {
            const Vector3& A = vertices[facet[j-1]];
            const Vector3& B = vertices[facet[j]];

            //tester si la face commune intersecte facet
            // pour les 3 aretes de facet avec le carre
// 						static helper::DistanceSegTri proximitySolver;
// 						proximitySolver.NewComputation( a,b,c, A, B,P,Q);
// 						if( (Q-P).norm2() < 1.0e-6 ) return true;
// 						proximitySolver.NewComputation( a,c,d, A, B,P,Q);
// 						if( (Q-P).norm2() < 1.0e-6 ) return true;

            if (intersectionSegmentTriangle( A,B,a,b,c) ) return true;
            if( intersectionSegmentTriangle( A,B,a,c,d) ) return true;

        }
    }

    return false;
}


void SparseGridRamificationTopology::buildRamifiedFinestLevel()
{

    SeqHexahedra& hexahedra = *seqHexahedra.beginEdit();


    helper::vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    for(unsigned i=0; i<hexahedra.size(); ++i)
    {
        CubeCorners c;
        for(int w=0; w<8; ++w)
            c[w]=getPointPos( hexahedra[i][w] );
        cubeCorners.push_back( c );
    }



    hexahedra.clear();

    nbPoints = 0;

    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
        {
            hexahedra.push_back( Hexa(nbPoints, nbPoints+1, nbPoints+2, nbPoints+3, nbPoints+4, nbPoints+5, nbPoints+6, nbPoints+7) );
            (*it)->_hexaIdx = hexahedra.size()-1;
            (*it)->_nonRamifiedHexaIdx = i;
            nbPoints += 8;
        }
    }


    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( unsigned j=0; j<_connexions[i].size(); ++j)
        {
            _mapHexa_Connexion[ _connexions[i][j]->_hexaIdx ] = std::pair<helper::vector<Connexion*>,int>(_connexions[i],j);
        }
    }



    // which cube is neigbor of which cube? in order to link similar vertices (link entiere faces)
    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
        {

            Hexa& hexa = hexahedra[ (*it)->_hexaIdx ]; // the hexa corresponding to the connexion


            for(std::set<Connexion*>::iterator neig = (*it)->_neighbors[BEFORE].begin();
                neig != (*it)->_neighbors[BEFORE].end(); ++neig)
            {
                Hexa& neighbor = hexahedra[ (*neig)->_hexaIdx ]; // the hexa corresponding to the neighbor connexion

                changeIndices( hexa[0], neighbor[4] );
                changeIndices( hexa[1], neighbor[5] );
                changeIndices( hexa[2], neighbor[6] );
                changeIndices( hexa[3], neighbor[7] );
            }

            for(std::set<Connexion*>::iterator neig = (*it)->_neighbors[DOWN].begin();
                neig != (*it)->_neighbors[DOWN].end(); ++neig)
            {
                Hexa& neighbor = hexahedra[ (*neig)->_hexaIdx ]; // the hexa corresponding to the neighbor connexion

                changeIndices( hexa[0], neighbor[3] );
                changeIndices( hexa[4], neighbor[7] );
                changeIndices( hexa[5], neighbor[6] );
                changeIndices( hexa[1], neighbor[2] );
            }

            for(std::set<Connexion*>::iterator neig = (*it)->_neighbors[LEFT].begin();
                neig != (*it)->_neighbors[LEFT].end(); ++neig)
            {
                Hexa& neighbor = hexahedra[ (*neig)->_hexaIdx ]; // the hexa corresponding to the neighbor connexion

                changeIndices( hexa[0], neighbor[1] );
                changeIndices( hexa[4], neighbor[5] );
                changeIndices( hexa[7], neighbor[6] );
                changeIndices( hexa[3], neighbor[2] );
            }

        }
    }

    // saving incident hexahedra for each points in order to be able to link or not vertices
    helper::vector< helper::vector< helper::fixed_array<unsigned,3> > > hexahedraConnectedToThePoint(nbPoints);
    unsigned c=0;
    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
        {
            for(unsigned p=0; p<8; ++p)
            {
                hexahedraConnectedToThePoint[hexahedra[c][p]].push_back( helper::fixed_array<unsigned,3>  (c, p, i) );
            }
            c++;
        }
    }

    helper::vector<defaulttype::Vec<3,SReal> >& seqPoints = *this->seqPoints.beginEdit(); seqPoints.clear();
    nbPoints=0;
    for(unsigned i=0; i<hexahedraConnectedToThePoint.size(); ++i)
    {
        if( !hexahedraConnectedToThePoint[i].empty() )
        {
            for(unsigned j=0; j<hexahedraConnectedToThePoint[i].size(); ++j)
            {
                hexahedra[ hexahedraConnectedToThePoint[i][j][0] ][ hexahedraConnectedToThePoint[i][j][1] ] = nbPoints;
            }


            // find corresponding 3D coordinate
            seqPoints.push_back( cubeCorners[ hexahedraConnectedToThePoint[i][0][2] ][ hexahedraConnectedToThePoint[i][0][1] ] );

            ++nbPoints;
        }
    }

    this->seqPoints.endEdit();
    seqHexahedra.endEdit();

}



void SparseGridRamificationTopology::buildFromFiner()
{
// 				serr<<"SparseGridRamificationTopology::buildFromFiner()"<<sendl;

// 				SparseGridTopology::buildFromFiner();	return;


    SparseGridRamificationTopology* finerSparseGridRamification = dynamic_cast<SparseGridRamificationTopology*>(_finerSparseGrid);



    setNx( _finerSparseGrid->getNx()/2+1 );
    setNy( _finerSparseGrid->getNy()/2+1 );
    setNz( _finerSparseGrid->getNz()/2+1 );


    _regularGrid->setSize(getNx(),getNy(),getNz());

    setMin(_finerSparseGrid->getMin());
    setMax(_finerSparseGrid->getMax());


    // the cube size of the coarser mesh is twice the cube size of the finer mesh
    // if the finer mesh contains an odd number of cubes in any direction,
    // the coarser mesh will be a half cube size larger in that direction
    Vector3 dx = _finerSparseGrid->_regularGrid->getDx();
    Vector3 dy = _finerSparseGrid->_regularGrid->getDy();
    Vector3 dz = _finerSparseGrid->_regularGrid->getDz();
    setXmax(getXmin() + (getNx()-1) * (SReal)2.0 * dx[0]);
    setYmax(getYmin() + (getNy()-1) * (SReal)2.0 * dy[1]);
    setZmax(getZmin() + (getNz()-1) * (SReal)2.0 * dz[2]);




    _regularGrid->setPos(getXmin(), getXmax(), getYmin(), getYmax(), getZmin(), getZmax());


    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid->getNbHexahedra(), -1 ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid



    helper::vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values


    HierarchicalCubeMap nonRamifiedHierarchicalCubeMap;
    sofa::helper::vector<Type> nonRamifiedTypes;

    // as classical SparseGridTopology, build the hierarchy and deduce who is inside or boundary
    for(int i=0; i<getNx()-1; i++)
        for(int j=0; j<getNy()-1; j++)
            for(int k=0; k<getNz()-1; k++)
            {
                int x = 2*i;
                int y = 2*j;
                int z = 2*k;

                helper::fixed_array<int,8> fineIndices;
                for(int idx=0; idx<8; ++idx)
                {
                    const int idxX = x + (idx & 1);
                    const int idxY = y + (idx & 2)/2;
                    const int idxZ = z + (idx & 4)/4;
                    if(idxX < _finerSparseGrid->getNx()-1 && idxY < _finerSparseGrid->getNy()-1 && idxZ < _finerSparseGrid->getNz()-1)
                        fineIndices[idx] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid->cube(idxX,idxY,idxZ) ];
                    else
                        fineIndices[idx] = -1;
                }

                bool inside = true;
                bool outside = true;
                for( int w=0; w<8 && (inside || outside); ++w)
                {
                    if( fineIndices[w] == -1 ) inside=false;
                    else
                    {

                        if( _finerSparseGrid->getType( fineIndices[w] ) == BOUNDARY ) { inside=false; outside=false; }
                        else if( _finerSparseGrid->getType( fineIndices[w] ) == INSIDE ) {outside=false;}
                    }
                }

                if(outside) continue;
                if( inside )
                {
                    nonRamifiedTypes.push_back(INSIDE);
                }
                else
                {
                    nonRamifiedTypes.push_back(BOUNDARY);
                }

                int coarseRegularIndice = _regularGrid->cube( i,j,k );
                Hexa c = _regularGrid->getHexaCopy( coarseRegularIndice );

                CubeCorners corners;
                for(int w=0; w<8; ++w)
                {
                    corners[w] = _regularGrid->getPoint( c[w] );
                    cubeCornerPositionIndiceMap[corners[w]] = 0;
                }

                cubeCorners.push_back(corners);

                _indicesOfRegularCubeInSparseGrid[coarseRegularIndice] = cubeCorners.size()-1;
                _indicesOfCubeinRegularGrid.push_back( coarseRegularIndice );

                nonRamifiedHierarchicalCubeMap.push_back( fineIndices );
            }




    // deduce the number of independant connexions inside each coarse hexahedra depending on the fine connexion graph

    _connexions.resize( cubeCorners.size() );

    // find all fine connexions included in every coarse hexa
    for(unsigned idx=0; idx<_connexions.size(); ++idx) // for all the coarse hexa
    {
        helper::vector< Connexion* > allFineConnexions; // to save all fine connexions included in the coarse hexa
        helper::vector< unsigned > allFineConnexionsPlace;


        helper::fixed_array<int,8>& children = nonRamifiedHierarchicalCubeMap[idx]; // the child hexa

        for( int child = 0; child < 8 ; child ++)
        {
            int childIdx = children[child];

            if( childIdx != -1 )
            {
                helper::vector<Connexion*> & childConnexions = finerSparseGridRamification->_connexions[ childIdx ]; // all connexions of the child hexa

                for(helper::vector<Connexion*>::iterator fineConnexion = childConnexions.begin() ; fineConnexion != childConnexions.end() ; ++fineConnexion)
                {
                    allFineConnexionsPlace.push_back( child );
                    allFineConnexions.push_back( *fineConnexion );
                    (*fineConnexion)->_tmp = -1;
                }
            }
        }



        int connexionNumber=0;
        // who is neighbor between allFineConnexions of a coarse hexa
        for( unsigned i=0; i<allFineConnexions.size(); ++i )
        {
            allFineConnexions[i]->propagateConnexionNumberToNeighbors( connexionNumber, allFineConnexions ); // give the same connexion number to all the neighborood
            ++connexionNumber;
        }


        // keep only unique ConnexionNumber
        std::map<int,int> uniqueConnexionNumber;
        for(unsigned i=0; i<allFineConnexions.size(); ++i )
        {
            uniqueConnexionNumber[ allFineConnexions[i]->_tmp]++;
        }

        // for each ConnexionNumber, build a new coarse connexion
        for( std::map<int,int>::iterator it=uniqueConnexionNumber.begin(); it!=uniqueConnexionNumber.end(); ++it)
        {
            Connexion* newConnexion = new Connexion();
            _connexions[idx].push_back(newConnexion);

            for(unsigned i=0; i<allFineConnexions.size(); ++i )
            {
                if( allFineConnexions[i]->_tmp == (*it).first )
                {
                    newConnexion->_children.push_back( Connexion::Children( allFineConnexionsPlace[i],allFineConnexions[i]) );
                    allFineConnexions[i]->_parent = newConnexion;
                }
            }
        }
    }


    // print debug
// 				for(int z=0; z<getNz()-1; ++z)
// 				{
// 					for(int y=0; y<getNy()-1; ++y)
// 					{
// 						for(int x=0; x<getNx()-1; ++x)
// 						{
// 							const int cubeIdxRG = _regularGrid->cube(x,y,z);
// 							const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];
//
// 							if( cubeIdx==-1)
// 								sout << "  ";
// 							else
// 							{
// 								sout << _connexions[cubeIdx].size() << " ";
// 							}
// 						}
// 						sout << sendl;
// 					}
// 					sout << " -- " << sendl;
// 				}



    // to know in what cube is a connexion
    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
        {
            (*it)->_tmp = i;
        }
    }


    // build connectivity graph between new Connexions by looking into children neighborood


    for(int z=0; z<_finerSparseGrid->getNz()-1; ++z)
    {
        for(int y=0; y<_finerSparseGrid->getNy()-1; ++y)
        {
            for(int x=0; x<_finerSparseGrid->getNx()-1; ++x)
            {
                int fineIdx = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid->cube(x,y,z) ];
                if( fineIdx != -1 )
                {

                    for( helper::vector<Connexion*>::iterator it = finerSparseGridRamification->_connexions[fineIdx].begin(); it != finerSparseGridRamification->_connexions[fineIdx].end() ; ++it)
                    {
                        Connexion* fineConnexion1 = *it;
                        Connexion* coarseConnexion1 = fineConnexion1->_parent;
                        int coarseHexa1 = coarseConnexion1->_tmp;

                        for( unsigned i=0; i<NUM_CONNECTED_NODES; ++i) // in all directions
                        {
                            for(std::set<Connexion*>::iterator n = fineConnexion1->_neighbors[i].begin(); n!=fineConnexion1->_neighbors[i].end(); ++n)
                            {

                                Connexion* fineConnexion2 = *n;
                                Connexion* coarseConnexion2 = fineConnexion2->_parent;
                                int coarseHexa2 = coarseConnexion2->_tmp;

// 											serr<<"coarseHexa : "<<coarseHexa1<<" "<<coarseHexa2<<sendl;

// 											if( coarseConnexion1 != coarseConnexion2 )
                                if( coarseHexa1 != coarseHexa2 ) // the both fine hexahedra are not in the same coarse hexa
                                {
                                    // 											sout<<"coarseConnexion1 : "<<coarseConnexion1<<"   "<<(coarseConnexion1==NULL?"NULL":"not")<<sendl;
                                    coarseConnexion1->_neighbors[i].insert( coarseConnexion2 );
                                    coarseConnexion2->_neighbors[ !(i%2)?i+1:i-1 ].insert( coarseConnexion1 );
                                }
                            }
                        }
                    }
                }
            }
        }
    }



// 				printNeighborhood();





    // build an new coarse hexa for each independnnt connexion

    SeqHexahedra& hexahedra = *seqHexahedra.beginEdit();
    hexahedra.clear();

    nbPoints = 0;

    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
        {
            hexahedra.push_back( Hexa(nbPoints, nbPoints+1, nbPoints+2, nbPoints+3, nbPoints+4, nbPoints+5, nbPoints+6, nbPoints+7) );
            (*it)->_hexaIdx = hexahedra.size()-1;
            (*it)->_nonRamifiedHexaIdx = i;
            nbPoints += 8;
        }
    }



    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( unsigned j=0; j<_connexions[i].size(); ++j)
        {
            _mapHexa_Connexion[ _connexions[i][j]->_hexaIdx ] = std::pair<helper::vector<Connexion*>,int>(_connexions[i],j);
        }
    }




    // which cube is neigbor of which cube? in order to link similar vertices (link entiere faces)
    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
        {

            Hexa& hexa = hexahedra[ (*it)->_hexaIdx ]; // the hexa corresponding to the connexion


            for(std::set<Connexion*>::iterator neig = (*it)->_neighbors[BEFORE].begin();
                neig != (*it)->_neighbors[BEFORE].end(); ++neig)
            {
                Hexa& neighbor = hexahedra[ (*neig)->_hexaIdx ]; // the hexa corresponding to the neighbor connexion

                changeIndices( hexa[0], neighbor[4] );
                changeIndices( hexa[1], neighbor[5] );
                changeIndices( hexa[2], neighbor[6] );
                changeIndices( hexa[3], neighbor[7] );
            }

            for(std::set<Connexion*>::iterator neig = (*it)->_neighbors[DOWN].begin();
                neig != (*it)->_neighbors[DOWN].end(); ++neig)
            {
                Hexa& neighbor = hexahedra[ (*neig)->_hexaIdx ]; // the hexa corresponding to the neighbor connexion

                changeIndices( hexa[0], neighbor[3] );
                changeIndices( hexa[4], neighbor[7] );
                changeIndices( hexa[5], neighbor[6] );
                changeIndices( hexa[1], neighbor[2] );
            }

            for(std::set<Connexion*>::iterator neig = (*it)->_neighbors[LEFT].begin();
                neig != (*it)->_neighbors[LEFT].end(); ++neig)
            {
                Hexa& neighbor = hexahedra[ (*neig)->_hexaIdx ]; // the hexa corresponding to the neighbor connexion

                changeIndices( hexa[0], neighbor[1] );
                changeIndices( hexa[4], neighbor[5] );
                changeIndices( hexa[7], neighbor[6] );
                changeIndices( hexa[3], neighbor[2] );
            }

        }
    }




// 				serr<<hexahedra<<sendl;




    // saving incident hexahedra for each points in order to be able to link or not vertices
    helper::vector< helper::vector< helper::fixed_array<unsigned,3> > > hexahedraConnectedToThePoint(nbPoints);
    unsigned c=0;
    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
        {
            for(unsigned p=0; p<8; ++p)
            {
                hexahedraConnectedToThePoint[hexahedra[c][p]].push_back( helper::fixed_array<unsigned,3>  (c, p, i) );
            }
            c++;
        }
    }

// 				serr<<seqPoints.size()<<" "<<hexahedra.size()<<sendl;


    helper::vector<defaulttype::Vec<3,SReal> >& seqPoints = *this->seqPoints.beginEdit(); seqPoints.clear();
    nbPoints=0;
    for(unsigned i=0; i<hexahedraConnectedToThePoint.size(); ++i)
    {
        if( !hexahedraConnectedToThePoint[i].empty() )
        {
            for(unsigned j=0; j<hexahedraConnectedToThePoint[i].size(); ++j)
            {
                hexahedra[ hexahedraConnectedToThePoint[i][j][0] ][ hexahedraConnectedToThePoint[i][j][1] ] = nbPoints;
            }

            // find corresponding 3D coordinate
            seqPoints.push_back( cubeCorners[ hexahedraConnectedToThePoint[i][0][2] ][ hexahedraConnectedToThePoint[i][0][1] ] );



            ///DEBUG  FAAALLLSSSSEEEE
// 						if( hexahedraConnectedToThePoint[i].size() == 1 )
// 						{
// 							seqPoints.back()[2] -= hexahedraConnectedToThePoint[i][0][0]*3; // decale les particles dedoubl�es du nb de leur connexion pour bien les diff�rencier
// 						}



            ++nbPoints;
        }
    }


    this->seqPoints.endEdit();
    seqHexahedra.endEdit();


    for(unsigned i=0 ; i<_connexions.size(); ++i)
    {
        helper::fixed_array<int,8> nonRamifiedFineIndices = nonRamifiedHierarchicalCubeMap[ i ];

        if( _connexions[i].size()==1 ) // 1 seule connexion pour l'element ==> element normal non ramifi� ==> meme enfants
        {
            _hierarchicalCubeMap.push_back( nonRamifiedFineIndices );
            _types.push_back( nonRamifiedTypes[i] );
        }
        else // plrs connexion pour un element normal ==> trouver quels fils sont dans quelle connexion
        {

            for( helper::vector<Connexion*>::iterator it = _connexions[i].begin(); it != _connexions[i].end() ; ++it)
            {
                helper::fixed_array<int,8> fineIndices;

                for( std::list<Connexion::Children>::iterator child=(*it)->_children.begin(); child!=(*it)->_children.end(); ++child)
                {
                    unsigned childIdx=(*child).second->_nonRamifiedHexaIdx;
                    for(unsigned p=0; p<8; ++p)
                    {
                        if( childIdx == (unsigned)nonRamifiedFineIndices[p] )
                            fineIndices[p] = (*child).second->_hexaIdx;
                        else
                            fineIndices[p] = -1;
                    }
                }
                _hierarchicalCubeMap.push_back( fineIndices );
                _types.push_back( BOUNDARY );
            }
        }
    }



    _hierarchicalCubeMapRamification.resize(this->getNbHexahedra());
    for(int i=0; i<this->getNbHexahedra(); ++i)
    {
        Connexion*& coarsecon = _mapHexa_Connexion[ i ].first[ _mapHexa_Connexion[ i ].second ];

        for( std::list<Connexion::Children>::iterator child=coarsecon->_children.begin(); child!=coarsecon->_children.end(); ++child)
        {
            _hierarchicalCubeMapRamification[i][(*child).first].push_back( (*child).second->_hexaIdx );
        }
    }



// 				serr<<"_connexions : "<<sendl;
// 				for(unsigned i=0;i<_connexions.size();++i)
// 				{
// 					serr<<i<<" -> ";
// 					for(unsigned j=0;j<_connexions[i].size();++j)
// 					{
// 						serr<<"[ ";
// 						for( std::list<Connexion::Children>::iterator child=_connexions[i][j]->_children.begin();child!=_connexions[i][j]->_children.end();++child)
// 							serr<<(*child).second->_hexaIdx<<" ";
// 						serr<<"] , ";
// 					}
// 					serr<<sendl;
// 				}
//
// 				serr<<"_hierarchicalCubeMap : "<<sendl;
// 				for(unsigned i=0;i<_hierarchicalCubeMap.size();++i)
// 				{
// 					serr<<i<<" -> ";
// 					for(int w=0;w<8;++w)
// 					{
// 						serr<<_hierarchicalCubeMap[i][w]<<" ";
// 					}
// 					serr<<sendl;
// 				}
//
// 				serr<<"_mapHexa_Connexion : "<<sendl;
// 				for(unsigned i=0;i<_mapHexa_Connexion.size();++i)
// 				{
// 					serr<<i<<" -> ";
// 					serr<<"["<<_mapHexa_Connexion[i].first[0]->_nonRamifiedHexaIdx<<"]  ";
// 					serr<<_mapHexa_Connexion[i].second;
// 					serr<<sendl;
// 				}
//
//
// 				serr<<_finerSparseGrid->getNbHexahedra()<<sendl;
// 				serr<<"_hierarchicalCubeMapRamification : "<<sendl;
// 				for(unsigned i=0;i<_hierarchicalCubeMapRamification.size();++i)
// 				{
// 					serr<<i<<" -> ";
// 					for(int w=0;w<8;++w)
// 					{
// 						serr<<"[ ";
// 						for(unsigned j=0;j<_hierarchicalCubeMapRamification[i][w].size();++j)
// 						{
// 							serr<<_hierarchicalCubeMapRamification[i][w][j]<<" ";
// 						}
// 						serr<<"] , ";
// 					}
// 					serr<<sendl;
// 				}








    // compute stiffness coefficient from children
    _stiffnessCoefs.resize( this->getNbHexahedra() );
    _massCoefs.resize( this->getNbHexahedra() );
    for(int i=0; i<this->getNbHexahedra(); ++i)
    {
        helper::fixed_array<int,8> finerChildren = this->_hierarchicalCubeMap[i];
        unsigned nbchildren = 0;
        for(int w=0; w<8; ++w)
        {
            if( finerChildren[w] != -1 )
            {
                _massCoefs[i] += this->_finerSparseGrid->getMassCoef(finerChildren[w]);
                _stiffnessCoefs[i] += this->_finerSparseGrid->getStiffnessCoef(finerChildren[w]);
                ++nbchildren;
            }
        }
        _stiffnessCoefs[i] /= 8.0;//(float)nbchildren;
        _massCoefs[i] /= 8.0;//(float)nbchildren;
    }





// 				serr<<"nbPoints : "<<seqPoints.size()<<sendl;

// 				serr<<"END SparseGridRamificationTopology::buildFromFiner()"<<sendl;
}




void SparseGridRamificationTopology::buildVirtualFinerLevels()
{
    int nb = _nbVirtualFinerLevels.getValue();

    _virtualFinerLevels.resize(nb);

    int newnx=n.getValue()[0],newny=n.getValue()[1],newnz=n.getValue()[2];
    for( int i=0; i<nb; ++i)
    {
        newnx = (newnx-1)*2+1;
        newny = (newny-1)*2+1;
        newnz = (newnz-1)*2+1;
    }


    SparseGridRamificationTopology::SPtr sgrt = sofa::core::objectmodel::New< SparseGridRamificationTopology >(true);

    _virtualFinerLevels[0] = sgrt;
    _virtualFinerLevels[0]->setName("virtualLevel0");
    _virtualFinerLevels[0]->setNx( newnx );
    _virtualFinerLevels[0]->setNy( newny );
    _virtualFinerLevels[0]->setNz( newnz );
    this->addSlave(_virtualFinerLevels[0]); //->setContext( this->getContext() );
    sgrt->_finestConnectivity.setValue( _finestConnectivity.getValue() );
    _virtualFinerLevels[0]->_fillWeighted.setValue( _fillWeighted.getValue() );
    _virtualFinerLevels[0]->setMin( _min.getValue() );
    _virtualFinerLevels[0]->setMax( _max.getValue() );
    const std::string& fileTopology = this->fileTopology.getValue();
    if (fileTopology.empty()) // If no file is defined, try to build from the input Datas
    {
        _virtualFinerLevels[0]->vertices.setParent(&this->vertices);
        _virtualFinerLevels[0]->facets.setParent(&this->facets);
        _virtualFinerLevels[0]->input_triangles.setParent(&this->input_triangles);
        _virtualFinerLevels[0]->input_quads.setParent(&this->input_quads);
    }
    else
        _virtualFinerLevels[0]->load(fileTopology.c_str());
    _virtualFinerLevels[0]->init();

    sout<<"buildVirtualFinerLevels : ";
    sout<<"("<<newnx<<"x"<<newny<<"x"<<newnz<<") -> "<< _virtualFinerLevels[0]->getNbHexahedra() <<" elements , ";

    for(int i=1; i<nb; ++i)
    {
        _virtualFinerLevels[i] = sofa::core::objectmodel::New< SparseGridRamificationTopology >(true);
        std::ostringstream oname;
        oname << "virtualLevel" << i;
        _virtualFinerLevels[i]->setName(oname.str());
        this->addSlave(_virtualFinerLevels[i]);

        _virtualFinerLevels[i]->setFinerSparseGrid(_virtualFinerLevels[i-1].get());

        _virtualFinerLevels[i]->init();


        sout<<"("<<_virtualFinerLevels[i]->getNx()<<"x"<<_virtualFinerLevels[i]->getNy()<<"x"<<_virtualFinerLevels[i]->getNz()<<") -> "<< _virtualFinerLevels[i]->getNbHexahedra() <<" elements , ";
    }

    sout<<sendl;

    this->setFinerSparseGrid(_virtualFinerLevels[nb-1].get());
}



int SparseGridRamificationTopology::findCube(const Vector3 &pos, SReal &fx, SReal &fy, SReal &fz)
{
// 				if(  !this->isVirtual && _nbVirtualFinerLevels.getValue() == 0 )
// 					return SparseGridTopology::findCube(pos, fx, fy, fz);

    if(  _nbVirtualFinerLevels.getValue() == 0 )
        return SparseGridTopology::findCube(pos, fx, fy, fz);


    SparseGridRamificationTopology* finestSparseGridTopology = dynamic_cast<SparseGridRamificationTopology*>(_virtualFinerLevels[0].get());

    int finestSparseCube = finestSparseGridTopology->SparseGridTopology::findCube(pos,fx,fy,fz);

    if( finestSparseCube == -1 ) return -1;

    Connexion * finestConnexion = finestSparseGridTopology->_connexions[ finestSparseCube ][0];


    _regularGrid->findCube( pos,fx,fy,fz); // only to compute fx,fy,fz

    return finestConnexion->_coarsestParent;

}

int SparseGridRamificationTopology::findNearestCube(const Vector3 &pos, SReal &fx, SReal &fy, SReal &fz)
{
// 				if( !this->isVirtual && _nbVirtualFinerLevels.getValue() == 0 )
// 					return SparseGridTopology::findNearestCube(pos, fx, fy, fz);


    if( _nbVirtualFinerLevels.getValue() == 0 )
        return SparseGridTopology::findNearestCube(pos, fx, fy, fz);


    SparseGridRamificationTopology* finestSparseGridTopology = dynamic_cast<SparseGridRamificationTopology*>(_virtualFinerLevels[0].get());

    int finestSparseCube = finestSparseGridTopology->SparseGridTopology::findNearestCube(pos,fx,fy,fz);

    if( finestSparseCube == -1 ) return -1;

    Connexion * finestConnexion = finestSparseGridTopology->_connexions[ finestSparseCube ][0];


    _regularGrid->findNearestCube( pos,fx,fy,fz); // only to compute fx,fy,fz

    return finestConnexion->_coarsestParent;
}





void SparseGridRamificationTopology::findCoarsestParents()
{
    for( unsigned i=0; i<_virtualFinerLevels.size(); ++i)
    {
        SparseGridRamificationTopology* finestSGRT = dynamic_cast<SparseGridRamificationTopology*>(_virtualFinerLevels[i].get());


        for(int z=0; z<finestSGRT->getNz()-1; ++z)
        {
            for(int y=0; y<finestSGRT->getNy()-1; ++y)
            {
                for(int x=0; x<finestSGRT->getNx()-1; ++x)
                {
                    const int cubeIdxRG = finestSGRT->_regularGrid->cube(x,y,z);
                    const int cubeIdx = finestSGRT->_indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                    if( cubeIdx!=-1)
                    {
                        for( helper::vector<Connexion*>::iterator it = finestSGRT->_connexions[cubeIdx].begin(); it != finestSGRT->_connexions[cubeIdx].end() ; ++it)
                        {
                            Connexion * finestConnexion = *it;
                            while( finestConnexion->_parent != NULL )
                                finestConnexion = finestConnexion->_parent;

                            (*it)->_coarsestParent = finestConnexion->_hexaIdx;
                        }
                    }
                }
            }
        }
    }

    for(int z=0; z<getNz()-1; ++z)
    {
        for(int y=0; y<getNy()-1; ++y)
        {
            for(int x=0; x<getNx()-1; ++x)
            {
                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                if( cubeIdx!=-1)
                {
                    for( helper::vector<Connexion*>::iterator it = _connexions[cubeIdx].begin(); it != _connexions[cubeIdx].end() ; ++it)
                    {
                        (*it)->_coarsestParent = (*it)->_hexaIdx;
                    }
                }
            }
        }
    }


// 				printParents();

}



void SparseGridRamificationTopology::changeIndices(unsigned oldidx, unsigned newidx)
{
    SeqHexahedra& hexahedra = *seqHexahedra.beginEdit();
    for(unsigned i=0; i<hexahedra.size(); ++i)
        for(int j=0; j<8; ++j)
        {
            if( hexahedra[i][j] == oldidx )
            {
                hexahedra[i][j] = newidx;
                break; // very small optimization (if the point is found on a hexa, it is no in the follow points of the same hexa)
            }
        }
}






///////////////// DEBUG PRINTING /////////////////////////////////


void SparseGridRamificationTopology::printNeighborhood()
{
    // 				// print draw neighbors
    serr<<sendl<<sendl;
    for(int z=0; z<getNz()-1; ++z)
    {
        for(int y=getNy()-2; y>=0; --y)
        {


            for(int x=0; x<getNx()-1; ++x)
            {

                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                if( cubeIdx==-1)
                    sout << "     ";
                else
                {
                    for( helper::vector<Connexion*>::iterator it = _connexions[cubeIdx].begin(); it != _connexions[cubeIdx].end() ; ++it)
                    {
                        if( ! (*it)->_neighbors[UP].empty() ) sout<<"  | ";
                        else sout<<"   ";
                    }
                }
            }
            sout << sendl;



            for(int x=0; x<getNx()-1; ++x)
            {
                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];




                if( cubeIdx==-1)
                    sout << "     ";
                else
                {
                    sout<<"[";
                    for( helper::vector<Connexion*>::iterator it = _connexions[cubeIdx].begin(); it != _connexions[cubeIdx].end() ; ++it)
                    {
                        if( it!= _connexions[cubeIdx].begin() ) sout<<",";

                        if( ! (*it)->_neighbors[LEFT].empty() ) sout<<"-";
                        else sout<<" ";

                        if( ! (*it)->_neighbors[BEFORE].empty() ) sout<<"X";
                        else sout<<"0";
                    }
                    sout<<"]";
                }



            }
            sout << sendl;




        }
        sout << " -- " << sendl;
    }
}

void SparseGridRamificationTopology::printNeighbors()
{
    // print nb neighbors per cube
    for(int z=0; z<getNz()-1; ++z)
    {
        for(int y=getNy()-2; y>=0; --y)
        {
            for(int x=0; x<getNx()-1; ++x)
            {
                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                if( cubeIdx==-1)
                    sout << "  ";
                else
                {
                    int sum=0;
                    for(int i=0; i<NUM_CONNECTED_NODES; ++i)
                        sum+=_connexions[cubeIdx][0]->_neighbors[i].size();
                    sout << sum << " ";
                }
            }
            sout << sendl;
        }
        sout << " -- " << sendl;
    }
}

void SparseGridRamificationTopology::printNbConnexions()
{
    // print nb connexions
    for(int z=0; z<getNz()-1; ++z)
    {
        for(int y=getNy()-2; y>=0; --y)
        {
            for(int x=0; x<getNx()-1; ++x)
            {
                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                if( cubeIdx==-1)
                    sout << "  ";
                else
                {
                    sout << _connexions[cubeIdx].size() << " ";
                }
            }
            sout << sendl;
        }
        sout << " -- " << sendl;
    }
}


void SparseGridRamificationTopology::printParents()
{
    serr<<"\n\nPARENTS\n"<<sendl;

    for( unsigned i=0; i<_virtualFinerLevels.size(); ++i)
    {
        serr<<"level "<<i<<" :"<<sendl;

        SparseGridRamificationTopology* finestSGRT = dynamic_cast<SparseGridRamificationTopology*>(_virtualFinerLevels[i].get());

        for(int z=0; z<finestSGRT->getNz()-1; ++z)
        {
            for(int y=finestSGRT->getNy()-2; y>=0; --y)
            {
                for(int x=0; x<finestSGRT->getNx()-1; ++x)
                {
                    const int cubeIdxRG = finestSGRT->_regularGrid->cube(x,y,z);
                    const int cubeIdx = finestSGRT->_indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                    if( cubeIdx==-1)
                        sout << " ";
                    else
                    {
                        sout<<"[";
                        for( helper::vector<Connexion*>::iterator it = finestSGRT->_connexions[cubeIdx].begin(); it != finestSGRT->_connexions[cubeIdx].end() ; ++it)
                        {
                            if( it!= finestSGRT->_connexions[cubeIdx].begin() ) sout<<",";

                            sout << cubeIdx << "->"<<(*it)->_coarsestParent;
                        }
                        sout<<"]";
                        sout<<" ";
                    }
                }
                sout << sendl;
            }
            sout << " -- " << sendl;
        }
    }

    for(int z=0; z<getNz()-1; ++z)
    {
        for(int y=getNy()-2; y>=0; --y)
        {
            for(int x=0; x<getNx()-1; ++x)
            {
                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                if( cubeIdx==-1)
                    sout << " ";
                else
                {
                    sout<<"[";
                    for( helper::vector<Connexion*>::iterator it = _connexions[cubeIdx].begin(); it != _connexions[cubeIdx].end() ; ++it)
                    {
                        if( it!= _connexions[cubeIdx].begin() ) sout<<",";
                        sout << (*it)->_coarsestParent;
                    }
                    sout<<"]";
                    sout<<" ";
                }
            }
            sout << sendl;
        }
        sout << " -- " << sendl;
    }
}


void SparseGridRamificationTopology::printHexaIdx()
{
    // print hexa nb
    for(int z=0; z<getNz()-1; ++z)
    {
        for(int y=getNy()-2; y>=0; --y)
        {
            for(int x=0; x<getNx()-1; ++x)
            {
                const int cubeIdxRG = _regularGrid->cube(x,y,z);
                const int cubeIdx = _indicesOfRegularCubeInSparseGrid[cubeIdxRG];

                if( cubeIdx==-1)
                    sout << "  ";
                else
                {
                    sout<<"[";
                    for( helper::vector<Connexion*>::iterator it = _connexions[cubeIdx].begin(); it != _connexions[cubeIdx].end() ; ++it)
                    {
                        if( it!= _connexions[cubeIdx].begin() ) sout<<",";
                        sout << (*it)->_hexaIdx;
                    }
                    sout<<"]";
                    sout<<" ";
                }
            }
            sout << sendl;
        }
        sout << " -- " << sendl;
    }
}




bool SparseGridRamificationTopology::intersectionSegmentTriangle(Vector3 s0, Vector3 s1, Vector3 t0, Vector3 t1, Vector3 t2)
{
    // compute the offset origin, edges, and normal
    Vector3 kDiff = s0 - t0;
    Vector3 kEdge1 = t1 - t0;
    Vector3 kEdge2 = t2 - t0;
    Vector3 kNormal = kEdge1.cross(kEdge2);

    Vector3 dir = s1-s0;
    SReal norm = (s1-s0).norm();
    dir /= norm;

    // Solve Q + t*D = b1*E1 + b2*E2 (Q = kDiff, D = segment direction,
    // E1 = kEdge1, E2 = kEdge2, N = Cross(E1,E2)) by
    //   |Dot(D,N)|*b1 = sign(Dot(D,N))*Dot(D,Cross(Q,E2))
    //   |Dot(D,N)|*b2 = sign(Dot(D,N))*Dot(D,Cross(E1,Q))
    //   |Dot(D,N)|*t = -sign(Dot(D,N))*Dot(Q,N)
    SReal fDdN = dir * kNormal;
    SReal fSign;
    if (fDdN > 1.0e-10)
    {
        fSign = (SReal)1.0;
    }
    else if (fDdN < -1.0e-10)
    {
        fSign = (SReal)-1.0;
        fDdN = -fDdN;
    }
    else
    {
        // Segment and triangle are parallel, call it a "no intersection"
        // even if the segment does intersect.
        return false;
    }

    SReal fDdQxE2 = fSign * (dir * kDiff.cross(kEdge2));
    if (fDdQxE2 >= (SReal)0.0)
    {
        SReal fDdE1xQ = fSign* (dir * kEdge1.cross(kDiff));
        if (fDdE1xQ >= (SReal)0.0)
        {
            if (fDdQxE2 + fDdE1xQ <= fDdN)
            {
                // line intersects triangle, check if segment does
                SReal fQdN = -fSign*(kDiff*kNormal);
                SReal fExtDdN = norm*fDdN;
                if (-fExtDdN <= fQdN && fQdN <= fExtDdN)
                {
                    // segment intersects triangle
                    return true;
                }
                // else: |t| > extent, no intersection
            }
            // else: b1+b2 > 1, no intersection
        }
        // else: b2 < 0, no intersection
    }
    // else: b1 < 0, no intersection

    return false;
}

} // namespace topology

} // namespace component

} // namespace sofa
