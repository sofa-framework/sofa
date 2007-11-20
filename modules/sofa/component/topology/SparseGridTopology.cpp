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
#include <sofa/component/topology/SparseGridTopology.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/polygon_cube_intersection/polygon_cube_intersection.h>


using std::cerr;
using std::endl;
using std::pair;

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(SparseGridTopology)

int SparseGridTopologyClass = core::RegisterObject("Sparse grid in 3D")
        .addAlias("SparseGrid")
        .add< SparseGridTopology >()
        ;


// 	  const float SparseGridTopology::WEIGHT[8][8] =
// 	  {
// 		  { 1, .5, .5, .25,  .5,.25,.25, .125 }, // fine cube 0 from coarser corner 0 -> what weight for a vertex?
// 		  { .5,1,.25,.5,.25,.5,.125,.25 },
// 		  {.5,.25,1,.5,.25,.125,.5,.25},
// 		  {.25,.5,.5,1,.125,.25,.25,.5},
// 		  {.5,.25,.25,.125,1,.5,.5,.25},
// 		  {.25,.5,.125,.25,.5,1,.25,.5},
// 		  {.25,.125,.5,.25,.5,.25,1,.5},
// 		  {.125,.25,.25,.5,.25,.5,.5,1}
// 	  };


const float SparseGridTopology::WEIGHT27[8][27] =
{
    {1,0.5,0,0.5,0.25,0,0,0,0,0.5,0.25,0,0.25,0.125,0,0,0,0,0,0,0,0,0,0,0,0,0,}, // each weight of the jth fine vertex to the ith coarse vertex
    {0,0,0,0,0,0,0,0,0,0.5,0.25,0,0.25,0.125,0,0,0,0,1,0.5,0,0.5,0.25,0,0,0,0},
    {0,0,0,0.5,0.25,0,1,0.5,0,0,0,0,0.25,0.125,0,0.5,0.25,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0.25,0.125,0,0.5,0.25,0,0,0,0,0.5,0.25,0,1,0.5,0},
    {0,0.5,1,0,0.25,0.5,0,0,0,0,0.25,0.5,0,0.125,0.25,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0.25,0.5,0,0.125,0.25,0,0,0,0,0.5,1,0,0.25,0.5,0,0,0},
    {0,0,0,0,0.25,0.5,0,0.5,1,0,0,0,0,0.125,0.25,0,0.25,0.5,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0.125,0.25,0,0.25,0.5,0,0,0,0,0.25,0.5,0,0.5,1}
};

const int SparseGridTopology::cornerIndicesFromFineToCoarse[8][8]=
{
    { 0,9 ,3, 12 ,1 ,10 ,4 ,13}, // fine vertices forming the 0th coarse cube (with XYZ order)
    { 9, 18, 12 ,21, 10 ,19 ,13, 22},
    { 3, 12, 6 ,15 ,4, 13 ,7 ,16},
    { 12, 21, 15, 24, 13, 22 ,16, 25},
    { 1, 10, 4 ,13 ,2, 11 ,5, 14},
    { 10, 19, 13, 22, 11, 20, 14 ,23},
    { 4, 13, 7 ,16 ,5 ,14 ,8, 17},
    { 13, 22, 16, 25, 14, 23 ,17 ,26}

};


SparseGridTopology::SparseGridTopology(): nx(initData(&nx,0,"nx","x grid resolution")), ny(initData(&ny,0,"ny","y grid resolution")), nz(initData(&nz,0,"nz","z grid resolution")),
    xmin(initData(&xmin,0.0,"xmin","xmin grid")),ymin(initData(&ymin,0.0,"ymin","ymin grid")),zmin(initData(&zmin,0.0,"zmin","zmin grid")),
    xmax(initData(&xmax,0.0,"xmax","xmax grid")),ymax(initData(&ymax,0.0,"ymax","ymax grid")),zmax(initData(&zmax,0.0,"zmax","zmax grid"))
{
    _alreadyInit = false;
    _finerSparseGrid=NULL;
}



bool SparseGridTopology::load(const char* filename)
{
    this->filename.setValue( filename );
// 		cerr<<"SparseGridTopology::load : "<<filename<<"    "<<this->filename.getValue()<<endl;
    return true;
}







void SparseGridTopology::init()
{
    if(_alreadyInit) return;
    _alreadyInit = true;

    this->MeshTopology::init();
    invalidate();


    if( _finerSparseGrid != NULL )
        buildFromFiner();
    else
        buildAsFinest();

// 		  cerr<<"SparseGridTopology::init() :   "<<this->getName()<<"    cubes size = ";
    cerr<<seqCubes.size()<<"       ";
    cerr<<_types.size()<<endl;
}






void SparseGridTopology::buildAsFinest(  )
{
// 		  cerr<<"SparseGridTopology::buildAsFinest(  )\n";
    if (!filename.getValue().empty())
    {
//           						std::cout << "SparseGridTopology: using mesh "<<filename.getValue()<<std::endl;
        helper::io::Mesh* mesh = helper::io::Mesh::Create(filename.getValue().c_str());


        if (mesh)
        {
            const helper::vector<Vec3>& vertices = mesh->getVertices();

            // if not given sizes -> bounding box
            if( xmin.getValue()==0.0 && xmax.getValue()==0.0 && ymin.getValue()==0.0 && ymax.getValue()==0.0 && zmin.getValue()==0.0 && zmax.getValue()==0.0 )
            {
                // bounding box computation
                xmin.setValue( vertices[0][0] );
                ymin.setValue( vertices[0][1] );
                zmin.setValue( vertices[0][2] );
                xmax.setValue( vertices[0][0] );
                ymax.setValue( vertices[0][1] );
                zmax.setValue( vertices[0][2] );

                for(unsigned w=1; w<vertices.size(); ++w)
                {
                    if( vertices[w][0] > xmax.getValue() ) xmax.setValue( vertices[w][0] );
                    else if( vertices[w][0] < xmin.getValue() ) xmin.setValue( vertices[w][0] );
                    if( vertices[w][1] > ymax.getValue() ) ymax.setValue( vertices[w][1] );
                    else if( vertices[w][1] < ymin.getValue() ) ymin.setValue( vertices[w][1] );
                    if( vertices[w][2] > zmax.getValue() ) zmax.setValue( vertices[w][2] );
                    else if( vertices[w][2] < zmin.getValue() ) zmin.setValue( vertices[w][2] );
                }

                // increase the box a little
                Vec3 diff ( fabs(xmax.getValue()-xmin.getValue()), fabs(ymax.getValue()-ymin.getValue()),fabs(zmax.getValue()-zmin.getValue()) );
                diff /= 100.0;
                xmax.setValue( xmax.getValue() + diff[0] );
                xmin.setValue( xmin.getValue() - diff[0] );
                ymax.setValue( ymax.getValue() + diff[1] );
                ymin.setValue( ymin.getValue() - diff[1] );
                zmax.setValue( zmax.getValue() + diff[2] );
                zmin.setValue( zmin.getValue() - diff[2] );
            }



            _regularGrid.setSize(getNx(),getNy(),getNz());
            _regularGrid.setPos(xmin.getValue(), xmax.getValue(), ymin.getValue(), ymax.getValue(), zmin.getValue(), zmax.getValue());



            vector<Type> _regularGridTypes(_regularGrid.getNbCubes()); // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
            _indicesOfRegularCubeInSparseGrid.resize( _regularGrid.getNbCubes() ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
            for(int w=0; w<_regularGrid.getNbCubes(); ++w)
            {
                _regularGridTypes[w]=INSIDE;
                _indicesOfRegularCubeInSparseGrid[w] = -1;
            }

// 			// find all initial mesh edges to compute intersection with cubes
//             const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();
//             std::set< SegmentForIntersection,ltSegmentForIntersection > segmentsForIntersection;
//             for (unsigned int i=0;i<facets.size();i++)
//             {
//               const helper::vector<int>& facet = facets[i][0];
//               for (unsigned int j=2; j<facet.size(); j++) // Triangularize
//               {
//                 segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j]] ) );
//                 segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j-1]] ) );
//                 segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[j]],vertices[facet[j-1]] ) );
//               }
//             }


            vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
            MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values


            for(int i=0; i<_regularGrid.getNbCubes(); ++i) // all possible cubes (even empty)
            {
                Cube c = _regularGrid.getCubeCopy(i);
                CubeCorners corners;
                for(int j=0; j<8; ++j)
                    corners[j] = _regularGrid.getPoint( c[j] );

//               CubeForIntersection cubeForIntersection( corners );
//
//               for(std::set< SegmentForIntersection,ltSegmentForIntersection >::iterator it=segmentsForIntersection.begin();it!=segmentsForIntersection.end();++it)
//                 {
//                   if(intersectionSegmentBox( *it, cubeForIntersection ))
//                   {
//                     _types.push_back(BOUNDARY);
// 					_regularGridTypes[i]=BOUNDARY;
//
//                     for(int k=0;k<8;++k)
//                       cubeCornerPositionIndiceMap[corners[k]] = 0;
//
// 					cubeCorners.push_back(corners);
// 					_indicesOfRegularCubeInSparseGrid[i] = cubeCorners.size()-1;
//
//                     break;
//                   }
//                 }

                Vec3 cubeDiagonal = corners[7] - corners[0];
                Vec3 cubeCenter = corners[0] + cubeDiagonal*.5;


                bool notAlreadyIntersected = true;

                const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();
                for (unsigned int f=0; f<facets.size() && notAlreadyIntersected; f++)
                {
                    const helper::vector<int>& facet = facets[f][0];
                    for (unsigned int j=2; j<facet.size()&& notAlreadyIntersected; j++) // Triangularize
                    {
                        const Vec3& A = vertices[facet[0]];
                        const Vec3& B = vertices[facet[j-1]];
                        const Vec3& C = vertices[facet[j]];

                        // Scale the triangle to the unit cube matching
                        float points[3][3];
                        for (unsigned short w=0; w<3; ++w)
                        {
                            points[0][w] = (float) ((A[w]-cubeCenter[w])/cubeDiagonal[w]);
                            points[1][w] = (float) ((B[w]-cubeCenter[w])/cubeDiagonal[w]);
                            points[2][w] = (float) ((C[w]-cubeCenter[w])/cubeDiagonal[w]);
                        }

                        float normal[3];
                        helper::polygon_cube_intersection::get_polygon_normal(normal,3,points);

                        if (helper::polygon_cube_intersection::fast_polygon_intersects_cube(3,points,normal,0,0))
                        {
                            _types.push_back(BOUNDARY);
                            _regularGridTypes[i]=BOUNDARY;

                            for(int k=0; k<8; ++k)
                                cubeCornerPositionIndiceMap[corners[k]] = 0;

                            cubeCorners.push_back(corners);
                            _indicesOfRegularCubeInSparseGrid[i] = cubeCorners.size()-1;

                            notAlreadyIntersected=false;
                        }

                    }
                }


            }


            // TODO: regarder les cellules pleines, et les ajouter

            vector<bool> alreadyTested(_regularGrid.getNbCubes());
            for(int w=0; w<_regularGrid.getNbCubes(); ++w)
                alreadyTested[w]=false;

            // x==0 and x=nx-2
            for(int y=0; y<_regularGrid.getNy()-1; ++y)
                for(int z=0; z<_regularGrid.getNz()-1; ++z)
                {
                    propagateFrom( 0, y, z, _regularGrid, _regularGridTypes, alreadyTested );
                    propagateFrom( _regularGrid.getNx()-2, y, z, _regularGrid, _regularGridTypes, alreadyTested );
                }
            // y==0 and y=ny-2
            for(int x=0; x<_regularGrid.getNx()-1; ++x)
                for(int z=0; z<_regularGrid.getNz()-1; ++z)
                {
                    propagateFrom( x, 0, z, _regularGrid, _regularGridTypes, alreadyTested );
                    propagateFrom( x, _regularGrid.getNy()-2, z, _regularGrid, _regularGridTypes, alreadyTested );
                }
            // z==0 and z==Nz-2
            for(int y=0; y<_regularGrid.getNy()-1; ++y)
                for(int x=0; x<_regularGrid.getNx()-1; ++x)
                {
                    propagateFrom( x, y, 0, _regularGrid, _regularGridTypes, alreadyTested );
                    propagateFrom( x, y, _regularGrid.getNz()-2, _regularGrid, _regularGridTypes, alreadyTested );
                }

            // add INSIDE cubes to valid cells
            for(int w=0; w<_regularGrid.getNbCubes(); ++w)
                if( _regularGridTypes[w] == INSIDE )
                {
                    _types.push_back(INSIDE);

                    Cube c = _regularGrid.getCubeCopy(w);
                    CubeCorners corners;
                    for(int j=0; j<8; ++j)
                    {
                        corners[j] = _regularGrid.getPoint( c[j] );
                        cubeCornerPositionIndiceMap[corners[j]] = 0;
                    }
                    cubeCorners.push_back(corners);
                    _indicesOfRegularCubeInSparseGrid[w] = cubeCorners.size()-1;
                }


            // compute corner indices
            int cornerCounter=0;
            for(MapBetweenCornerPositionAndIndice::iterator it=cubeCornerPositionIndiceMap.begin(); it!=cubeCornerPositionIndiceMap.end(); ++it,++cornerCounter)
            {
                (*it).second = cornerCounter;
                seqPoints.push_back( (*it).first );
            }
            nbPoints = cubeCornerPositionIndiceMap.size();

            for( unsigned w=0; w<cubeCorners.size(); ++w)
            {
                Cube c;
                for(int j=0; j<8; ++j)
                    c[j] = cubeCornerPositionIndiceMap[cubeCorners[w][j]];

                seqCubes.push_back(c);
            }

            delete mesh;


        }
//         else
//           std::cerr << "SparseGridTopology: loading mesh "<<filename.getValue()<<" failed."<<std::endl;
    }

}



void SparseGridTopology::buildFromFiner(  )
{
// 		cerr<<"SparseGridTopology::buildFromFiner(  )\n";


    setNx( _finerSparseGrid->getNx()/2+1 );
    setNy( _finerSparseGrid->getNy()/2+1 );
    setNz( _finerSparseGrid->getNz()/2+1 );

    _regularGrid.setSize(getNx(),getNy(),getNz());

    xmin = _finerSparseGrid->getXmin();
    xmax = _finerSparseGrid->getXmax();
    ymin = _finerSparseGrid->getYmin();
    ymax = _finerSparseGrid->getYmax();
    zmin = _finerSparseGrid->getZmin();
    zmax = _finerSparseGrid->getZmax();
    _regularGrid.setPos(xmin.getValue(), xmax.getValue(), ymin.getValue(), ymax.getValue(), zmin.getValue(), zmax.getValue());

    _indicesOfRegularCubeInSparseGrid.resize( _regularGrid.getNbCubes() ); // to redirect an indice of a cube in the regular grid to its indice in the sparse grid
    for(int w=0; w<_regularGrid.getNbCubes(); ++w)
    {
        _indicesOfRegularCubeInSparseGrid[w] = -1;
    }

    vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
    MapBetweenCornerPositionAndIndice cubeCornerPositionIndiceMap; // to compute cube corner indice values


    for(int i=0; i<getNx()-1; i++)
        for(int j=0; j<getNy()-1; j++)
            for(int k=0; k<getNz()-1; k++)
            {
                int x = 2*i;
                int y = 2*j;
                int z = 2*k;

                fixed_array<int,8> fineIndices;
                fineIndices[0] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x,y,z) ];
                fineIndices[1] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x+1,y,z ) ];
                fineIndices[2] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x,y+1,z ) ];
                fineIndices[3] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x+1,y+1,z ) ];
                fineIndices[4] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x,y,z+1 ) ];
                fineIndices[5] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x+1,y,z+1 ) ];
                fineIndices[6] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x,y+1,z+1 ) ];
                fineIndices[7] = _finerSparseGrid->_indicesOfRegularCubeInSparseGrid[ _finerSparseGrid->_regularGrid.cube( x+1,y+1,z+1 ) ];

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
                if( inside ) _types.push_back(INSIDE);
                else _types.push_back(BOUNDARY);


                int coarseRegularIndice = _regularGrid.cube( i,j,k );
                Cube c = _regularGrid.getCubeCopy( coarseRegularIndice );

                CubeCorners corners;
                for(int w=0; w<8; ++w)
                {
                    corners[w] = _regularGrid.getPoint( c[w] );
                    cubeCornerPositionIndiceMap[corners[w]] = 0;
                }

                cubeCorners.push_back(corners);

                _indicesOfRegularCubeInSparseGrid[coarseRegularIndice] = cubeCorners.size()-1;

                _hierarchicalCubeMap[cubeCorners.size()-1]=fineIndices;
            }


    // compute corner indices
    int cornerCounter=0;
    for(MapBetweenCornerPositionAndIndice::iterator it=cubeCornerPositionIndiceMap.begin(); it!=cubeCornerPositionIndiceMap.end(); ++it,++cornerCounter)
    {
        (*it).second = cornerCounter;
        seqPoints.push_back( (*it).first );
    }
    nbPoints = cubeCornerPositionIndiceMap.size();

    for( unsigned w=0; w<cubeCorners.size(); ++w)
    {
        Cube c;
        for(int j=0; j<8; ++j)
            c[j] = cubeCornerPositionIndiceMap[cubeCorners[w][j]];

        seqCubes.push_back(c);
    }


    // for interpolation and restriction
    _hierarchicalPointMap.resize(seqPoints.size());
    for( unsigned w=0; w<seqCubes.size(); ++w)
    {
        const fixed_array<int, 8>& child = _hierarchicalCubeMap[w];



        helper::vector<int> fineCorners(27);
        fineCorners.fill(-1);
        for(int fineCube=0; fineCube<8; ++fineCube)
        {
            if( child[fineCube] == -1 ) continue;

            const Cube& cube = _finerSparseGrid->getCube(child[fineCube]);

            for(int vertex=0; vertex<8; ++vertex)
            {
// 					if( fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=-1 && fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]!=cube[vertex] )
// 						cerr<<"couille fineCorners\n";
                fineCorners[cornerIndicesFromFineToCoarse[fineCube][vertex]]=cube[vertex];
            }

        }


        for(int coarseCornerLocalIndice=0; coarseCornerLocalIndice<8; ++coarseCornerLocalIndice)
        {
            for( int fineVertexLocalIndice=0; fineVertexLocalIndice<27; ++fineVertexLocalIndice)
            {
                if( fineCorners[fineVertexLocalIndice] == -1 ) continue; // this fine vertex is not in any fine cube

                int coarseCornerGlobalIndice = seqCubes[w][coarseCornerLocalIndice];
                int fineVertexGlobalIndice = fineCorners[fineVertexLocalIndice];

                if( WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice] )
                    _hierarchicalPointMap[coarseCornerGlobalIndice][fineVertexGlobalIndice] = WEIGHT27[coarseCornerLocalIndice][fineVertexLocalIndice];
            }
        }

    }

// 		for(unsigned i=0;i<_finerSparseGrid->seqPoints.size();++i)
// 		{
// 			cerr<<i<<" : "<<_finerSparseGrid->seqPoints[i]<<endl;
// 		}
//
// 		for(unsigned i=0;i<_finerSparseGrid->seqCubes.size();++i)
// 		{
// 			cerr<<i<<" : "<<_finerSparseGrid->seqCubes[i]<<endl;
//
// 		}




// // 		afficher la _hierarchicalPointMap
// 		for(unsigned i=0;i<_hierarchicalPointMap.size();++i)
// 		{
// 			cerr<<"POINT "<<i<<" "<<seqPoints[i]<<" : "<<_hierarchicalPointMap[i].size()<<" : ";
// 			for(std::map<int,float>::iterator it = _hierarchicalPointMap[i].begin();it != _hierarchicalPointMap[i].end() ; ++it )
// 			{
// 				cerr<<(*it).first<<", "<<(*it).second<<" # ";
// 			}
// 			cerr<<endl;
// 		}


// 		for(int o=0;o<_hierarchicalPointMap.size();++o)
// 		{
// 			cerr<<o<<" : ";
// 			for(std::set<int>::iterator it=_hierarchicalPointMap[o].begin();it!=_hierarchicalPointMap[o].end();++it)
// 				cerr<<*it<<" ";
// 			cerr<<endl;
// 		}


// 		cerr<<"seqCubes : "<<seqCubes<<endl;
// 		cerr<<"seqPoints : "<<seqPoints<<endl;
}


///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////




/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int SparseGridTopology::findCube(const Vec3& pos, double& fx, double &fy, double &fz)
{
    int indiceInRegularGrid = _regularGrid.findCube( pos,fx,fy,fz);
    if( indiceInRegularGrid == -1 )
        return -1;
    else
        return _indicesOfRegularCubeInSparseGrid[indiceInRegularGrid];
}

/// return the cube containing the given point (or -1 if not found),
/// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
int SparseGridTopology::findNearestCube(const Vec3& pos, double& fx, double &fy, double &fz)
{
    int indice = 0;
    float lgmin = 99999999.0f;

    for(unsigned w=0; w<seqCubes.size(); ++w)
    {
        if(_types[w]!=BOUNDARY)continue;

        const Cube& c = getCube( w );
        int c0 = c[0];
        int c7 = c[7];
        Vec3 p0(getPX(c0),getPY(c0),getPZ(c0));
        Vec3 p7(getPX(c7),getPY(c7),getPZ(c7));

        Vec3 barycenter = (p0+p7) * .5;

        float lg = (float)((pos-barycenter).norm());
        if( lg < lgmin )
        {
            lgmin = lg;
            indice = w;
        }
    }

    const Cube& c = getCube( indice );
    int c0 = c[0];
    int c7 = c[7];
    Vec3 p0(getPX(c0),getPY(c0),getPZ(c0));
    Vec3 p7(getPX(c7),getPY(c7),getPZ(c7));

    Vec3 relativePos = pos-p0;
    Vec3 diagonal = p7 - p0;

    fx = relativePos[0] / diagonal[0];
    fy = relativePos[1] / diagonal[1];
    fz = relativePos[2] / diagonal[2];

    return indice;
}


SparseGridTopology::Type SparseGridTopology::getType( int i )
{
    return _types[i];
}



///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////



void SparseGridTopology::updateLines()
{
    std::map<pair<int,int>,bool> edgesMap;
    for(unsigned i=0; i<seqCubes.size(); ++i)
    {
        Cube c = seqCubes[i];
        // horizontal
        edgesMap[pair<int,int>(c[0],c[1])]=0;
        edgesMap[pair<int,int>(c[2],c[3])]=0;
        edgesMap[pair<int,int>(c[4],c[5])]=0;
        edgesMap[pair<int,int>(c[6],c[7])]=0;
        // vertical
        edgesMap[pair<int,int>(c[0],c[2])]=0;
        edgesMap[pair<int,int>(c[1],c[3])]=0;
        edgesMap[pair<int,int>(c[4],c[6])]=0;
        edgesMap[pair<int,int>(c[5],c[7])]=0;
        // profondeur
        edgesMap[pair<int,int>(c[0],c[4])]=0;
        edgesMap[pair<int,int>(c[1],c[5])]=0;
        edgesMap[pair<int,int>(c[2],c[6])]=0;
        edgesMap[pair<int,int>(c[3],c[7])]=0;
    }


    SeqLines& lines = *seqLines.beginEdit();
    lines.clear();
    lines.reserve(edgesMap.size());
    for( std::map<pair<int,int>,bool>::iterator it=edgesMap.begin(); it!=edgesMap.end(); ++it)
        lines.push_back( Line( (*it).first.first,  (*it).first.second ));
    seqLines.endEdit();
}

void SparseGridTopology::updateQuads()
{
    std::map<fixed_array<int,4>,bool> quadsMap;
    for(unsigned i=0; i<seqCubes.size(); ++i)
    {
        Cube c = seqCubes[i];

        fixed_array<int,4> v;
        v[0]=c[0]; v[1]=c[1]; v[2]=c[3]; v[3]=c[2];
        quadsMap[v]=0;
        v[0]=c[4]; v[1]=c[5]; v[2]=c[7]; v[3]=c[6];
        quadsMap[v]=0;
        v[0]=c[0]; v[1]=c[1]; v[2]=c[5]; v[3]=c[4];
        quadsMap[v]=0;
        v[0]=c[2]; v[1]=c[3]; v[2]=c[7]; v[3]=c[6];
        quadsMap[v]=0;
        v[0]=c[0]; v[1]=c[4]; v[2]=c[6]; v[3]=c[2];
        quadsMap[v]=0;
        v[0]=c[1]; v[1]=c[5]; v[2]=c[7]; v[3]=c[3];
        quadsMap[v]=0;

    }

    seqQuads.clear();
    seqQuads.reserve(quadsMap.size());
    for( std::map<fixed_array<int,4>,bool>::iterator it=quadsMap.begin(); it!=quadsMap.end(); ++it)
        seqQuads.push_back( Quad( (*it).first[0],  (*it).first[1],(*it).first[2],(*it).first[3] ));
}

void SparseGridTopology::updateCubes()
{
    // 					seqCubes.clear();
    // 					seqCubes.reserve(_cubes.size());

    // 								seqCubes.push_back(Cube(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
    // 										point(x  ,y+1,z  ),point(x+1,y+1,z  ),
    // 										point(x  ,y  ,z+1),point(x+1,y  ,z+1),
    // 										point(x  ,y+1,z+1),point(x+1,y+1,z+1)));
}


////////////////////////////////////////
////////////////////////////////////////

//     bool SparseGridTopology::intersectionSegmentBox( const SegmentForIntersection& seg, const CubeForIntersection& cube  )
//     {
//       Vec3 afAWdU, afADdU, afAWxDdU;
//       Real fRhs;
//
//
//       Vec3 kDiff = seg.center - cube.center;
//
//       afAWdU[0] = fabs( seg.dir* cube.dir[0]);
//       afADdU[0] = fabs( kDiff * cube.dir[0] );
//       fRhs = cube.norm[0] + seg.norm*afAWdU[0];
//       if (afADdU[0] > fRhs)
//       {
//         return false;
//       }
//
//       afAWdU[1] = fabs(seg.dir*cube.dir[1]);
//       afADdU[1] = fabs(kDiff*cube.dir[1]);
//       fRhs = cube.norm[1] + seg.norm*afAWdU[1];
//       if (afADdU[1] > fRhs)
//       {
//         return false;
//       }
//
//       afAWdU[2] = fabs(seg.dir*cube.dir[2]);
//       afADdU[2] = fabs(kDiff*cube.dir[2]);
//       fRhs = cube.norm[2] + seg.norm*afAWdU[2];
//       if (afADdU[2] > fRhs)
//       {
//         return false;
//       }
//
//       Vec3 kWxD = seg.dir.cross(kDiff);
//
//       afAWxDdU[0] = fabs(kWxD*cube.dir[0]);
//       fRhs = cube.norm[1]*afAWdU[2] + cube.norm[2]*afAWdU[1];
//       if (afAWxDdU[0] > fRhs)
//       {
//         return false;
//       }
//
//       afAWxDdU[1] = fabs(kWxD*cube.dir[1]);
//       fRhs = cube.norm[0]*afAWdU[2] + cube.norm[2]*afAWdU[0];
//       if (afAWxDdU[1] > fRhs)
//       {
//         return false;
//       }
//
//       afAWxDdU[2] = fabs(kWxD*cube.dir[2]);
//       fRhs = cube.norm[0]*afAWdU[1] +cube.norm[1]*afAWdU[0];
//       if (afAWxDdU[2] > fRhs)
//       {
//         return false;
//       }
//
//       return true;
//     }


void SparseGridTopology::propagateFrom( const int i, const int j, const int k,  RegularGridTopology& _regularGrid, vector<Type>& _regularGridTypes, vector<bool>& alreadyTested  )
{
    assert( i>=0 && i<=_regularGrid.getNx()-2 && j>=0 && j<=_regularGrid.getNy()-2 && k>=0 && k<=_regularGrid.getNz()-2 );

    unsigned indice = _regularGrid.cube( i, j, k );

    if( alreadyTested[indice] || _regularGridTypes[indice] == BOUNDARY )
        return;

    alreadyTested[indice] = true;
    _regularGridTypes[indice] = OUTSIDE;

    if(i>0) propagateFrom( i-1, j, k, _regularGrid, _regularGridTypes, alreadyTested );
    if(i<_regularGrid.getNx()-2) propagateFrom( i+1, j, k, _regularGrid, _regularGridTypes, alreadyTested );
    if(j>0) propagateFrom( i, j-1, k, _regularGrid, _regularGridTypes, alreadyTested );
    if(j<_regularGrid.getNy()-2) propagateFrom( i, j+1, k, _regularGrid, _regularGridTypes, alreadyTested );
    if(k>0) propagateFrom( i, j, k-1, _regularGrid, _regularGridTypes, alreadyTested );
    if(k<_regularGrid.getNz()-2) propagateFrom( i, j, k+1, _regularGrid, _regularGridTypes, alreadyTested );
}




/////////////////////


const SparseGridTopology::SeqCubes& SparseGridTopology::getCubes()
{
    if( !_alreadyInit ) init();
    return MeshTopology::getCubes();
}


int SparseGridTopology::getNbPoints() const
{
    if( !_alreadyInit ) const_cast<SparseGridTopology*>(this)->init();
    return MeshTopology::getNbPoints();
}


} // namespace topology

} // namespace component

} // namespace sofa
