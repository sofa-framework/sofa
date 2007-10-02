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

SparseGridTopology::SparseGridTopology(): nx(dataField(&nx,0,"nx","x grid resolution")), ny(dataField(&ny,0,"ny","y grid resolution")), nz(dataField(&nz,0,"nz","z grid resolution")),
    xmin(dataField(&xmin,0.0,"xmin","xmin grid")),ymin(dataField(&ymin,0.0,"ymin","ymin grid")),zmin(dataField(&zmin,0.0,"zmin","zmin grid")),
    xmax(dataField(&xmax,0.0,"xmax","xmax grid")),ymax(dataField(&ymax,0.0,"ymax","ymax grid")),zmax(dataField(&zmax,0.0,"zmax","zmax grid"))
{}



bool SparseGridTopology::load(const char* filename)
{
    this->filename.setValue( filename );
    return true;
}







void SparseGridTopology::init()
{
    this->MeshTopology::init();
    invalidate();

    if (!filename.getValue().empty())
    {
        // 						std::cout << "SparseGridTopology: using mesh "<<filename.getValue()<<std::endl;
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
            }


            RegularGridTopology regularGrid;
            regularGrid.setSize(getNx(),getNy(),getNz());
            regularGrid.setPos(xmin.getValue(), xmax.getValue(), ymin.getValue(), ymax.getValue(), zmin.getValue(), zmax.getValue());


            // to compute filling types (OUTSIDE, INSIDE, BOUNDARY)
            vector<Type> regularGridTypes(regularGrid.getNbCubes());
            for(int w=0; w<regularGrid.getNbCubes(); ++w)
                regularGridTypes[w]=INSIDE;

            // find all initial mesh edges to compute intersection with cubes
            const helper::vector< helper::vector < helper::vector <int> > >& facets = mesh->getFacets();
            std::set< SegmentForIntersection,ltSegmentForIntersection > segmentsForIntersection;
            for (unsigned int i=0; i<facets.size(); i++)
            {
                const helper::vector<int>& facet = facets[i][0];
                for (unsigned int j=2; j<facet.size(); j++) // Triangularize
                {
                    segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j]] ) );
                    segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[0]],vertices[facet[j-1]] ) );
                    segmentsForIntersection.insert( SegmentForIntersection( vertices[facet[j]],vertices[facet[j-1]] ) );
                }
            }


            vector< CubeCorners > cubeCorners; // saving temporary positions of all cube corners
            std::map<Vec3,int> cubeCornerPositionIndiceMap; // to compute cube corner indice values

            for(int i=0; i<regularGrid.getNbCubes(); ++i) // all possible cubes (even empty)
            {
                Cube c = regularGrid.getCube(i);
                CubeCorners corners;
                for(int j=0; j<8; ++j)
                    corners[j] = regularGrid.getPoint( c[j] );
                CubeForIntersection cubeForIntersection( corners );

                for(std::set< SegmentForIntersection,ltSegmentForIntersection >::iterator it=segmentsForIntersection.begin(); it!=segmentsForIntersection.end(); ++it)
                {
                    if(intersectionSegmentBox( *it, cubeForIntersection ))
                    {
                        _types.push_back(BOUNDARY);
                        regularGridTypes[i]=BOUNDARY;

                        for(int k=0; k<8; ++k)
                            cubeCornerPositionIndiceMap[corners[k]] = 0;

                        cubeCorners.push_back(corners);

                        break;
                    }
                }
            }


            // TODO: regarder les cellules pleines, et les ajouter

            vector<bool> alreadyTested(regularGrid.getNbCubes());
            for(int w=0; w<regularGrid.getNbCubes(); ++w)
                alreadyTested[w]=false;

            // x==0 and x=nx-2
            for(int y=0; y<regularGrid.getNy()-1; ++y)
                for(int z=0; z<regularGrid.getNz()-1; ++z)
                {
                    propagateFrom( 0, y, z, regularGrid, regularGridTypes, alreadyTested );
                    propagateFrom( regularGrid.getNx()-2, y, z, regularGrid, regularGridTypes, alreadyTested );
                }
            // y==0 and y=ny-2
            for(int x=0; x<regularGrid.getNx()-1; ++x)
                for(int z=0; z<regularGrid.getNz()-1; ++z)
                {
                    propagateFrom( x, 0, z, regularGrid, regularGridTypes, alreadyTested );
                    propagateFrom( x, regularGrid.getNy()-2, z, regularGrid, regularGridTypes, alreadyTested );
                }
            // z==0 and z==Nz-2
            for(int y=0; y<regularGrid.getNy()-1; ++y)
                for(int x=0; x<regularGrid.getNx()-1; ++x)
                {
                    propagateFrom( x, y, 0, regularGrid, regularGridTypes, alreadyTested );
                    propagateFrom( x, y, regularGrid.getNz()-2, regularGrid, regularGridTypes, alreadyTested );
                }

            // add INSIDE cubes to valid cells
            for(int w=0; w<regularGrid.getNbCubes(); ++w)
                if( regularGridTypes[w] == INSIDE )
                {
                    _types.push_back(INSIDE);

                    Cube c = regularGrid.getCube(w);
                    CubeCorners corners;
                    for(int j=0; j<8; ++j)
                    {
                        corners[j] = regularGrid.getPoint( c[j] );
                        cubeCornerPositionIndiceMap[corners[j]] = 0;
                    }

                    cubeCorners.push_back(corners);
                }


            // compute corner indices
            int cornerCounter=0;
            for(std::map<Vec3,int>::iterator it=cubeCornerPositionIndiceMap.begin(); it!=cubeCornerPositionIndiceMap.end(); ++it,++cornerCounter)
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
        else
            std::cerr << "SparseGridTopology: loading mesh "<<filename.getValue()<<" failed."<<std::endl;
    }


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

bool SparseGridTopology::intersectionSegmentBox( const SegmentForIntersection& seg, const CubeForIntersection& cube  )
{
    Vec3 afAWdU, afADdU, afAWxDdU;
    float fRhs;


    Vec3 kDiff = seg.center - cube.center;

    afAWdU[0] = fabs( seg.dir* cube.dir[0]);
    afADdU[0] = fabs( kDiff * cube.dir[0] );
    fRhs = cube.norm[0] + seg.norm*afAWdU[0];
    if (afADdU[0] > fRhs)
    {
        return false;
    }

    afAWdU[1] = fabs(seg.dir*cube.dir[1]);
    afADdU[1] = fabs(kDiff*cube.dir[1]);
    fRhs = cube.norm[1] + seg.norm*afAWdU[1];
    if (afADdU[1] > fRhs)
    {
        return false;
    }

    afAWdU[2] = fabs(seg.dir*cube.dir[2]);
    afADdU[2] = fabs(kDiff*cube.dir[2]);
    fRhs = cube.norm[2] + seg.norm*afAWdU[2];
    if (afADdU[2] > fRhs)
    {
        return false;
    }

    Vec3 kWxD = seg.dir.cross(kDiff);

    afAWxDdU[0] = fabs(kWxD*cube.dir[0]);
    fRhs = cube.norm[1]*afAWdU[2] + cube.norm[2]*afAWdU[1];
    if (afAWxDdU[0] > fRhs)
    {
        return false;
    }

    afAWxDdU[1] = fabs(kWxD*cube.dir[1]);
    fRhs = cube.norm[0]*afAWdU[2] + cube.norm[2]*afAWdU[0];
    if (afAWxDdU[1] > fRhs)
    {
        return false;
    }

    afAWxDdU[2] = fabs(kWxD*cube.dir[2]);
    fRhs = cube.norm[0]*afAWdU[1] +cube.norm[1]*afAWdU[0];
    if (afAWxDdU[2] > fRhs)
    {
        return false;
    }

    return true;
}


void SparseGridTopology::propagateFrom( const int i, const int j, const int k,  RegularGridTopology& regularGrid, vector<Type>& regularGridTypes, vector<bool>& alreadyTested  )
{
    assert( i>=0 && i<=regularGrid.getNx()-2 && j>=0 && j<=regularGrid.getNy()-2 && k>=0 && k<=regularGrid.getNz()-2 );

    unsigned indice = regularGrid.cube( i, j, k );

    if( alreadyTested[indice] || regularGridTypes[indice] == BOUNDARY )
        return;

    alreadyTested[indice] = true;
    regularGridTypes[indice] = OUTSIDE;

    if(i>0) propagateFrom( i-1, j, k, regularGrid, regularGridTypes, alreadyTested );
    if(i<regularGrid.getNx()-2) propagateFrom( i+1, j, k, regularGrid, regularGridTypes, alreadyTested );
    if(j>0) propagateFrom( i, j-1, k, regularGrid, regularGridTypes, alreadyTested );
    if(j<regularGrid.getNy()-2) propagateFrom( i, j+1, k, regularGrid, regularGridTypes, alreadyTested );
    if(k>0) propagateFrom( i, j, k-1, regularGrid, regularGridTypes, alreadyTested );
    if(k<regularGrid.getNz()-2) propagateFrom( i, j, k+1, regularGrid, regularGridTypes, alreadyTested );
}










} // namespace topology

} // namespace component

} // namespace sofa
