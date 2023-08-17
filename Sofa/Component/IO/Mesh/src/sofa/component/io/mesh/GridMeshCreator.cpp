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
#include <sofa/component/io/mesh/GridMeshCreator.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::io::mesh
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::loader;
using type::vector;

int GridMeshCreatorClass = core::RegisterObject("Procedural creation of a two-dimensional mesh.")
        .add< GridMeshCreator >()
        ;



GridMeshCreator::GridMeshCreator(): MeshLoader()
    , resolution( initData(&resolution,Vec2i(2,2),"resolution","Number of vertices in each direction"))
    , trianglePattern( initData(&trianglePattern,2,"trianglePattern","0: no triangles, 1: alternate triangles, 2: upward triangles, 3: downward triangles"))
{
    // doLoad() is called only if d_filename is modified
    // but this loader in particular does not require a filename (refactoring would be needed)
    // we force d_filename to be dirty to trigger the callback, thus calling doLoad()
    d_filename.setDirtyValue();

    d_filename.setReadOnly(true);
}

void GridMeshCreator::doClearBuffers()
{

}


void GridMeshCreator::insertUniqueEdge( unsigned a, unsigned b )
{
    if( uniqueEdges.find(Edge(b,a))==uniqueEdges.end() ) // symmetric not found
        uniqueEdges.insert(Edge(a,b));                   // redundant elements are automatically pruned
}

void GridMeshCreator::insertTriangle(unsigned a, unsigned b, unsigned c)
{
    auto my_triangles = getWriteOnlyAccessor(d_triangles);

    my_triangles.push_back(Triangle( a,b,c ) );
    insertUniqueEdge(a,b);
    insertUniqueEdge(b,c);
    insertUniqueEdge(c,a);
}

void GridMeshCreator::insertQuad(unsigned a, unsigned b, unsigned c, unsigned d)
{
    auto my_quads = getWriteOnlyAccessor(d_quads);

    my_quads.push_back( Quad( a,b,c,d ) );
    insertUniqueEdge(a,b);
    insertUniqueEdge(b,c);
    insertUniqueEdge(c,d);
    insertUniqueEdge(d,a);
}


bool GridMeshCreator::doLoad()
{
    auto my_positions = getWriteOnlyAccessor(d_positions);
    const unsigned numX = resolution.getValue()[0], numY=resolution.getValue()[1];

    // Warning: Vertex creation order must be consistent with method vert.
    for(unsigned y=0; y<numY; y++)
    {
        for(unsigned x=0; x<numX; x++)
        {
            my_positions.push_back( Vec3(x * 1._sreal/(numX-1), y * 1._sreal/(numY-1), 0_sreal) );
        }
    }

    uniqueEdges.clear();

    if( trianglePattern.getValue()==0 ) // quads
        for(unsigned y=0; y<numY-1; y++ )
        {
            for(unsigned x=0; x<numX-1; x++ )
            {
                insertQuad( vert(x,y), vert(x+1,y), vert(x+1,y+1), vert(x,y+1) );
            }
        }
    else if( trianglePattern.getValue()==1 ) // alternate
        for(unsigned y=0; y<numY-1; y++ )
        {
            for(unsigned x=0; x<numX-1; x++ )
            {
                if( (x+y)%2 == 0 )
                {
                    insertTriangle( vert(x,y), vert(x+1,y), vert(x+1,y+1) );
                    insertTriangle( vert(x,y), vert(x+1,y+1), vert(x,y+1)   ) ;
                }
                else
                {
                    insertTriangle( vert(x  ,y), vert(x+1,y)  , vert(x,y+1) ) ;
                    insertTriangle( vert(x+1,y), vert(x+1,y+1), vert(x,y+1)    ) ;
                }
            }
        }
    else if( trianglePattern.getValue()==2 ) // upward
        for(unsigned y=0; y<numY-1; y++ )
        {
            for(unsigned x=0; x<numX-1; x++ )
            {
                insertTriangle( vert(x,y), vert(x+1,y), vert(x+1,y+1) ) ;
                insertTriangle( vert(x,y), vert(x+1,y+1), vert(x,y+1)   ) ;
            }
        }
    else if( trianglePattern.getValue()==3 ) // downward
        for(unsigned y=0; y<numY-1; y++ )
        {
            for(unsigned x=0; x<numX-1; x++ )
            {
                insertTriangle( vert(x  ,y), vert(x+1,y)  , vert(x,y+1) ) ;
                insertTriangle( vert(x+1,y), vert(x+1,y+1), vert(x,y+1)    ) ;
            }
        }

    auto my_edges = getWriteOnlyAccessor(d_edges);
    for( std::set<Edge>::const_iterator it=uniqueEdges.begin(),itEnd=uniqueEdges.end(); it!=itEnd; ++it )
        my_edges.push_back( *it );

    return true;
}


} // namespace sofa::component::io::mesh
