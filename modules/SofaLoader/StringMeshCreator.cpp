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
#include <sofa/core/ObjectFactory.h>
#include <SofaLoader/StringMeshCreator.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/SetDirectory.h>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

SOFA_DECL_CLASS(StringMeshCreator)

int StringMeshCreatorClass = core::RegisterObject("Procedural creation of a one-dimensional mesh.")
        .add< StringMeshCreator >()
        ;



StringMeshCreator::StringMeshCreator(): MeshLoader()
  , resolution( initData(&resolution,(unsigned)2,"resolution","Number of vertices"))
{
}


bool StringMeshCreator::load()
{
    helper::WriteAccessor<Data<vector<sofa::defaulttype::Vector3> > > my_positions (positions);
    unsigned numX = resolution.getValue();

    // Warning: Vertex creation order must be consistent with method vert.
    for(unsigned x=0; x<numX; x++)
    {
        my_positions.push_back( Vector3(x * 1./(numX-1), 0, 0) );
        //            cerr<<"StringMeshCreator::load, add point " << Vector3(i * 1./(numX-1), j * 1./(numY-1), 0) << endl;
    }
    helper::vector<Edge >& my_edges = *(edges.beginEdit());
    for( unsigned e=1; e<numX; e++ )
    {
        my_edges.push_back( Edge(e-1,e) );
    }
    edges.endEdit();

    return true;

}


}
}
}

