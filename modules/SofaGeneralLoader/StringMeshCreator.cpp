/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaGeneralLoader/StringMeshCreator.h>
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
using helper::vector;

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
    helper::WriteAccessor<Data<vector<sofa::defaulttype::Vector3> > > my_positions (d_positions);
    unsigned numX = resolution.getValue();

    // Warning: Vertex creation order must be consistent with method vert.
    for(unsigned x=0; x<numX; x++)
    {
        my_positions.push_back( Vector3(x * 1./(numX-1), 0, 0) );
        //            cerr<<"StringMeshCreator::load, add point " << Vector3(i * 1./(numX-1), j * 1./(numY-1), 0) << endl;
    }
    helper::vector<Edge >& my_edges = *(d_edges.beginEdit());
    for( unsigned e=1; e<numX; e++ )
    {
        my_edges.push_back( Edge(e-1,e) );
    }
    d_edges.endEdit();

    return true;

}


}
}
}

