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
#include <sofa/component/io/mesh/StringMeshCreator.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::io::mesh
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::loader;
using type::vector;

int StringMeshCreatorClass = core::RegisterObject("Procedural creation of a one-dimensional mesh.")
        .add< StringMeshCreator >()
        ;



StringMeshCreator::StringMeshCreator(): MeshLoader()
  , resolution( initData(&resolution,(unsigned)2,"resolution","Number of vertices"))
{
    addUpdateCallback("updateResolution", {&resolution}, [this](const core::DataTracker& )
    {
        if(load())
        {
            return sofa::core::objectmodel::ComponentState::Valid;
        }

        return sofa::core::objectmodel::ComponentState::Invalid;

    }, {&d_positions, &d_edges});
}

void StringMeshCreator::doClearBuffers()
{
    sofa::helper::getWriteOnlyAccessor(d_positions).clear();
    sofa::helper::getWriteOnlyAccessor(d_edges).clear();
}


bool StringMeshCreator::doLoad()
{
    auto my_positions = sofa::helper::getWriteOnlyAccessor(d_positions);
    auto my_edges = sofa::helper::getWriteOnlyAccessor(d_edges);

    const unsigned numX = resolution.getValue();

    // Warning: Vertex creation order must be consistent with method vert.
    for(unsigned x=0; x<numX; x++)
    {
        my_positions.push_back( Vec3(x * 1./(numX-1), 0, 0) );
    }

    for( unsigned e=1; e<numX; e++ )
    {
        my_edges.push_back( Edge(e-1,e) );
    }

    return true;

}


} // namespace sofa::component::io::mesh
