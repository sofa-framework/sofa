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
#define SOFA_COMPONENT_ENGINE_MeshBoundaryROI_CPP
#include <sofa/component/engine/select/MeshBoundaryROI.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::engine::select
{

int MeshBoundaryROIClass = core::RegisterObject("Outputs indices of boundary vertices of a triangle/quad mesh")
        .add< MeshBoundaryROI >(true);

MeshBoundaryROI::MeshBoundaryROI(): Inherited()
                                    , d_triangles(initData(&d_triangles,"triangles","input triangles"))
                                    , d_quads(initData(&d_quads,"quads","input quads"))
                                    , d_inputROI(initData(&d_inputROI,"inputROI","optional subset of the input mesh"))
                                    , d_indices(initData(&d_indices,"indices","Index lists of the closing vertices"))
{
    addInput(&d_triangles);
    addInput(&d_quads);
    addInput(&d_inputROI);
    addOutput(&d_indices);
}

void MeshBoundaryROI::doBaseObjectInit()
{
    setDirtyValue();
}

void MeshBoundaryROI::reinit()
{
    update();
}

void MeshBoundaryROI::doUpdate()
{
    const helper::ReadAccessor triangles(this->d_triangles);
    const helper::ReadAccessor quads(this->d_quads);

    helper::WriteOnlyAccessor<Data< SetIndex > >  indices(this->d_indices);
    indices.clear();

    std::map<PointPair, unsigned int> edgeCount;
    for(size_t i=0;i<triangles.size();i++)
    {
        if(inROI(triangles[i][0]) && inROI(triangles[i][1]) && inROI(triangles[i][2]))
        {
            for(unsigned int j=0;j<3;j++)
            {
                PointPair edge(triangles[i][j],triangles[i][(j==2)?0:j+1]);
                // increment the number of elements (triangles) associated to the edge.
                this->countEdge(edgeCount,edge);
            }
        }
    }
    
    for(size_t i=0;i<quads.size();i++)
    {
        if(inROI(quads[i][0]) && inROI(quads[i][1]) && inROI(quads[i][2]) && inROI(quads[i][3]))
        {
            for(unsigned int j=0;j<4;j++)
            {
                PointPair edge(quads[i][j],quads[i][(j==3)?0:j+1]);
                // increment the number of elements (quad) associated to the edge.
                this->countEdge(edgeCount,edge);
            }
        }
    }

    std::set<PointID> indexset; // enforce uniqueness since SetIndex is not a set..
    for(auto it=edgeCount.begin();it!=edgeCount.end();++it)
    {
        // consider edge only if it is on the boundary
        if(it->second==1)
        {
            indexset.insert(it->first.first);
            indexset.insert(it->first.second);
        }
    }
    indices.wref().insert(indices.end(), indexset.begin(), indexset.end());
}

void MeshBoundaryROI::countEdge(std::map<PointPair, unsigned>& edgeCount, PointPair& edge)
{
    if(edge.first > edge.second)
    {
        std::swap(edge.first, edge.second);
    }
    const auto it = edgeCount.find(edge);
    if(it != edgeCount.end())
    {
        it->second++;
    }
    else
    {
        edgeCount[edge]=1;
    }
}

bool MeshBoundaryROI::inROI(const PointID& index) const
{
    const SetIndex& ROI=this->d_inputROI.getValue();
    if(ROI.size()==0) return true; // ROI empty -> use all points
    if(std::find(ROI.begin(),ROI.end(),index)==ROI.end()) return false;
    return true;
}
    
} //namespace sofa::component::engine::select
