/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_ENGINE_MeshBoundaryROI_H
#define SOFA_COMPONENT_ENGINE_MeshBoundaryROI_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/SVector.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class outputs indices of boundary vertices of a triangle/quad mesh
 * @author benjamin gilles
 */
class MeshBoundaryROI : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(MeshBoundaryROI,Inherited);

    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef std::pair<PointID, PointID> PointPair;

    /// inputs
    Data< SeqTriangles > d_triangles;
    Data< SeqQuads > d_quads;
    Data< SetIndex > d_inputROI;

    /// outputs
    Data< SetIndex > d_indices;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const MeshBoundaryROI* = NULL) { return std::string();    }

protected:

    MeshBoundaryROI()    : Inherited()
      , d_triangles(initData(&d_triangles,"triangles","input triangles"))
      , d_quads(initData(&d_quads,"quads","input quads"))
      , d_inputROI(initData(&d_inputROI,"inputROI","optional subset of the input mesh"))
      , d_indices(initData(&d_indices,"indices","Index lists of the closing vertices"))
    {
    }

    virtual ~MeshBoundaryROI() {}

public:
    virtual void init() override
    {
        addInput(&d_triangles);
        addInput(&d_quads);
        addInput(&d_inputROI);
        addOutput(&d_indices);

        setDirtyValue();
    }

    virtual void reinit()    override { update();  }
    void update() override
    {
        helper::ReadAccessor<Data< SeqTriangles > > triangles(this->d_triangles);
        helper::ReadAccessor<Data< SeqQuads > > quads(this->d_quads);

        cleanDirty();

        helper::WriteOnlyAccessor<Data< SetIndex > >  indices(this->d_indices);
        indices.clear();

        std::map<PointPair, unsigned int> edgeCount;
        for(size_t i=0;i<triangles.size();i++)
            if(inROI(triangles[i][0]) && inROI(triangles[i][1]) && inROI(triangles[i][2]))
                for(unsigned int j=0;j<3;j++)
                {
                    PointPair edge(triangles[i][j],triangles[i][(j==2)?0:j+1]);
                    this->countEdge(edgeCount,edge);
                }
        for(size_t i=0;i<quads.size();i++)
            if(inROI(quads[i][0]) && inROI(quads[i][1]) && inROI(quads[i][2]) && inROI(quads[i][3]))
                for(unsigned int j=0;j<4;j++)
                {
                    PointPair edge(quads[i][j],quads[i][(j==3)?0:j+1]);
                    this->countEdge(edgeCount,edge);
                }

        std::set<PointID> indexset; // enforce uniqueness since SetIndex is not a set..
        for(std::map<PointPair, unsigned int>::iterator it=edgeCount.begin();it!=edgeCount.end();++it)
            if(it->second==1)
            {
                indexset.insert(it->first.first);
                indexset.insert(it->first.second);
            }
        indices.wref().insert(indices.end(), indexset.begin(), indexset.end());
    }

    void countEdge(std::map<PointPair, unsigned int>& edgeCount,PointPair& edge) const
    {
        if(edge.first>edge.second)
        {
            PointID i=edge.first;
            edge.first=edge.second;
            edge.second=i;
        }
        std::map<PointPair, unsigned int>::iterator it=edgeCount.find(edge);
        if(it!=edgeCount.end()) it->second++;
        else  edgeCount[edge]=1;
    }

    inline bool inROI(const PointID& index) const
    {
        const SetIndex& ROI=this->d_inputROI.getValue();
        if(ROI.size()==0) return true; // ROI empty -> use all points
        if(std::find(ROI.begin(),ROI.end(),index)==ROI.end()) return false;
        return true;
    }

};

} // namespace engine

} // namespace component

} // namespace sofa

#endif
