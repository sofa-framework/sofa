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
#pragma once

#include <SofaNonUniformFem/config.h>

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/defaulttype/Vec.h>
#include <set>

namespace sofa::core::topology
{

class TopologyChange;

} // namespace sofa::core::topology

namespace sofa::component::topology
{

class MultilevelHexahedronSetTopologyModifier;


class SOFA_SOFANONUNIFORMFEM_API MultilevelHexahedronSetTopologyContainer : public HexahedronSetTopologyContainer
{
    friend class MultilevelHexahedronSetTopologyModifier;

public:
    SOFA_CLASS(MultilevelHexahedronSetTopologyContainer,HexahedronSetTopologyContainer);

    typedef defaulttype::Vec<3,int>			Vec3i;
protected:
    MultilevelHexahedronSetTopologyContainer();

    MultilevelHexahedronSetTopologyContainer(const helper::vector< Hexahedron > &hexahedra);

    ~MultilevelHexahedronSetTopologyContainer() override;
public:
    void init() override;

    void clear() override;

    void getHexaNeighbors(const index_type hexaId,
            helper::vector<index_type> &neighbors);

    void getHexaFaceNeighbors(const index_type hexaId,
            const index_type faceId,
            helper::vector<index_type> &neighbors);

    void getHexaVertexNeighbors(const index_type hexaId,
            const index_type vertexId,
            helper::vector<index_type> &neighbors);

    void addTopologyChangeFine(const core::topology::TopologyChange *topologyChange)
    {
        m_changeListFine.push_back(topologyChange);
    }

    void resetTopologyChangeListFine()
    {
        for(std::list<const core::topology::TopologyChange *>::iterator it = m_changeListFine.begin();
            it != m_changeListFine.end(); ++it)
        {
            delete (*it);
        }
        m_changeListFine.clear();
    }

    std::list<const core::topology::TopologyChange *>::const_iterator beginChangeFine() const
    {
        return m_changeListFine.begin();
    }

    std::list<const core::topology::TopologyChange *>::const_iterator endChangeFine() const
    {
        return m_changeListFine.end();
    }

    const std::list<const core::topology::TopologyChange *>& getChangeListFine() const
    {
        return m_changeListFine;
    }

    int getLevel() const {return _level.getValue();}



    const Vec3i& getCoarseResolution() const { return _coarseResolution; }

    bool getHexaContainsPosition(const index_type hexaId, const defaulttype::Vector3& baryC) const;

    const Vec3i& getHexaIdxInCoarseRegularGrid(const index_type hexaId) const;
    int getHexaIdInCoarseRegularGrid(const index_type hexaId) const;

    const Vec3i& getHexaIdxInFineRegularGrid(const index_type hexaId) const;
    index_type getHexaIdInFineRegularGrid(const index_type hexaId) const;

    // gets a vector of fine hexahedra inside a specified coarse hexa
    index_type getHexaChildren(const index_type hexaId, helper::vector<index_type>& children) const;

    // gets a coarse hexa for a specified fine hexa
    index_type getHexaParent(const index_type hexaId) const;

    index_type getHexaInFineRegularGrid(const Vec3i& id) const;

    const std::set<Vec3i>& getHexaVoxels(const index_type hexaId) const;

    Data<int> _level; ///< Number of resolution levels between the fine and coarse mesh
    Data<Vec3i>	fineResolution;		///< width, height, depth (number of hexa in each direction)
    Data<helper::vector<index_type> > hexaIndexInRegularGrid; ///< indices of the hexa in the grid.

private:
    void setCoarseResolution(const Vec3i& res) { _coarseResolution = res; }

    void connectionToNodeAdjacency(const Vec3i& connection, std::map<index_type, index_type>& nodeMap) const;

    class SOFA_SOFANONUNIFORMFEM_API Component
    {
    public:
        Component(const Vec3i& id, const std::set<Vec3i>& voxels);
        virtual ~Component();

        bool isEmpty() const;

        bool isConnected(const Component* other) const;
        bool getConnection(const Component* other, Vec3i& connection) const;
        bool merge(Component* other);

        void split(std::set<Component*>& newComponents);

        void clear();
        void removeVoxels(const std::set<Vec3i>& voxels);

        bool hasVoxel(const Vec3i& voxel) const;

        const Vec3i& getVoxelId() const {return _id;}

        bool isStronglyConnected() const;

        int getLevel() const;

        inline friend std::ostream& operator<< (std::ostream& out, const Component* /*t*/)
        {
            return out;
        }

        inline friend std::istream& operator>>(std::istream& in, Component* /*t*/)
        {
            return in;
        }

    private:
        Component(const Vec3i& id);
        bool isConnected(const std::set<Vec3i>&, const Vec3i&) const;

    public:
        Component*				_parent;
        std::set<Component*>	_children;
        std::set<Vec3i>			_voxels;

    private:
        Vec3i					_id;		// voxel id in the corresponding level
    };

private:


    std::list<const core::topology::TopologyChange *>	m_changeListFine;

    HexahedronData<sofa::helper::vector<Component*> >		_coarseComponents;	///< map between hexahedra and components - coarse
    HexahedronData<sofa::helper::vector<Component*> >		_fineComponents;	///< map between hexahedra and components - fine

    // the fine mesh must be a regular grid - store its parameters here

    Vec3i	_coarseResolution;

    sofa::helper::vector<Component*>	_fineComponentInRegularGrid;
};

/** notifies change in the multilevel structure other than adding or removing coarse hexahedra */
class SOFA_SOFANONUNIFORMFEM_API MultilevelModification : public core::topology::TopologyChange
{
public:
    static const int MULTILEVEL_MODIFICATION = core::topology::TOPOLOGYCHANGE_LASTID + 1;

    typedef defaulttype::Vec<3,int>	Vec3i;
    using index_type = sofa::defaulttype::index_type;

    MultilevelModification(const sofa::helper::vector<index_type>& _tArray,
            const std::map<index_type, std::list<Vec3i> >& removedVoxels)
        : core::topology::TopologyChange((core::topology::TopologyChangeType) MULTILEVEL_MODIFICATION)
        , _modifiedHexahedraArray(_tArray)
        , _removedFineVoxels(removedVoxels)
    {}

    const sofa::helper::vector<index_type> &getArray() const
    {
        return _modifiedHexahedraArray;
    }

    const std::list<Vec3i> &getRemovedVoxels(const index_type hexaId) const
    {
        auto it = _removedFineVoxels.find(hexaId);
        if(it != _removedFineVoxels.end())
            return it->second;
        else
            return __dummyList;
    }

    size_t getNbModifiedHexahedra() const
    {
        return _modifiedHexahedraArray.size();
    }

private:
    sofa::helper::vector<index_type>		_modifiedHexahedraArray;
    std::map<index_type, std::list<Vec3i> > _removedFineVoxels;

    const std::list<Vec3i>	__dummyList;
};

} // namespace sofa::component::topology

