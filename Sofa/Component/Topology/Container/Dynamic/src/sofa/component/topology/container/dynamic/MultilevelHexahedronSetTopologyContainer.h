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

#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/type/Vec.h>
#include <set>

namespace sofa::core::topology
{

class TopologyChange;

} // namespace sofa::core::topology

namespace sofa::component::topology::container::dynamic
{

class MultilevelHexahedronSetTopologyModifier;


class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API MultilevelHexahedronSetTopologyContainer : public HexahedronSetTopologyContainer
{
    friend class MultilevelHexahedronSetTopologyModifier;

public:
    SOFA_CLASS(MultilevelHexahedronSetTopologyContainer,HexahedronSetTopologyContainer);

    typedef type::Vec<3,int>			Vec3i;
protected:
    MultilevelHexahedronSetTopologyContainer();

    MultilevelHexahedronSetTopologyContainer(const type::vector< Hexahedron > &hexahedra);

    ~MultilevelHexahedronSetTopologyContainer() override;
public:
    void init() override;

    void clear() override;

    void getHexaNeighbors(const Index hexaId,
            type::vector<Index> &neighbors);

    void getHexaFaceNeighbors(const Index hexaId,
            const Index faceId,
            type::vector<Index> &neighbors);

    void getHexaVertexNeighbors(const Index hexaId,
            const Index vertexId,
            type::vector<Index> &neighbors);

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

    bool getHexaContainsPosition(const Index hexaId, const type::Vec3& baryC) const;

    const Vec3i& getHexaIdxInCoarseRegularGrid(const Index hexaId) const;
    int getHexaIdInCoarseRegularGrid(const Index hexaId) const;

    const Vec3i& getHexaIdxInFineRegularGrid(const Index hexaId) const;
    Index getHexaIdInFineRegularGrid(const Index hexaId) const;

    // gets a vector of fine hexahedra inside a specified coarse hexa
    Index getHexaChildren(const Index hexaId, type::vector<Index>& children) const;

    // gets a coarse hexa for a specified fine hexa
    Index getHexaParent(const Index hexaId) const;

    Index getHexaInFineRegularGrid(const Vec3i& id) const;

    const std::set<Vec3i>& getHexaVoxels(const Index hexaId) const;

    Data<int> _level; ///< Number of resolution levels between the fine and coarse mesh
    Data<Vec3i>	fineResolution;		///< width, height, depth (number of hexa in each direction)
    Data<type::vector<Index> > hexaIndexInRegularGrid; ///< indices of the hexa in the grid.

private:
    void setCoarseResolution(const Vec3i& res) { _coarseResolution = res; }

    void connectionToNodeAdjacency(const Vec3i& connection, std::map<Index, Index>& nodeMap) const;

    class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API Component
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

    core::topology::HexahedronData<sofa::type::vector<Component*> >		_coarseComponents;	///< map between hexahedra and components - coarse
    core::topology::HexahedronData<sofa::type::vector<Component*> >		_fineComponents;	///< map between hexahedra and components - fine

    // the fine mesh must be a regular grid - store its parameters here

    Vec3i	_coarseResolution;

    sofa::type::vector<Component*>	_fineComponentInRegularGrid;
};

/** notifies change in the multilevel structure other than adding or removing coarse hexahedra */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API MultilevelModification : public core::topology::TopologyChange
{
public:
    static const int MULTILEVEL_MODIFICATION = core::topology::TOPOLOGYCHANGE_LASTID + 1;

    typedef type::Vec<3,int>	Vec3i;
    using Index = sofa::Index;

    MultilevelModification(const sofa::type::vector<Index>& _tArray,
            const std::map<Index, std::list<Vec3i> >& removedVoxels)
        : core::topology::TopologyChange((core::topology::TopologyChangeType) MULTILEVEL_MODIFICATION)
        , _modifiedHexahedraArray(_tArray)
        , _removedFineVoxels(removedVoxels)
    {}

    const sofa::type::vector<Index> &getArray() const
    {
        return _modifiedHexahedraArray;
    }

    const std::list<Vec3i> &getRemovedVoxels(const Index hexaId) const
    {
        const auto it = _removedFineVoxels.find(hexaId);
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
    sofa::type::vector<Index>		_modifiedHexahedraArray;
    std::map<Index, std::list<Vec3i> > _removedFineVoxels;

    const std::list<Vec3i>	__dummyList;
};

} // namespace sofa::component::topology::container::dynamic

