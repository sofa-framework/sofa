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

#include <SofaNonUniformFem/MultilevelHexahedronSetTopologyContainer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/loader/VoxelLoader.h>

#include <SofaBaseTopology/TopologyData.inl>

#include <set>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MultilevelHexahedronSetTopologyContainer)
int MultilevelHexahedronSetTopologyContainerClass = core::RegisterObject("Hexahedron set topology container")
        .add< MultilevelHexahedronSetTopologyContainer >()
        ;

MultilevelHexahedronSetTopologyContainer::MultilevelHexahedronSetTopologyContainer()
    : HexahedronSetTopologyContainer(),
      _level(initData(&_level, 0, "level", "Number of resolution levels between the fine and coarse mesh")),
      fineResolution(initData(&fineResolution,Vec3i(0,0,0),"resolution","fine resolution")),
      hexaIndexInRegularGrid(initData(&hexaIndexInRegularGrid,"idxInRegularGrid","indices of the hexa in the grid.")),
      _coarseComponents(initData(&_coarseComponents,"coarseComponents", "map between hexahedra and components - coarse")),
      _fineComponents(initData(&_fineComponents,"fineComponents", "map between hexahedra and components - fine")),
      _coarseResolution(0,0,0)
{ }


MultilevelHexahedronSetTopologyContainer::~MultilevelHexahedronSetTopologyContainer()
{
    clear();
}

void MultilevelHexahedronSetTopologyContainer::init()
{
    const Vec3i& _fineResolution = fineResolution.getValue();

    _fineComponentInRegularGrid.resize(_fineResolution[0] * _fineResolution[1] * _fineResolution[2], NULL);

    const size_t numVoxels = d_hexahedron.getValue().size();

    // initialize the components
    // at the beginning the components of both levels are the same
    _coarseResolution = _fineResolution;

    _fineComponents.beginEdit()->reserve(numVoxels);
    _fineComponents.endEdit();
    _coarseComponents.beginEdit()->reserve(numVoxels);
    _coarseComponents.endEdit();
    const helper::vector<unsigned int>&  _hexaIndexInRegularGrid= hexaIndexInRegularGrid.getValue();
    for(unsigned int idx=0; idx<numVoxels; ++idx)
    {
        //	id = k * nx * ny + j * nx + i;
        const unsigned int id = _hexaIndexInRegularGrid[idx];
        const unsigned int i = id % _fineResolution[0];
        const unsigned int j = (id / _fineResolution[0]) % _fineResolution[1];
        const unsigned int k = id / (_fineResolution[0] * _fineResolution[1]);

        Vec3i pos(i,j,k);
        std::set<Vec3i> voxels;
        voxels.insert(&pos, 1+&pos); // Using this insert method avoid the 39390 bug of gcc-4.4

        MultilevelHexahedronSetTopologyContainer::Component *comp = new MultilevelHexahedronSetTopologyContainer::Component(pos, voxels);
        _fineComponents.beginEdit()->push_back(comp);
        _fineComponents.endEdit();
        _coarseComponents.beginEdit()->push_back(comp);
        _coarseComponents.endEdit();

        _fineComponentInRegularGrid[id] = comp;
    }

    HexahedronSetTopologyContainer::init();

    _coarseComponents.createTopologicalEngine(this);
    _fineComponents.createTopologicalEngine(this);
    // do not register these engines for now...

}



void MultilevelHexahedronSetTopologyContainer::clear()
{
    HexahedronSetTopologyContainer::clear();

    // this will delete all coarse components and their children,
    // i.e., all fine components will be deleted as well
    for(unsigned int i=0; i<_coarseComponents.getValue().size(); ++i)
        delete _coarseComponents.getValue()[i];

    _fineComponents.beginEdit()->clear();
    _fineComponents.endEdit();
    _coarseComponents.beginEdit()->clear();
    _coarseComponents.endEdit();

    _level = 0;

    fineResolution.setValue(Vec3i(0,0,0));
    _coarseResolution = Vec3i(0,0,0);

    _fineComponentInRegularGrid.clear();
}

void MultilevelHexahedronSetTopologyContainer::getHexaNeighbors(const unsigned int hexa,
        helper::vector<unsigned int> &neighbors)
{
    std::set<unsigned int>	uniqueNeighbors;
    for(int vertexId=0; vertexId<8; ++vertexId)
    {
        helper::vector<unsigned int> vneighbors;

        getHexaVertexNeighbors(hexa, vertexId, vneighbors);

        for(unsigned int i=0; i<vneighbors.size(); ++i)
            uniqueNeighbors.insert(vneighbors[i]);
    }

    neighbors.reserve(uniqueNeighbors.size());
    for(std::set<unsigned int>::const_iterator iter = uniqueNeighbors.begin(); iter != uniqueNeighbors.end(); ++iter)
        neighbors.push_back(*iter);
}

void MultilevelHexahedronSetTopologyContainer::getHexaFaceNeighbors(const unsigned int hexa,
        const unsigned int faceId,
        helper::vector<unsigned int> &neighbors)
{
    const QuadsInHexahedron &hexaQuads = getQuadsInHexahedron(hexa);
    const sofa::helper::vector< unsigned int > &quadShell = getHexahedraAroundQuad(hexaQuads[faceId]);

    neighbors.clear();
    neighbors.reserve(quadShell.size()-1);
    for(unsigned int i=0; i<quadShell.size(); ++i)
    {
        if(quadShell[i] != hexa)
            neighbors.push_back(quadShell[i]);
    }
}

void MultilevelHexahedronSetTopologyContainer::getHexaVertexNeighbors(const unsigned int hexa,
        const unsigned int vertexId,
        helper::vector<unsigned int> &neighbors)
{
    helper::ReadAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;
    const sofa::helper::vector< unsigned int > &vertexShell = getHexahedraAroundVertex(m_hexahedron[hexa][vertexId]);

    neighbors.clear();
    neighbors.reserve(vertexShell.size()-1);
    for(unsigned int i=0; i<vertexShell.size(); ++i)
    {
        if(vertexShell[i] != hexa)
            neighbors.push_back(vertexShell[i]);
    }
}

bool MultilevelHexahedronSetTopologyContainer::getHexaContainsPosition(const unsigned int hexaId,
        const defaulttype::Vector3& baryC) const
{
    const Component& comp = *_coarseComponents.getValue()[hexaId];
    const Vec3i& coarseVoxelId = comp.getVoxelId();
    const unsigned int coarseVoxelSize = 1 << _level.getValue();

    const unsigned int coarse_i = coarseVoxelId[0];
    const unsigned int coarse_j = coarseVoxelId[1];
    const unsigned int coarse_k = coarseVoxelId[2];

    const float epsilon = 0.001f;

    Vector3 voxelRealMin((coarse_i + baryC[0] - epsilon) * coarseVoxelSize,
            (coarse_j + baryC[1] - epsilon) * coarseVoxelSize,
            (coarse_k + baryC[2] - epsilon) * coarseVoxelSize);

    Vector3 voxelRealMax((coarse_i + baryC[0] + epsilon) * coarseVoxelSize,
            (coarse_j + baryC[1] + epsilon) * coarseVoxelSize,
            (coarse_k + baryC[2] + epsilon) * coarseVoxelSize);

    Vec3i voxelIntMin((int) voxelRealMin[0], (int) voxelRealMin[1], (int) voxelRealMin[2]);
    Vec3i voxelIntMax((int) voxelRealMax[0], (int) voxelRealMax[1], (int) voxelRealMax[2]);

    for(int i=voxelIntMin[0]; i<=voxelIntMax[0]; ++i)
        for(int j=voxelIntMin[1]; j<=voxelIntMax[1]; ++j)
            for(int k=voxelIntMin[2]; k<=voxelIntMax[2]; ++k)
            {
                if(comp.hasVoxel(Vec3i(i,j,k)))
                    return true;
            }

    return false;
}

const MultilevelHexahedronSetTopologyContainer::Vec3i& MultilevelHexahedronSetTopologyContainer::getHexaIdxInCoarseRegularGrid(const unsigned int hexaId) const
{
    return _coarseComponents.getValue()[hexaId]->getVoxelId();
}

int MultilevelHexahedronSetTopologyContainer::getHexaIdInCoarseRegularGrid(const unsigned int hexaId) const
{
    const Vec3i& voxelId = getHexaIdxInCoarseRegularGrid(hexaId);
    return voxelId[0] + _coarseResolution[0] * (voxelId[1]  + voxelId[2] * _coarseResolution[1]);
}

const MultilevelHexahedronSetTopologyContainer::Vec3i& MultilevelHexahedronSetTopologyContainer::getHexaIdxInFineRegularGrid(const unsigned int hexaId) const
{
    return _fineComponents.getValue()[hexaId]->getVoxelId();
}

int MultilevelHexahedronSetTopologyContainer::getHexaIdInFineRegularGrid(const unsigned int hexaId) const
{
    const Vec3i& voxelId = getHexaIdxInFineRegularGrid(hexaId);
    const Vec3i& _fineResolution = fineResolution.getValue();
    return voxelId[0] + _fineResolution[0] * (voxelId[1]  + voxelId[2] * _fineResolution[1]);
}

int MultilevelHexahedronSetTopologyContainer::getHexaInFineRegularGrid(const Vec3i& voxelId) const
{
    const Vec3i& _fineResolution = fineResolution.getValue();
    Component* comp = _fineComponentInRegularGrid[voxelId[0] + _fineResolution[0] * (voxelId[1]  + voxelId[2] * _fineResolution[1])];
    if(comp != NULL)
    {
        for(unsigned int i=0; i<_fineComponents.getValue().size(); ++i)
        {
            if(_fineComponents.getValue()[i] == comp)
                return i;
        }
    }

    return -1;
}

int MultilevelHexahedronSetTopologyContainer::getHexaChildren(const unsigned int hexaId,
        helper::vector<unsigned int>& children) const
{
    std::list<Component*>	compList;
    compList.push_back(_coarseComponents.getValue()[hexaId]);

    Component* comp = compList.front();
    while(!comp->_children.empty())
    {
        for(std::set<Component*>::iterator iter = comp->_children.begin();
            iter != comp->_children.end(); ++iter)
        {
            compList.push_back(*iter);
        }

        compList.pop_front();
        comp = compList.front();
    }

    std::set<Component*> compSet;
    compSet.insert(compList.begin(), compList.end());

    children.reserve(compSet.size());

    for(unsigned int i=0; i<_fineComponents.getValue().size(); ++i)
    {
        if(compSet.find(_fineComponents.getValue()[i]) != compSet.end())
            children.push_back(i);
    }

    return (int) children.size();
}

const std::set<MultilevelHexahedronSetTopologyContainer::Vec3i>& MultilevelHexahedronSetTopologyContainer::getHexaVoxels(const unsigned int hexaId) const
{
    return _coarseComponents.getValue()[hexaId]->_voxels;
}

int MultilevelHexahedronSetTopologyContainer::getHexaParent(const unsigned int hexaId) const
{
    Component* comp = _fineComponents.getValue()[hexaId];

    while( comp->_parent != NULL)
        comp = comp->_parent;

    for(unsigned int i=0; i<_coarseComponents.getValue().size(); ++i)
    {
        if(_coarseComponents.getValue()[i] == comp)
            return i;
    }

    serr << "ERROR(MultilevelHexahedronSetTopologyContainer) No hexa parent found. \n";
    return 0;
}

void MultilevelHexahedronSetTopologyContainer::connectionToNodeAdjacency(const Vec3i& connection,
        std::map<unsigned int, unsigned int>& nodeMap) const
{
    for(unsigned int i=0; i<8; ++i)
        nodeMap[i] = i;

    if(connection[0] == 1) // cube is on the right from neighbor
    {
        nodeMap.erase(1); nodeMap.erase(2); nodeMap.erase(5); nodeMap.erase(6);
    }
    else if(connection[0] == -1) // cube is on the left from neighbor
    {
        nodeMap.erase(0); nodeMap.erase(3); nodeMap.erase(4); nodeMap.erase(7);
    }

    if(connection[1] == 1) // cube is on the top from neighbor
    {
        nodeMap.erase(3); nodeMap.erase(2); nodeMap.erase(7); nodeMap.erase(6);
    }
    else if(connection[1] == -1) // cube is on the bottom from neighbor
    {
        nodeMap.erase(0); nodeMap.erase(1); nodeMap.erase(4); nodeMap.erase(5);
    }

    if(connection[2] == 1) // cube is on the front from neighbor
    {
        nodeMap.erase(4); nodeMap.erase(5); nodeMap.erase(6); nodeMap.erase(7);
    }
    else if(connection[2] == -1) // cube is on the back from neighbor
    {
        nodeMap.erase(0); nodeMap.erase(1); nodeMap.erase(2); nodeMap.erase(3);
    }

    if(nodeMap.size() == 4) // face connection
    {
        if((connection - Vec3i(1,0,0)).norm2() == 0) // cube is on the right from neighbor
        {
            nodeMap[0] = 1; nodeMap[3] = 2; nodeMap[4] = 5; nodeMap[7] = 6;
        }
        else if((connection - Vec3i(-1,0,0)).norm2() == 0) // cube is on the left from neighbor
        {
            nodeMap[1] = 0; nodeMap[2] = 3; nodeMap[5] = 4; nodeMap[6] = 7;
        }
        else if((connection - Vec3i(0,1,0)).norm2() == 0) // cube is on the top from neighbor
        {
            nodeMap[0] = 3; nodeMap[1] = 2; nodeMap[4] = 7; nodeMap[5] = 6;
        }
        else if((connection - Vec3i(0,-1,0)).norm2() == 0) // cube is on the bottom from neighbor
        {
            nodeMap[3] = 0; nodeMap[2] = 1; nodeMap[7] = 4; nodeMap[6] = 5;
        }
        else if((connection - Vec3i(0,0,1)).norm2() == 0) // cube is on the front from neighbor
        {
            nodeMap[0] = 4; nodeMap[1] = 5; nodeMap[2] = 6; nodeMap[3] = 7;
        }
        else if((connection - Vec3i(0,0,-1)).norm2() == 0) // cube is on the back from neighbor
        {
            nodeMap[4] = 0; nodeMap[5] = 1; nodeMap[6] = 2; nodeMap[7] = 3;
        }
    }
    else if(nodeMap.size() == 2) // edge connection
    {
        switch(nodeMap.begin()->first)
        {
        case 0:
            switch(nodeMap.rbegin()->first)
            {
            case 1: nodeMap[0] = 7; nodeMap[1] = 6;	break;
            case 3: nodeMap[0] = 5; nodeMap[3] = 6;	break;
            case 4: nodeMap[0] = 2; nodeMap[4] = 6;	break;
            }
            break;
        case 1:
            switch(nodeMap.rbegin()->first)
            {
            case 2: nodeMap[1] = 4; nodeMap[2] = 7;	break;
            case 5: nodeMap[1] = 3; nodeMap[5] = 7;	break;
            }
            break;
        case 2:
            switch(nodeMap.rbegin()->first)
            {
            case 3: nodeMap[2] = 5; nodeMap[3] = 4;	break;
            case 6: nodeMap[2] = 0; nodeMap[6] = 4;	break;
            }
            break;
        case 3:
            switch(nodeMap.rbegin()->first)
            {
            case 7:	nodeMap[3] = 1; nodeMap[7] = 5;	break;
            }
            break;
        case 4:
            switch(nodeMap.rbegin()->first)
            {
            case 5:	nodeMap[4] = 3; nodeMap[5] = 2;	break;
            case 7: nodeMap[4] = 1; nodeMap[7] = 2;	break;
            }
            break;
        case 5:
            switch(nodeMap.rbegin()->first)
            {
            case 6:	nodeMap[5] = 0; nodeMap[6] = 3;	break;
            }
            break;
        case 6:
            switch(nodeMap.rbegin()->first)
            {
            case 7:	nodeMap[6] = 1; nodeMap[7] = 0;	break;
            }
            break;
        }
    }
    else if(nodeMap.size() == 1) // vertex connection
    {
        switch(nodeMap.begin()->first)
        {
        case 0: nodeMap[0] = 6; break;
        case 1: nodeMap[1] = 7; break;
        case 2: nodeMap[2] = 4; break;
        case 3: nodeMap[3] = 5; break;
        case 4: nodeMap[4] = 2; break;
        case 5: nodeMap[5] = 3; break;
        case 6: nodeMap[6] = 0; break;
        case 7: nodeMap[7] = 1; break;
        }
    }
}

MultilevelHexahedronSetTopologyContainer::Component::Component(const Vec3i& id)
    : _parent(NULL), _id(id)
{
}

MultilevelHexahedronSetTopologyContainer::Component::Component(const Vec3i& id, const std::set<Vec3i>& voxels)
    : _parent(NULL), _id(id)
{
    // copy the voxels
    this->_voxels.insert(voxels.begin(), voxels.end());
}

MultilevelHexahedronSetTopologyContainer::Component::~Component()
{
    for(std::set<Component*>::iterator iter = _children.begin(); iter != _children.end(); ++iter)
    {
        delete (*iter);
    }
}

void MultilevelHexahedronSetTopologyContainer::Component::clear()
{
    std::set<Vec3i> voxels;
    voxels.insert(this->_voxels.begin(), this->_voxels.end());
    this->_voxels.clear();

    if(_parent != NULL)
    {
        _parent->removeVoxels(voxels);
    }
}

void MultilevelHexahedronSetTopologyContainer::Component::removeVoxels(const std::set<Vec3i>& voxels)
{
    for(std::set<Vec3i>::const_iterator voxelIter = voxels.begin();
        voxelIter != voxels.end(); ++voxelIter)
    {
        this->_voxels.erase(this->_voxels.find(*voxelIter));
    }

    for(std::set<Component*>::iterator iter = this->_children.begin();
        iter != this->_children.end(); /*++iter*/)
    {
        if((*iter)->isEmpty())
        {
            std::set<Component*>::iterator it = iter;
            ++iter;
            delete *it;
            this->_children.erase(it);
        }
        else
            ++iter;
    }

    if(_parent != NULL)
        _parent->removeVoxels(voxels);
}

bool MultilevelHexahedronSetTopologyContainer::Component::hasVoxel(const Vec3i& voxel) const
{
    return (this->_voxels.find(voxel) != this->_voxels.end());
}

bool MultilevelHexahedronSetTopologyContainer::Component::isEmpty() const
{
    return _voxels.empty();
}

int MultilevelHexahedronSetTopologyContainer::Component::getLevel() const
{
    int level = 0;

    Component* comp = (Component*) this;

    while(!comp->_children.empty())
    {
        ++level;
        comp = *comp->_children.begin();
    }

    return level;
}

bool MultilevelHexahedronSetTopologyContainer::Component::isStronglyConnected() const
{
    std::set<Vec3i>	set1;
    std::set<Vec3i> set2;

    set2.insert(_voxels.begin(), _voxels.end());

    bool change = true;

    while(change)
    {
        change = false;

        for(std::set<Vec3i>::iterator iter = set2.begin();
            iter != set2.end(); /*++iter*/)
        {
            if(isConnected(set1, *iter) || set1.empty())
            {
                set1.insert(*iter);

                std::set<Vec3i>::iterator it = iter;
                ++iter;
                set2.erase(it);
                change = true;
            }
            else
            {
                ++iter;
            }
        }
    }

    return set2.empty();
}

bool MultilevelHexahedronSetTopologyContainer::Component::isConnected(const Component* other) const
{
    if((this->_id - other->_id).norm2() > 3)
        return false;

    for(std::set<Vec3i>::const_iterator voxelIter = other->_voxels.begin();
        voxelIter != other->_voxels.end(); ++voxelIter)
    {
        if(isConnected(this->_voxels, *voxelIter))
            return true;
    }

    return false;
}

bool MultilevelHexahedronSetTopologyContainer::Component::isConnected(const std::set<Vec3i>& voxelSet,
        const Vec3i& voxel) const
{
    // check if the two sets contain neighboring voxels
    for(std::set<Vec3i>::const_iterator voxelIter = voxelSet.begin();
        voxelIter != voxelSet.end(); ++voxelIter)
    {
        const Vec3i diff(*voxelIter-voxel);

        if(diff.norm2() < 4)
            return true;
    }

    return false;
}

bool MultilevelHexahedronSetTopologyContainer::Component::getConnection(const Component* other,
        Vec<3, int>& connection) const
{
    if((this->_id - other->_id).norm2() > 3)
        return false;

    const int level = this->getLevel();
    const int size = 1 << level;

    // check if the two components contain neighboring voxels
    for(std::set<Vec3i>::const_iterator voxelIter = this->_voxels.begin();
        voxelIter != this->_voxels.end(); ++voxelIter)
    {
        const Vec3i& voxel1 = (*voxelIter);

        if((voxel1[0] % size > 0) && (voxel1[1] % size > 0) && (voxel1[2] % size > 0)
           && (voxel1[0] % size < size -1) && (voxel1[1] % size < size - 1) && (voxel1[2] % size < size - 1))
            continue;

        for(std::set<Vec3i>::const_iterator voxelIter2 = other->_voxels.begin();
            voxelIter2 != other->_voxels.end(); ++voxelIter2)
        {
            const Vec3i& voxel2 = (*voxelIter2);

            if((voxel2[0] % size > 0) && (voxel2[1] % size > 0) && (voxel2[2] % size > 0)
               && (voxel2[0] % size < size -1) && (voxel2[1] % size < size - 1) && (voxel2[2] % size < size - 1))
                continue;

            const Vec3i diff(voxel1 - voxel2);
            const int diffNorm2 = diff.norm2();

            if(diffNorm2 == 1) // face connection
            {
                connection = diff;
                return true;
            }
            else if(diffNorm2 == 2) // edge connection iff the connecting edge is also an edge of the top level comp.
            {
                if(level == 0)
                {
                    connection = diff;
                }
                else
                {
                    Vec3i d0, d1;
                    d0[0] = (diff[0] == 1) ? 1 : 0;
                    d0[1] = (diff[1] == 1) ? 1 : 0;
                    d0[2] = (diff[2] == 1) ? 1 : 0;

                    d1[0] = (diff[0] == 1) ? 1 : ((diff[0] == 0) ? 1 : 0);
                    d1[1] = (diff[1] == 1) ? 1 : ((diff[1] == 0) ? 1 : 0);
                    d1[2] = (diff[2] == 1) ? 1 : ((diff[2] == 0) ? 1 : 0);

                    Vec3i pnt0((*voxelIter2) + d0);
                    pnt0[0] = pnt0[0] % size;
                    pnt0[1] = pnt0[1] % size;
                    pnt0[2] = pnt0[2] % size;

                    pnt0[0] = (pnt0[0] == 0) ? 1 : 0;
                    pnt0[1] = (pnt0[1] == 0) ? 1 : 0;
                    pnt0[2] = (pnt0[2] == 0) ? 1 : 0;

                    Vec3i pnt1((*voxelIter2) + d1);
                    pnt1[0] = pnt1[0] % size;
                    pnt1[1] = pnt1[1] % size;
                    pnt1[2] = pnt1[2] % size;

                    pnt1[0] = (pnt1[0] == 0) ? 1 : 0;
                    pnt1[1] = (pnt1[1] == 0) ? 1 : 0;
                    pnt1[2] = (pnt1[2] == 0) ? 1 : 0;

                    connection[0] = diff[0] * pnt0[0] * pnt1[0];
                    connection[1] = diff[1] * pnt0[1] * pnt1[1];
                    connection[2] = diff[2] * pnt0[2] * pnt1[2];
                }
                return true;
            }
            else if(diffNorm2 == 3) // vertex connection iff the connecting vertex is also a vertex of the top level comp.
            {
                if(level == 0)
                {
                    connection = diff;
                }
                else
                {
                    Vec3i d;
                    d[0] = (diff[0] == 1) ? 1 : 0;
                    d[1] = (diff[1] == 1) ? 1 : 0;
                    d[2] = (diff[2] == 1) ? 1 : 0;

                    Vec3i pnt((*voxelIter2) + d);
                    pnt[0] = pnt[0] % size;
                    pnt[1] = pnt[1] % size;
                    pnt[2] = pnt[2] % size;

                    pnt[0] = (pnt[0] == 0) ? 1 : 0;
                    pnt[1] = (pnt[1] == 0) ? 1 : 0;
                    pnt[2] = (pnt[2] == 0) ? 1 : 0;

                    connection[0] = diff[0] * pnt[0];
                    connection[1] = diff[1] * pnt[1];
                    connection[2] = diff[2] * pnt[2];
                }
                return true;
            }
        }
    }

    return false;
}

bool MultilevelHexahedronSetTopologyContainer::Component::merge(Component* other)
{
    if((this->_id - other->_id).norm2() != 0 || this->_parent != other->_parent)
        return false;

    for(std::set<Component*>::const_iterator it = other->_children.begin();
        it != other->_children.end(); ++it)
    {
        (*it)->_parent = this;
    }

    // merge children and voxels
    this->_children.insert(other->_children.begin(), other->_children.end());
    this->_voxels.insert(other->_voxels.begin(), other->_voxels.end());

    other->_children.clear();
    other->_voxels.clear();

    return true;
}

void MultilevelHexahedronSetTopologyContainer::Component::split(std::set<Component*>& newComponents)
{
    // split all children
    if(!this->_children.empty())
    {
        std::set<Component*> newChildren;
        for(std::set<Component*>::iterator iter = this->_children.begin();
            iter != this->_children.end(); /* ++iter */)
        {
            if(!(*iter)->isStronglyConnected())
            {
                (*iter)->split(newChildren);
                std::set<Component*>::iterator it = iter;
                ++iter;
                this->_children.erase(it);
            }
            else
                ++iter;
        }

        if(!newChildren.empty())
        {
            this->_children.insert(newChildren.begin(), newChildren.end());
        }
    }

    // find connectivity components
    // create new Components with corresponding voxels and children
    if(!this->isStronglyConnected())
    {
        std::list<Component*> set1;
        std::list<Component*> set2;

        set2.insert(set2.end(), _children.begin(), _children.end());

        while(!set2.empty())
        {
            bool change = true;

            // init set1 with the first element of set2
            set1.push_back(*set2.begin());
            set2.erase(set2.begin());

            while(change)
            {
                change = false;

                for(std::list<Component*>::iterator iter = set1.begin();
                    iter != set1.end(); ++iter)
                    for(std::list<Component*>::iterator iter2 = set2.begin();
                        iter2 != set2.end(); /*++iter2*/)
                    {
                        if((*iter2)->isConnected(*iter))
                        {
                            set1.push_back(*iter2);
                            std::list<Component*>::iterator it = iter2;
                            ++iter2;
                            set2.erase(it);
                            change = true;
                            break;
                        }
                        else
                        {
                            ++iter2;
                        }
                    }
            }

            Component*	comp = new Component(this->getVoxelId());
            comp->_parent = this->_parent;
            comp->_children.insert(set1.begin(), set1.end());
            set1.clear();

            for(std::set<Component*>::iterator iter = comp->_children.begin();
                iter != comp->_children.end(); ++iter)
            {
                comp->_voxels.insert((*iter)->_voxels.begin(), (*iter)->_voxels.end());

                (*iter)->_parent = comp;
            }

            newComponents.insert(comp);
        }

        this->_children.clear();
    }
}

} // namespace topology

} // namespace component

} // namespace sofa

