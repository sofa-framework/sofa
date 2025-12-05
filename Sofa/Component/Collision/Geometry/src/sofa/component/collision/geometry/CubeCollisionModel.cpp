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
#include <sofa/component/collision/geometry/CubeCollisionModel.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/visual/DrawTool.h>
#include <sofa/core/ObjectFactory.h>
#include <algorithm>

namespace sofa::component::collision::geometry
{

using namespace sofa::type;
using namespace sofa::defaulttype;

void registerCubeCollisionModel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Collision model representing a cube.")
        .add< CubeCollisionModel >());
}

CubeCollisionModel::CubeCollisionModel()
{
    enum_type = AABB_TYPE;
}

void CubeCollisionModel::resize(sofa::Size size)
{
    const auto size0 = this->size;
    if (size == size0) return;
    // reset parent
    CollisionModel* parent = getPrevious();
    while(parent != nullptr)
    {
        parent->resize(0);
        parent = parent->getPrevious();
    }
    this->core::CollisionModel::resize(size);
    this->elems.resize(size);
    this->parentOf.resize(size);
    // set additional indices
    for (sofa::Size i=size0; i<size; ++i)
    {
        this->elems[i].children.first=core::CollisionElementIterator(getNext(), i);
        this->elems[i].children.second=core::CollisionElementIterator(getNext(), i+1);
        this->parentOf[i] = i;
    }
}

void CubeCollisionModel::setParentOf(sofa::Index childIndex, const Vec3& min, const Vec3& max)
{
    const sofa::Index i = parentOf[childIndex];
    elems[i].minBBox = min;
    elems[i].maxBBox = max;
    elems[i].coneAngle = 2*M_PI;
}

void CubeCollisionModel::setParentOf(sofa::Index childIndex, const Vec3& min, const Vec3& max, const Vec3& normal, const SReal angle)
{
    const sofa::Index i = parentOf[childIndex];
    elems[i].minBBox = min;
    elems[i].maxBBox = max;

    elems[i].coneAxis = normal;
    elems[i].coneAngle = angle;
}

void CubeCollisionModel::setLeafCube(sofa::Index cubeIndex, sofa::Index childIndex)
{
    parentOf[childIndex] = cubeIndex;
    this->elems[cubeIndex].children.first=core::CollisionElementIterator(getNext(), childIndex);
    this->elems[cubeIndex].children.second=core::CollisionElementIterator(getNext(), childIndex+1);
    //elems[cubeIndex].minBBox = min;
    //elems[cubeIndex].maxBBox = max;
}

void CubeCollisionModel::setLeafCube(sofa::Index cubeIndex, std::pair<core::CollisionElementIterator,core::CollisionElementIterator> children, const Vec3& min, const Vec3& max)
{
    elems[cubeIndex].minBBox = min;
    elems[cubeIndex].maxBBox = max;
    elems[cubeIndex].children = children;
}

Index CubeCollisionModel::addCube(Cube subcellsBegin, Cube subcellsEnd)
{
    const sofa::Index index = size;

    this->core::CollisionModel::resize(index + 1);
    elems.resize(index + 1);
    
    elems[index].subcells.first = subcellsBegin;
    elems[index].subcells.second = subcellsEnd;
    elems[index].children.first = core::CollisionElementIterator();
    elems[index].children.second = core::CollisionElementIterator();
    updateCube(index);
    return index;
}

void CubeCollisionModel::updateCube(sofa::Index index)
{
    const std::pair<Cube,Cube>& subcells = elems[index].subcells;
    if (subcells.first != subcells.second)
    {
        Cube c = subcells.first;
        Vec3 minBBox = c.minVect();
        Vec3 maxBBox = c.maxVect();

        elems[index].coneAxis = c.getConeAxis();
        elems[index].coneAngle = c.getConeAngle();

        ++c;
        while(c != subcells.second)
        {
            const Vec3& cmin = c.minVect();
            const Vec3& cmax = c.maxVect();

            const SReal alpha = std::max<SReal>(elems[index].coneAngle, c.getConeAngle());
            if(alpha <= M_PI/2)
            {
                const SReal beta = acos(c.getConeAxis() *  elems[index].coneAxis);
                elems[index].coneAxis = (c.getConeAxis() + elems[index].coneAxis).normalized();
                elems[index].coneAngle = beta/2 + alpha;
            }
            else
                elems[index].coneAngle = 2*M_PI;
            
            for (int j=0; j<3; j++)
            {
                if (cmax[j] > maxBBox[j]) maxBBox[j] = cmax[j];
                if (cmin[j] < minBBox[j]) minBBox[j] = cmin[j];
            }
            ++c;
        }
        elems[index].minBBox = minBBox;
        elems[index].maxBBox = maxBBox;
    }
}

void CubeCollisionModel::updateCubes()
{
    for (sofa::Index i=0; i<size; i++)
        updateCube(i);
}

void CubeCollisionModel::drawCollisionModel(const core::visual::VisualParams* vparams)
{
    // The deeper in the CubeModel graph, the higher the transparency of the bounding cube lines  
    const float* collisionColor = getColor4f();
    sofa::type::RGBAColor c(collisionColor[0], collisionColor[1], collisionColor[2], collisionColor[3]);
    CollisionModel* m = getPrevious();
    while (m!=nullptr)
    {
        m = m->getPrevious();
        c.a() *= 0.8f;
    }

    std::vector< Vec3 > points;
    points.reserve( size * 8 * 3);
    for (sofa::Index i=0; i<size; i++)
    {
        const Vec3& vmin = elems[i].minBBox;
        const Vec3& vmax = elems[i].maxBBox;

        points.emplace_back(vmin[0], vmin[1], vmin[2]);
        points.emplace_back(vmin[0], vmin[1], vmax[2]);
        points.emplace_back(vmin[0], vmax[1], vmin[2]);
        points.emplace_back(vmin[0], vmax[1], vmax[2]);
        points.emplace_back(vmax[0], vmin[1], vmin[2]);
        points.emplace_back(vmax[0], vmin[1], vmax[2]);
        points.emplace_back(vmax[0], vmax[1], vmin[2]);
        points.emplace_back(vmax[0], vmax[1], vmax[2]);

        points.emplace_back(vmin[0], vmin[1], vmin[2]);
        points.emplace_back(vmin[0], vmax[1], vmin[2]);
        points.emplace_back(vmin[0], vmin[1], vmax[2]);
        points.emplace_back(vmin[0], vmax[1], vmax[2]);
        points.emplace_back(vmax[0], vmin[1], vmin[2]);
        points.emplace_back(vmax[0], vmax[1], vmin[2]);
        points.emplace_back(vmax[0], vmin[1], vmax[2]);
        points.emplace_back(vmax[0], vmax[1], vmax[2]);

        points.emplace_back(vmin[0], vmin[1], vmin[2]);
        points.emplace_back(vmax[0], vmin[1], vmin[2]);
        points.emplace_back(vmin[0], vmax[1], vmin[2]);
        points.emplace_back(vmax[0], vmax[1], vmin[2]);
        points.emplace_back(vmin[0], vmin[1], vmax[2]);
        points.emplace_back(vmax[0], vmin[1], vmax[2]);
        points.emplace_back(vmin[0], vmax[1], vmax[2]);
        points.emplace_back(vmax[0], vmax[1], vmax[2]);
    }

    vparams->drawTool()->drawLines(points, 1, c);
}

std::pair<core::CollisionElementIterator,core::CollisionElementIterator> CubeCollisionModel::getInternalChildren(sofa::Index index) const
{
    return elems[index].subcells;
}

std::pair<core::CollisionElementIterator,core::CollisionElementIterator> CubeCollisionModel::getExternalChildren(sofa::Index index) const
{
    return elems[index].children;
}

bool CubeCollisionModel::isLeaf(sofa::Index index ) const
{
    return elems[index].children.first.valid();
}

void CubeCollisionModel::computeBoundingTree(int maxDepth)
{

    dmsg_info() << ">CubeCollisionModel::computeBoundingTree(" << maxDepth << ")";
    std::list<CubeCollisionModel*> levels;
    levels.push_front(createPrevious<CubeCollisionModel>());
    for (int i=0; i<maxDepth; i++)
        levels.push_front(levels.front()->createPrevious<CubeCollisionModel>());
    CubeCollisionModel* root = levels.front();

    if (root->empty() || root->getPrevious() != nullptr)
    {
        // Tree must be reconstructed
        dmsg_info() << "Building Tree with depth " << maxDepth << " from " << size << " elements.";
        // First remove extra levels
        while(root->getPrevious()!=nullptr)
        {
            core::CollisionModel::SPtr m = root->getPrevious();
            root->setPrevious(m->getPrevious());
            if (m->getMaster()) m->getMaster()->removeSlave(m);
            //delete m;
            m.reset();
        }

        // Then clear all existing levels
        {
            for (const auto & level : levels)
                level->resize(0);
        }

        // Then build root cell
        dmsg_info() << "CubeCollisionModel: add root cube";
        root->addCube(Cube(this,0),Cube(this,size));
        // Construct tree by splitting cells along their biggest dimension
        auto it = levels.begin();
        CubeCollisionModel* level = *it;
        ++it;
        int lvl = 0;
        while(it != levels.end())
        {
            dmsg_info() << "CubeCollisionModel: split level " << lvl;
            CubeCollisionModel* clevel = *it;
            clevel->elems.reserve(level->size*2);
            for(Cube cell = Cube(level->begin()); level->end() != cell; ++cell)
            {
                const std::pair<Cube,Cube>& subcells = cell.subcells();
                const sofa::Index ncells = subcells.second.getIndex() - subcells.first.getIndex();
                dmsg_info() << "CubeCollisionModel: level " << lvl << " cell " << cell.getIndex() << ": current subcells " << subcells.first.getIndex() << " - " << subcells.second.getIndex();
                if (ncells > 4)
                {
                    // Only split cells with more than 4 childs
                    // Find the biggest dimension
                    int splitAxis;
                    Vec3 l = cell.maxVect()-cell.minVect();
                    const sofa::Index middle = subcells.first.getIndex()+(ncells+1)/2;
                    if(l[0]>l[1])
                        if (l[0]>l[2])
                            splitAxis = 0;
                        else
                            splitAxis = 2;
                    else if (l[1]>l[2])
                        splitAxis = 1;
                    else
                        splitAxis = 2;

                    // Separate cells on each side of the median cell
                    const CubeSortPredicate sortpred(splitAxis);
                    std::stable_sort(elems.begin() + subcells.first.getIndex(), elems.begin() + subcells.second.getIndex(), sortpred);

                    // Create the two new subcells
                    const Cube cmiddle(this, middle);
                    sofa::Index c1 = clevel->addCube(subcells.first, cmiddle);
                    sofa::Index c2 = clevel->addCube(cmiddle, subcells.second);
                    dmsg_info() << "L" << lvl << " cell " << cell.getIndex() << " split along " << (splitAxis == 0 ? 'X' : splitAxis == 1 ? 'Y' : 'Z') << " in cell " << c1 << " size " << middle - subcells.first.getIndex() << " and cell " << c2 << " size " << subcells.second.getIndex() - middle << ".";
                    //level->elems[cell.getIndex()].subcells = std::make_pair(Cube(clevel,c1),Cube(clevel,c2+1));
                    level->elems[cell.getIndex()].subcells.first = Cube(clevel,c1);
                    level->elems[cell.getIndex()].subcells.second = Cube(clevel,c2+1);
                }
            }
            ++it;
            level = clevel;
            ++lvl;
        }
        if (!parentOf.empty())
        {
            // Finally update parentOf to reflect new cell order
            for (sofa::Size i=0; i<size; i++)
                parentOf[elems[i].children.first.getIndex()] = i;
        }
    }
    else
    {
        // Simply update the existing tree, starting from the bottom
        int lvl = 0;
        for (auto it = levels.rbegin(); it != levels.rend(); ++it)
        {
            dmsg_info() << "CubeCollisionModel: update level " << lvl;
            (*it)->updateCubes();
            ++lvl;
        }
    }
    dmsg_info() << "<CubeCollisionModel::computeBoundingTree(" << maxDepth << ")";
}

} // namespace sofa::component::collision::geometry
