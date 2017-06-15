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
#ifndef SOFA_COMPONENT_COLLISION_CUBEMODEL_H
#define SOFA_COMPONENT_COLLISION_CUBEMODEL_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace collision
{

class CubeModel;

class Cube : public core::TCollisionElementIterator<CubeModel>
{
public:
    Cube(CubeModel* model=NULL, int index=0);

    explicit Cube(const core::CollisionElementIterator& i);

    const sofa::defaulttype::Vector3& minVect() const;

    const sofa::defaulttype::Vector3& maxVect() const;

    const std::pair<Cube,Cube>& subcells() const;
};

class SOFA_BASE_COLLISION_API CubeModel : public core::CollisionModel
{
public:
    SOFA_CLASS(CubeModel,sofa::core::CollisionModel);

    struct CubeData
    {
        sofa::defaulttype::Vector3 minBBox, maxBBox;
        std::pair<Cube,Cube> subcells;
        std::pair<core::CollisionElementIterator,core::CollisionElementIterator> children; ///< Note that children is only meaningfull if subcells in empty
    };

    class CubeSortPredicate
    {
        int axis;
    public:
        CubeSortPredicate(int axis) : axis(axis) {}
        bool operator()(const CubeData& c1,const CubeData& c2) const
        {
            SReal v1 = c1.minBBox[axis]+c1.maxBBox[axis];
            SReal v2 = c2.minBBox[axis]+c2.maxBBox[axis];
            return v1 < v2;
        }
        template<int Axis>
        static int sortCube(const void* p1, const void* p2)
        {
            const CubeModel::CubeData* c1 = (const CubeModel::CubeData*)p1;
            const CubeModel::CubeData* c2 = (const CubeModel::CubeData*)p2;
            SReal v1 = c1->minBBox[Axis] + c1->maxBBox[Axis];
            SReal v2 = c2->minBBox[Axis] + c2->maxBBox[Axis];

            if (v1 < v2)
                return -1;
            else if (v1 > v2)
                return 1;
            else
                return 0;
        }
    };

protected:

    //class CubeSortPredicate;

    sofa::helper::vector<CubeData> elems;
    sofa::helper::vector<int> parentOf; ///< Given the index of a child leaf element, store the index of the parent cube

public:
    typedef core::CollisionElementIterator ChildIterator;
    typedef sofa::defaulttype::Vec3Types DataTypes;
    typedef Cube Element;
    friend class Cube;
protected:
    CubeModel();
public:
    virtual void resize(int size);

    void setParentOf(int childIndex, const sofa::defaulttype::Vector3& min, const sofa::defaulttype::Vector3& max);
    void setLeafCube(int cubeIndex, int childIndex);
    void setLeafCube(int cubeIndex, std::pair<core::CollisionElementIterator,core::CollisionElementIterator> children, const sofa::defaulttype::Vector3& min, const sofa::defaulttype::Vector3& max);


    unsigned int getNumberCells() { return (unsigned int)elems.size();}

    void getBoundingTree ( sofa::helper::vector< std::pair< sofa::defaulttype::Vector3, sofa::defaulttype::Vector3> > &bounding )
    {
        bounding.resize(elems.size());
        for (unsigned int index=0; index<elems.size(); index++)
        {
            bounding[index] = std::make_pair( elems[index].minBBox, elems[index].maxBBox);
        }
    }

    int getLeafIndex(int index) const
    {
        return elems[index].children.first.getIndex();
    }

    int getLeafEndIndex(int index) const
    {
        return elems[index].children.second.getIndex();
    }

    const CubeData & getCubeData(int index)const{return elems[index];}

    // -- CollisionModel interface

    /**
      *Here we make up the hierarchy (a tree) of bounding boxes which contain final CollisionElements like Spheres or Triangles.
      *The leafs of the tree contain final CollisionElements. This hierarchy is made up from the top to the bottom, i.e., we begin
      *to compute a bounding box containing all CollisionElements, then we divide this big bounding box into two boxes.
      *These new two boxes inherit from the root box and have depth 1. Then we can do the same operation for the new boxes.
      *The division is done only if the box contains more than 4 final CollisionElements and if the depth doesn't exceed
      *the max depth. The division is made along an axis. This axis corresponds to the biggest dimension of the current bounding box.
      *Note : a bounding box is a Cube here.
      */
    virtual void computeBoundingTree(int maxDepth=0);

    virtual std::pair<core::CollisionElementIterator,core::CollisionElementIterator> getInternalChildren(int index) const;

    virtual std::pair<core::CollisionElementIterator,core::CollisionElementIterator> getExternalChildren(int index) const;

    virtual bool isLeaf( int index ) const;

    void draw(const core::visual::VisualParams* vparams);

    int addCube(Cube subcellsBegin, Cube subcellsEnd);
    void updateCube(int index);
    void updateCubes();
};

inline Cube::Cube(CubeModel* model, int index)
    : core::TCollisionElementIterator<CubeModel>(model, index)
{}

inline Cube::Cube(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<CubeModel>(static_cast<CubeModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const sofa::defaulttype::Vector3& Cube::minVect() const
{
    return model->elems[index].minBBox;
}

inline const sofa::defaulttype::Vector3& Cube::maxVect() const
{
    return model->elems[index].maxBBox;
}


inline const std::pair<Cube,Cube>& Cube::subcells() const
{
    return model->elems[index].subcells;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
