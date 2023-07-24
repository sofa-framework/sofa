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
#include <sofa/component/collision/geometry/config.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::collision::geometry
{

class CubeCollisionModel;

class Cube : public core::TCollisionElementIterator<CubeCollisionModel>
{
public:
    Cube(CubeCollisionModel* model=nullptr, sofa::Index index=0);

    explicit Cube(const core::CollisionElementIterator& i);

    const sofa::type::Vec3& minVect() const;

    const sofa::type::Vec3& maxVect() const;

    const std::pair<Cube,Cube>& subcells() const;

    SReal getConeAngle() const;
    
    const sofa::type::Vec3& getConeAxis() const;
};

class SOFA_COMPONENT_COLLISION_GEOMETRY_API CubeCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(CubeCollisionModel,sofa::core::CollisionModel);

    struct CubeData
    {
        sofa::type::Vec3 minBBox, maxBBox;
        std::pair<Cube,Cube> subcells;
        std::pair<core::CollisionElementIterator,core::CollisionElementIterator> children; ///< Note that children is only meaningfull if subcells in empty

        // additional datas for implementing Volino's method for efficient cloth self collision 
        sofa::type::Vec3 coneAxis;
        SReal coneAngle;
    };

    class CubeSortPredicate
    {
        int axis;
    public:
        CubeSortPredicate(int axis) : axis(axis) {}
        bool operator()(const CubeData& c1,const CubeData& c2) const
        {
            const SReal v1 = c1.minBBox[axis]+c1.maxBBox[axis];
            const SReal v2 = c2.minBBox[axis]+c2.maxBBox[axis];
            return v1 < v2;
        }
    };

protected:
    sofa::type::vector<CubeData> elems;
    sofa::type::vector<sofa::Index> parentOf; ///< Given the index of a child leaf element, store the index of the parent cube

public:
    typedef core::CollisionElementIterator ChildIterator;
    typedef sofa::defaulttype::Vec3Types DataTypes;
    typedef Cube Element;
    friend class Cube;
protected:
    CubeCollisionModel();
public:
    void resize(sofa::Size size) override;

    void setParentOf(sofa::Index childIndex, const sofa::type::Vec3& min, const sofa::type::Vec3& max);
    void setParentOf(sofa::Index childIndex, const sofa::type::Vec3& min, const sofa::type::Vec3& max, const sofa::type::Vec3& normal, const SReal angle=0);
    void setLeafCube(sofa::Index cubeIndex, sofa::Index childIndex);
    void setLeafCube(sofa::Index cubeIndex, std::pair<core::CollisionElementIterator,core::CollisionElementIterator> children, const sofa::type::Vec3& min, const sofa::type::Vec3& max);

    sofa::Size getNumberCells() const { return sofa::Size(elems.size());}

    void getBoundingTree ( sofa::type::vector< std::pair< sofa::type::Vec3, sofa::type::Vec3> > &bounding )
    {
        bounding.resize(elems.size());
        for (sofa::Size index=0; index<elems.size(); index++)
        {
            bounding[index] = std::make_pair( elems[index].minBBox, elems[index].maxBBox);
        }
    }

    sofa::Index getLeafIndex(sofa::Index index) const
    {
        return elems[index].children.first.getIndex();
    }

    sofa::Index getLeafEndIndex(sofa::Index index) const
    {
        return elems[index].children.second.getIndex();
    }

    const CubeData & getCubeData(sofa::Index index)const{return elems[index];}

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
    void computeBoundingTree(int maxDepth=0) override;

    std::pair<core::CollisionElementIterator,core::CollisionElementIterator> getInternalChildren(sofa::Index index) const override;

    std::pair<core::CollisionElementIterator,core::CollisionElementIterator> getExternalChildren(sofa::Index index) const override;

    bool isLeaf(sofa::Index index ) const override;

    void draw(const core::visual::VisualParams* vparams) override;

    sofa::Index addCube(Cube subcellsBegin, Cube subcellsEnd);
    void updateCube(sofa::Index index);
    void updateCubes();
};

inline Cube::Cube(CubeCollisionModel* model, Index index)
    : core::TCollisionElementIterator<CubeCollisionModel>(model, index)
{}

inline Cube::Cube(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<CubeCollisionModel>(static_cast<CubeCollisionModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const sofa::type::Vec3& Cube::minVect() const
{
    return model->elems[index].minBBox;
}

inline const sofa::type::Vec3& Cube::maxVect() const
{
    return model->elems[index].maxBBox;
}


inline const std::pair<Cube,Cube>& Cube::subcells() const
{
    return model->elems[index].subcells;
}


inline SReal Cube::getConeAngle() const
{
    return model->elems[index].coneAngle;
}


inline const sofa::type::Vec3& Cube::getConeAxis() const
{
    return model->elems[index].coneAxis;
}

} // namespace sofa::component::collision::geometry
