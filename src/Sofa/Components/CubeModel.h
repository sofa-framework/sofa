#ifndef SOFA_COMPONENTS_CUBEMODEL_H
#define SOFA_COMPONENTS_CUBEMODEL_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class CubeModel;

class Cube : public Abstract::TCollisionElementIterator<CubeModel>
{
public:
    Cube(CubeModel* model=NULL, int index=0);

    explicit Cube(const Abstract::CollisionElementIterator& i);

    const Vector3& minVect() const;

    const Vector3& maxVect() const;

    const std::pair<Cube,Cube>& subcells() const;
};

class CubeModel : public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:

    struct CubeData
    {
        Vector3 minBBox, maxBBox;
        std::pair<Cube,Cube> subcells;
        Abstract::CollisionElementIterator leaf; ///< Note that leaf is only meaningfull if subcells in empty
    };

    class CubeSortPredicate;

    std::vector<CubeData> elems;
    std::vector<int> parentOf; ///< Given the index of a child leaf element, store the index of the parent cube

    bool static_;
public:
    typedef Abstract::CollisionElementIterator ChildIterator;
    typedef Vec3Types DataTypes;
    typedef Cube Element;
    friend class Cube;

    CubeModel();

    virtual void resize(int size);

    void setParentOf(int childIndex, const Vector3& min, const Vector3& max);

    // -- CollisionModel interface

    virtual void computeBoundingTree(int maxDepth=0);

    bool isStatic() { return static_; }
    void setStatic(bool val=true) { static_ = val; }

    virtual std::pair<Abstract::CollisionElementIterator,Abstract::CollisionElementIterator> getInternalChildren(int index) const;

    virtual std::pair<Abstract::CollisionElementIterator,Abstract::CollisionElementIterator> getExternalChildren(int index) const;

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }

protected:

    int addCube(Cube subcellsBegin, Cube subcellsEnd);
    void updateCube(int index);
    void updateCubes();
};

inline Cube::Cube(CubeModel* model, int index)
    : Abstract::TCollisionElementIterator<CubeModel>(model, index)
{}

inline Cube::Cube(const Abstract::CollisionElementIterator& i)
    : Abstract::TCollisionElementIterator<CubeModel>(static_cast<CubeModel*>(i->getCollisionModel()), i->getIndex())
{
}

inline const Vector3& Cube::minVect() const
{
    return model->elems[index].minBBox;
}

inline const Vector3& Cube::maxVect() const
{
    return model->elems[index].maxBBox;
}


inline const std::pair<Cube,Cube>& Cube::subcells() const
{
    return model->elems[index].subcells;
}

} // namespace Components

} // namespace Sofa

#endif
