#ifndef SOFA_COMPONENTS_LINEMODEL_H
#define SOFA_COMPONENTS_LINEMODEL_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalObject.h"
#include "MeshTopology.h"
#include "Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class LineModel;

class Line : public Abstract::TCollisionElementIterator<LineModel>
{
public:
    Line(LineModel* model, int index);

    explicit Line(Abstract::CollisionElementIterator& i);

    const Vector3& p1() const;
    const Vector3& p2() const;

    const Vector3& v1() const;
    const Vector3& v2() const;
};

class LineModel : public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    struct LineData
    {
        int i1,i2;
    };

    std::vector<LineData> elems;

    bool static_;
public:
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Line Element;
    friend class Line;

    LineModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    bool isStatic() { return static_; }
    void setStatic(bool val=true) { static_ = val; }

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }


    Core::MechanicalModel<Vec3Types>* getMechanicalModel() { return mmodel; }

    MeshTopology* getTopology() { return mesh; }

protected:

    Core::MechanicalModel<Vec3Types>* mmodel;
    MeshTopology* mesh;

};

inline Line::Line(LineModel* model, int index)
    : Abstract::TCollisionElementIterator<LineModel>(model, index)
{}

inline Line::Line(Abstract::CollisionElementIterator& i)
    : Abstract::TCollisionElementIterator<LineModel>(static_cast<LineModel*>(i->getCollisionModel()), i->getIndex())
{
}

inline const Vector3& Line::p1() const { return (*model->mmodel->getX())[model->elems[index].i1]; }
inline const Vector3& Line::p2() const { return (*model->mmodel->getX())[model->elems[index].i2]; }

inline const Vector3& Line::v1() const { return (*model->mmodel->getV())[model->elems[index].i1]; }
inline const Vector3& Line::v2() const { return (*model->mmodel->getV())[model->elems[index].i2]; }

} // namespace Components

} // namespace Sofa

#endif
