#ifndef SOFA_COMPONENTS_TRIANGLEMODEL_H
#define SOFA_COMPONENTS_TRIANGLEMODEL_H

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

class TriangleModel;

class Triangle : public Abstract::TCollisionElementIterator<TriangleModel>
{
public:
    Triangle(TriangleModel* model, int index);

    explicit Triangle(Abstract::CollisionElementIterator& i);

    const Vector3& p1() const;
    const Vector3& p2() const;
    const Vector3& p3() const;

    const Vector3& v1() const;
    const Vector3& v2() const;
    const Vector3& v3() const;

    const Vector3& n() const;
    Vector3& n();
};

class TriangleModel : public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    struct TriangleData
    {
        int i1,i2,i3;
        Vector3 normal;
    };

    std::vector<TriangleData> elems;

    class Loader;

    bool static_;
    int meshRevision;
    bool updateFromTopology();
public:
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Triangle Element;
    friend class Triangle;

    TriangleModel();

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

    virtual const char* getTypeName() const { return "Triangle"; }

protected:

    Core::MechanicalModel<Vec3Types>* mmodel;
    MeshTopology* mesh;

};

inline Triangle::Triangle(TriangleModel* model, int index)
    : Abstract::TCollisionElementIterator<TriangleModel>(model, index)
{}

inline Triangle::Triangle(Abstract::CollisionElementIterator& i)
    : Abstract::TCollisionElementIterator<TriangleModel>(static_cast<TriangleModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Triangle::p1() const { return (*model->mmodel->getX())[model->elems[index].i1]; }
inline const Vector3& Triangle::p2() const { return (*model->mmodel->getX())[model->elems[index].i2]; }
inline const Vector3& Triangle::p3() const { return (*model->mmodel->getX())[model->elems[index].i3]; }

inline const Vector3& Triangle::v1() const { return (*model->mmodel->getV())[model->elems[index].i1]; }
inline const Vector3& Triangle::v2() const { return (*model->mmodel->getV())[model->elems[index].i2]; }
inline const Vector3& Triangle::v3() const { return (*model->mmodel->getV())[model->elems[index].i3]; }

inline const Vector3& Triangle::n() const { return model->elems[index].normal; }
inline       Vector3& Triangle::n()       { return model->elems[index].normal; }

} // namespace Components

} // namespace Sofa

#endif
