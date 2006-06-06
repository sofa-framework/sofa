#ifndef SOFA_COMPONENTS_LINEMODEL_H
#define SOFA_COMPONENTS_LINEMODEL_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalModel.h"
#include "MeshTopology.h"
#include "Common/Vec3Types.h"
#include "Line.h"

namespace Sofa
{

namespace Components
{

class LineModel : public Abstract::CollisionModel, public Abstract::VisualModel
{
public:
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;

    LineModel();

    ~LineModel();

    bool isStatic() { return static_; }
    void setStatic(bool val=true) { static_ = val; }

    virtual void init();

    // --- CollisionModel interface

    void computeBoundingBox (void);
    void computeContinuousBoundingBox (double dt);

    std::vector<Abstract::CollisionElement*> & getCollisionElements() {return elems;};

    Abstract::CollisionModel* getNext()
    { return next; }

    Abstract::CollisionModel* getPrevious()
    { return previous; }

    void setNext(Abstract::CollisionModel* n)
    { next = n; }

    void setPrevious(Abstract::CollisionModel* p)
    { previous = p; }

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }

    Core::MechanicalModel<Vec3Types>* getMechanicalModel() { return mmodel; }

    MeshTopology* getTopology() { return mesh; }

protected:

    Core::MechanicalModel<Vec3Types>* mmodel;
    MeshTopology* mesh;

    std::vector<Abstract::CollisionElement*> elems;
    Abstract::CollisionModel* previous;
    Abstract::CollisionModel* next;

    bool static_;

    void findBoundingBox(const std::vector<Vector3> &verts, Vector3 &minBB, Vector3 &maxBB);

    friend class Line;
};

} // namesapce Components

} // namespace Sofa

#endif
