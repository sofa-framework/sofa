#ifndef _TRIANGLEMODEL_H_
#define _TRIANGLEMODEL_H_

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/Vec3Types.h"
#include "Triangle.h"

namespace Sofa
{

namespace Components
{

class TriangleModel : public Core::MechanicalObject<Vec3Types>, public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    std::vector<Abstract::CollisionElement*> elems;
    Abstract::CollisionModel* previous;
    Abstract::CollisionModel* next;
    Abstract::BehaviorModel* object;

    VecCoord* internalForces; // ?? What does this can do ??
    VecCoord* externalForces; // ?? What does this can do ??

    class Loader;
public:
    TriangleModel();

    TriangleModel (const char *filename)
    {
        previous = NULL;
        next = NULL;
        init(filename);
    };

    ~TriangleModel()
    {
    };

    TriangleModel* getTriangleModel () {return this;};

private:
    void init(const char *filename);
    void findBoundingBox(const std::vector<Vector3> &verts, Vector3 &minBB, Vector3 &maxBB);

    //attributes

//	std::vector<Vector3*> vertices;
//	std::vector<Vector3*> velocityVertices;

public:

//	const std::string& getTypeName(void) {return typeName;};

    // -- MechanicalModel interface

    void setObject(Abstract::BehaviorModel* obj);

    void beginIteration(double dt);

    void endIteration(double dt);

    void accumulateForce();

    // --- here is the interface for CollisionModel
    //void computeBoundingBox(void);
    void applyTranslation (double dx, double dy, double dz);

    // --- interface for debugging model
    //std::vector < Triangle* >& getTriangles() {return triangles;};
    //void computeSphereVolume (void);
    void computeBoundingBox (void);
    void computeContinueBoundingBox (void);

    std::vector<Abstract::CollisionElement*> & getCollisionElements() {return elems;};

    Abstract::CollisionModel* getNext()
    { return next; }

    Abstract::CollisionModel* getPrevious()
    { return previous; }

    void setNext(Abstract::CollisionModel* n)
    { next = n; }

    void setPrevious(Abstract::CollisionModel* p)
    { previous = p; }

    virtual Abstract::BehaviorModel* getObject()
    { return object; }

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

} // namesapce Components

} // namespace Sofa

#endif /* _TRIANGLEMODEL_H_ */
