#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "Sofa/Abstract/CollisionElement.h"
#include "Common/Vec.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>

namespace Sofa
{

namespace Components
{

class TriangleModel;

using namespace Common;

class Triangle : public Abstract::CollisionElement
{

public:
    Vector3 *p1, *p2, *p3;
    Vector3 *v1, *v2, *v3;
    Vector3 normal;
    TriangleModel * trMdl;

    Triangle(Vector3 *_p1, Vector3 *_p2, Vector3 *_p3, Vector3 *_v1, Vector3 *_v2, Vector3 *_v3, Vector3 norm, TriangleModel *_trMdl): p1(_p1), p2(_p2), p3(_p3), v1(_v1), v2(_v2), v3(_v3), normal(norm), trMdl(_trMdl) {};
    Triangle(Vector3 *_p1, Vector3 *_p2, Vector3 *_p3, Vector3 *_v1, Vector3 *_v2, Vector3 *_v3, TriangleModel *_trMdl): p1(_p1), p2(_p2), p3(_p3), v1(_v1), v2(_v2), v3(_v3), trMdl(_trMdl) {};

    void getBBox(Vector3 &minBBox, Vector3 &maxBBox);
    void getBBox(double* minVect, double* maxVect);

    Triangle *getTriangle(void) {return this;};

    // for debugging only
    void draw (void);

    friend std::ostream& operator<< (std::ostream& os, const Triangle &tri);

    Abstract::CollisionModel* getCollisionModel();
};

} // namespace Components

} // namespace Sofa
#endif /* _TRIANGLE_H_ */
