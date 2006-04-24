#ifndef SOFA_COMPONENTS_GL_REPERE_H
#define SOFA_COMPONENTS_GL_REPERE_H

#ifdef WIN32
# include <windows.h>
# include <mmsystem.h>
#endif
#include "../Common/Vec.h"
#include "../Common/Quat.h"

#include <GL/glu.h>

namespace Sofa
{

namespace Components
{

namespace GL
{

using namespace Common;

class Axis
{
public:
    Axis();
    Axis(Vector3 &center, Quaternion &orient, float len = 1.0f);
    Axis(Vector3 center, double orient[4][4], float len = 1.0f);
    Axis(double *mat,float Longueur=0.2f);
    ~Axis();

    void update(Vector3 &center, Quaternion &orient);
    void update(Vector3 &center, double orient[4][4]);
    void update(double *mat);

    void draw();
    void push();
    void pop();

private:
    float length;
    double matTransOpenGL[16];
    void init (float len = 0.2f);
    GLUquadricObj *quadratic;
    static GLuint displayList;
    bool initDraw();
};

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
