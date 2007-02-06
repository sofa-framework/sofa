#ifndef SOFA_COMPONENTS_GL_AXIS_H
#define SOFA_COMPONENTS_GL_AXIS_H

#ifdef WIN32
# include <windows.h>
# include <mmsystem.h>
#endif
#include "../Common/Vec.h"
#include "../Common/Quat.h"

#include <GL/glu.h>
#include <map>

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

    Axis(double len=1);
    Axis(const Vector3& len);
    Axis(const Vector3& center, const Quaternion &orient, const Vector3& length);
    Axis(const Vector3& center, const double orient[4][4], const Vector3& length);
    Axis(const double *mat, const Vector3& length);
    Axis(const Vector3& center, const Quaternion &orient, double length=1);
    Axis(const Vector3& center, const double orient[4][4], double length=1);
    Axis(const double *mat, double length=1.0);

    ~Axis();

    void update(const Vector3& center, const Quaternion& orient = Quaternion());
    void update(const Vector3& center, const double orient[4][4]);
    void update(const double *mat);

    void draw();

    static void draw(const Vector3& center, const Quaternion& orient, const Vector3& length);
    static void draw(const Vector3& center, const double orient[4][4], const Vector3& length);
    static void draw(const double *mat, const Vector3& length);
    static void draw(const Vector3& center, const Quaternion& orient, double length=1);
    static void draw(const Vector3& center, const double orient[4][4], double length=1);
    static void draw(const double *mat, double length=1.0);

private:

    Vector3 length;
    double matTransOpenGL[16];

    GLUquadricObj *quadratic;
    GLuint displayList;

    void initDraw();

    static std::map < std::pair<std::pair<float,float>,float>, Axis* > axisMap;
    static Axis* get(const Vector3& len);

};

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
