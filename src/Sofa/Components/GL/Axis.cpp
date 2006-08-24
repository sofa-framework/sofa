#include "Axis.h"

#include <GL/gl.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;


namespace Sofa
{

namespace Components
{

namespace GL
{

static const int quadricDiscretisation = 16;
//GLuint Axis::displayList;
//GLUquadricObj *Axis::quadratic = NULL;
std::map < std::pair<std::pair<float,float>,float>, Axis* > Axis::axisMap;

void Axis::initDraw()
{
    if (quadratic!=NULL) return;

    Vector3 L= length;
    double Lmin = L[0];
    if (L[1]<Lmin) Lmin = L[1];
    if (L[2]<Lmin) Lmin = L[2];
    double Lmax = L[0];
    if (L[1]>Lmax) Lmax = L[1];
    if (L[2]>Lmax) Lmax = L[2];
    if (Lmax > Lmin*2)
        Lmax = Lmin*2;
    if (Lmax > Lmin*2)
        Lmin = Lmax/1.414;
    Vector3 l(Lmin / 10, Lmin / 10, Lmin / 10);
    Vector3 lc(Lmax / 5, Lmax / 5, Lmax / 5); // = L / 5;
    Vector3 Lc = lc;

    quadratic=gluNewQuadric();
    gluQuadricNormals(quadratic, GLU_SMOOTH);
    gluQuadricTexture(quadratic, GL_TRUE);

    displayList=glGenLists(1);

    glNewList(displayList, GL_COMPILE);

    // Center
    glColor3f(1,1,1);
    gluSphere(quadratic,l[0],quadricDiscretisation,quadricDiscretisation/2);

    // X Axis
    glColor3f(1,0,0);
    glRotatef(90,0,1,0);
    gluCylinder(quadratic,l[0],l[0],L[0],quadricDiscretisation,quadricDiscretisation);
    glRotatef(-90,0,1,0);

    glTranslated(L[0],0,0);
    glRotatef(90,0,1,0);
    gluDisk(quadratic,0,lc[0],quadricDiscretisation,quadricDiscretisation);
    gluCylinder(quadratic,lc[0],0,Lc[0],quadricDiscretisation,quadricDiscretisation);
    glRotatef(-90,0,1,0);
    glTranslated(-L[0],0,0);

    // Y Axis
    glColor3f(0,1,0);
    glRotatef(-90,1,0,0);
    gluCylinder(quadratic,l[1],l[1],L[1],quadricDiscretisation,quadricDiscretisation);
    glRotatef(90,1,0,0);

    glTranslated(0,L[1],0);
    glRotatef(-90,1,0,0);
    gluDisk(quadratic,0,lc[1],quadricDiscretisation,quadricDiscretisation);
    gluCylinder(quadratic,lc[1],0,Lc[1],quadricDiscretisation,quadricDiscretisation);
    glRotatef(90,1,0,0);
    glTranslated(0,-L[1],0);

    // Z Axis
    glColor3f(0,0,1);
    gluCylinder(quadratic,l[2],l[2],L[2],quadricDiscretisation,quadricDiscretisation);

    glTranslated(0,0,L[2]);
    gluDisk(quadratic,0,lc[2],quadricDiscretisation,quadricDiscretisation);
    gluCylinder(quadratic,lc[2],0,Lc[2],quadricDiscretisation,quadricDiscretisation);
    glTranslated(0,0,-L[2]);
    glEndList();
}

void Axis::draw()
{
    initDraw();

    glPushMatrix();
    glPushAttrib(GL_ENABLE_BIT);

    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glMultMatrixd(matTransOpenGL);
    glCallList(displayList);

    glPopAttrib();
    glPopMatrix();
}

void Axis::update(const double *mat)
{
    std::copy(mat,mat+16, matTransOpenGL);
}

void Axis::update(const Vector3& center, const double orient[4][4])
{
    matTransOpenGL[0] = orient[0][0];
    matTransOpenGL[1] = orient[0][1];
    matTransOpenGL[2] = orient[0][2];
    matTransOpenGL[3] = 0;

    matTransOpenGL[4] = orient[1][0];
    matTransOpenGL[5] = orient[1][1];
    matTransOpenGL[6] = orient[1][2];
    matTransOpenGL[7] = 0;

    matTransOpenGL[8] = orient[2][0];
    matTransOpenGL[9] = orient[2][1];
    matTransOpenGL[10]= orient[2][2];
    matTransOpenGL[11] = 0;

    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
    matTransOpenGL[15] = 1;
}

void Axis::update(const Vector3& center, const Quaternion& orient)
{
    orient.writeOpenGlMatrix(matTransOpenGL);
    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
}

Axis::Axis(double len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(Vector3(0,0,0),  Quaternion(1,0,0,0));
}

Axis::Axis(const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(Vector3(0,0,0),  Quaternion(1,0,0,0));
}

Axis::Axis(const Vector3& center, const Quaternion& orient, const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(center, orient);
}

Axis::Axis(const Vector3& center, const double orient[4][4], const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(center, orient);
}

Axis::Axis(const double *mat, const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(mat);
}

Axis::Axis(const Vector3& center, const Quaternion& orient, double len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(center, orient);
}
Axis::Axis(const Vector3& center, const double orient[4][4], double len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(center, orient);
}

Axis::Axis(const double *mat, double len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(mat);
}

Axis::~Axis()
{
    if (quadratic != NULL)
        gluDeleteQuadric(quadratic);
}

Axis* Axis::get(const Vector3& len)
{
    Axis*& a = axisMap[std::make_pair(std::make_pair((float)len[0],(float)len[1]),(float)len[2])];
    if (a==NULL)
        a = new Axis(len);
    return a;
}

void Axis::draw(const Vector3& center, const Quaternion& orient, const Vector3& len)
{
    Axis* a = get(len);
    a->update(center, orient);
    a->draw();
}

void Axis::draw(const Vector3& center, const double orient[4][4], const Vector3& len)
{
    Axis* a = get(len);
    a->update(center, orient);
    a->draw();
}

void Axis::draw(const double *mat, const Vector3& len)
{
    Axis* a = get(len);
    a->update(mat);
    a->draw();
}

void Axis::draw(const Vector3& center, const Quaternion& orient, double len)
{
    Axis* a = get(Vector3(len,len,len));
    a->update(center, orient);
    a->draw();
}

void Axis::draw(const Vector3& center, const double orient[4][4], double len)
{
    Axis* a = get(Vector3(len,len,len));
    a->update(center, orient);
    a->draw();
}

void Axis::draw(const double *mat, double len)
{
    Axis* a = get(Vector3(len,len,len));
    a->update(mat);
    a->draw();
}

} // namespace GL

} // namespace Components

} // namespace Sofa
