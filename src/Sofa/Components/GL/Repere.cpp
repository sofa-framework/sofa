#include "Repere.h"

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
GLuint Axis::displayList = 0;

bool Axis::initDraw()
{
    float L = length;
    float l = (L / 15.0f);
    float lc = 2 * l;
    float Lc = lc;

    displayList=glGenLists(1);
    glNewList(displayList, GL_COMPILE);

    //axe des x
    glColor3f(1.0,0.0,0.0);
    glRotatef(90,0.0,1.0,0.0);
    gluCylinder(quadratic,l,l,L,quadricDiscretisation,quadricDiscretisation);
    glRotatef(-90,0.0,1.0,0.0);

    glTranslatef(L,0.0,0.0);
    glRotatef(90,0.0,1.0,0.0);
    //gluDisk(quadratic,0.0f,lc,quadricDiscretisation,quadricDiscretisation);
    gluCylinder(quadratic,lc,0,Lc,quadricDiscretisation,quadricDiscretisation);
    glRotatef(-90,0.0,1.0,0.0);
    glTranslatef(-L,0.0,0.0);

    //axe des y
    glColor3f(0.0,1.0,0.0);
    glRotatef(-90,1.0,0.0,0.0);
    gluCylinder(quadratic,l,l,L,quadricDiscretisation,quadricDiscretisation);
    glRotatef(90,1.0,0.0,0.0);

    glTranslatef(0.0,L,0.0);
    glRotatef(-90,1.0,0.0,0.0);
    //gluDisk(quadratic,0.0f,lc,quadricDiscretisation,quadricDiscretisation);
    gluCylinder(quadratic,lc,0,Lc,quadricDiscretisation,quadricDiscretisation);
    glRotatef(90,1.0,0.0,0.0);
    glTranslatef(0.0,-L,0.0);

    //axe des z
    glColor3f(0.0,0.0,1.0);
    gluCylinder(quadratic,l,l,L,quadricDiscretisation,quadricDiscretisation);

    glTranslatef(0.0,0.0,L);
    //gluDisk(quadratic,0.0f,lc,quadricDiscretisation,quadricDiscretisation);
    gluCylinder(quadratic,lc,0,Lc,quadricDiscretisation,quadricDiscretisation);
    glTranslatef(0.0,0.0,-L);

    glEndList();

    return true;
}


void Axis::push()
{
    glPushMatrix();
    glMultMatrixd(matTransOpenGL);
}

void Axis::pop()
{
    glPopMatrix();
}

void Axis::draw()
{
    //float scalefactor = length / 0.2f;
    float scalefactor = length;

    glPushMatrix();

    glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_DEPTH_BUFFER_BIT );
    //glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);


    //glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

    //glLoadIdentity();
    //glMatrixMode(GL_MODELVIEW);
    glMultMatrixd(matTransOpenGL);
    glScalef(scalefactor,scalefactor,scalefactor);
    //glLoadMatrixd(MatTransOpenGL);
    glCallList(displayList);

    glPopAttrib();
    glPopMatrix();
}

void Axis::update(double *mat)
{
    //cout<<"Axis::update(double *mat), sizeof(mat) = "<<sizeof(mat)<<endl;
    //assert (sizeof(mat)/sizeof(double) == 16);
    std::copy(mat,mat+16, matTransOpenGL);
}


void Axis::update(Vector3 &center, double orient[4][4])
{
    matTransOpenGL[0] = orient[0][0];
    matTransOpenGL[1] = orient[0][1];
    matTransOpenGL[2] = orient[0][2];

    matTransOpenGL[4] = orient[1][0];
    matTransOpenGL[5] = orient[1][1];
    matTransOpenGL[6] = orient[1][2];

    matTransOpenGL[8] = orient[2][0];
    matTransOpenGL[9] = orient[2][1];
    matTransOpenGL[10]= orient[2][2];

    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
}

void Axis::init(float len)
{
    memset(matTransOpenGL, 0, sizeof(matTransOpenGL));
    matTransOpenGL[15]=1.0;

    quadratic=gluNewQuadric();
    gluQuadricNormals(quadratic, GLU_SMOOTH);
    gluQuadricTexture(quadratic, GL_TRUE);

    length = len;

    //displaylist initialisation
    static bool first = true;
    if (first) { initDraw(); first = false; }

}

void Axis::update(Vector3 &center, Quaternion &orient)
{
    double mOrient[4][4];

    orient.buildRotationMatrix(mOrient);
    //QuatToMat3D(orient, matOrient);

    update(center, mOrient);
}

Axis::Axis()
{
    init(1.0);
    Vector3 o(0,0,0);
    Quaternion q(1,0,0,0);
    update( o,q );
}

Axis::Axis(Vector3 &center, Quaternion & orient, float length)
{
    init(length);
    update(center, orient);
}

Axis::Axis( Vector3 center, double orient[4][4], float length)
{
    init(length);
    update (center, orient);
}

Axis::Axis(double *mat, float length)
{
    assert(sizeof(mat)/sizeof(double) == 16);
    init(length);
    update(mat);
}

Axis::~Axis()
{
    gluDeleteQuadric(quadratic);
}

} // namespace GL

} // namespace Components

} // namespace Sofa
