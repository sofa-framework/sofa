/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/


#ifndef DRAWTOOLOGRE_H
#define DRAWTOOLOGRE_H

#include <sofa/core/visual/DrawTool.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <OgreManualObject.h>
#include <stack>

namespace sofa
{

namespace core
{

namespace visual
{

class DrawToolOGRE: public DrawTool
{

public:
    typedef DrawTool::Vec4f      Vec4f;
    typedef DrawTool::Vec3f      Vec3f;
    typedef DrawTool::Vector3    Vector3;
    typedef DrawTool::Vec3i      Vec3i;
    typedef DrawTool::Vec2i      Vec2i;
    typedef DrawTool::Quaternion Quaternion;

    DrawToolOGRE();
    ~DrawToolOGRE();

    virtual void drawPoints(const std::vector<Vector3> &points, float size,  const Vec4f colour);

    virtual void drawLines(const std::vector<Vector3> &points, float size, const Vec4f colour);
    virtual void drawLines(const std::vector<Vector3> &points, const std::vector< Vec2i > &index, float size, const Vec4f colour);

    virtual void drawTriangles(const std::vector<Vector3> &points, const Vec4f colour);
    virtual void drawTriangles(const std::vector<Vector3> &points, const Vector3 normal, const Vec4f colour);
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector< defaulttype::Vec<3,int> > &index,
            const std::vector<Vector3>  &normal,
            const Vec4f colour);
    virtual void drawTriangles(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const std::vector< Vec4f > &colour);

    virtual void drawTriangleStrip(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f colour);

    virtual void drawTriangleFan(const std::vector<Vector3> &points,
            const std::vector<Vector3>  &normal,
            const Vec4f colour);

    virtual void drawFrame   (const Vector3& position, const Quaternion &orientation, const Vec3f &size);

    virtual void drawSpheres (const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec4f colour);
    virtual void drawSpheres (const std::vector<Vector3> &points, float radius, const Vec4f colour);


    virtual void drawCone    (const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const Vec4f colour, int subd=16);

    virtual void drawCube    (const float& radius, const Vec4f& colour, const int& subd=16); // Draw a cube of size one centered on the current point.

    virtual void drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16);

    virtual void drawArrow   (const Vector3& p1, const Vector3 &p2, float radius, const Vec4f colour,  int subd=16);

    virtual void drawPlus    (const float& radius, const Vec4f& colour, const int& subd=16); // Draw a plus sign of size one centered on the current point.

    virtual void drawPoint   (const Vector3 &p, const Vec4f &c);
    virtual void drawPoint   (const Vector3 &p, const Vector3 &n, const Vec4f &c);

    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal, const Vec4f &c);
    virtual void drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
            const Vector3 &normal,
            const Vec4f &c1, const Vec4f &c2, const Vec4f &c3);

    virtual void drawSphere  (const Vector3 &p, float radius);

    virtual void clear();

    virtual void setMaterial(const Vec4f &colour, std::string name=std::string());

    virtual void resetMaterial(const Vec4f &colour, std::string name=std::string());

    // Fonctions Ogre
    void drawOgreVertexPosition(const Vector3 &p);
    void drawOgreVertexColour(const Vec4f &p);
    void drawOgreVertexNormal(const Vector3 &p);

    void pushMatrix();
    void popMatrix();
    void multMatrix(float* glTransform );
    void scale(float s);

protected:
    // data member
    bool mLightEnabled;
    int  mPolygonMode;      //0: no cull, 1 front (CULL_CLOCKWISE), 2 back (CULL_ANTICLOCKWISE)
    bool mWireFrameEnabled;

    // data OGRE
    Ogre::ManualObject* pOgreDraw;
    Ogre::MaterialPtr   pCurrentMaterial;
    Ogre::SceneManager* pSceneMgr;

    std::stack<Ogre::SceneNode*> sceneNodeStack;

    Ogre::ManualObject* pOgreDrawInternal; // created by the DrawTool
    Ogre::ManualObject* pOgreDrawSave; // pointer to swap between ogredraw and ogredrawinternal

public:
    static int mMaterialName;
    static int mEntityName;
    static int mMeshName;


    // getter & setter
    virtual void setLightingEnabled(bool _isAnabled);

    bool getLightEnabled()        {return mLightEnabled;}

    virtual void setPolygonMode(int _mode, bool _wireframe);

    bool getPolygonMode()             {return mPolygonMode;}
    bool getWireFrameEnabled()        {return mWireFrameEnabled;}

    void setOgreObject(Ogre::ManualObject* _ManuObj) {pOgreDraw=_ManuObj;}
    void setOgreMaterial(Ogre::MaterialPtr _mat)    {pCurrentMaterial=_mat;}
    void setSceneMgr(Ogre::SceneManager* _scnMgr);

};
} //namespace gl

} // namespace helper

} //namespace sofa

#endif // DRAWTOOLOGRE_H
