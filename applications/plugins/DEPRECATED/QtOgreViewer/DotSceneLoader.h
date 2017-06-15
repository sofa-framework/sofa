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
// http://www.ogre3d.org/wiki/index.php/New_DotScene_Loader

#ifndef DOT_SCENELOADER_H
#define DOT_SCENELOADER_H

// Includes
#include <OgreString.h>
#include <OgreVector3.h>
#include <OgreQuaternion.h>
#include <OgreColourValue.h>
#include <vector>

#include <tinyxml.h>
// Forward declarations
class TiXmlElement;

namespace Ogre
{
// Forward declarations
class SceneManager;
class SceneNode;

class nodeProperty
{
public:
    String nodeName;
    String propertyNm;
    String valueName;
    String typeName;

    nodeProperty(const String &node, const String &propertyName, const String &value, const String &type)
        : nodeName(node), propertyNm(propertyName), valueName(value), typeName(type) {}
};

class EnvironmentProperty
{
public:
    ColourValue ambientColour;
    ColourValue backgroundColour;
    Real farClipDistance;
    Real nearClipDistance;
};

class DotSceneLoader
{
public:
    DotSceneLoader();
    virtual ~DotSceneLoader() {}

    void parseDotScene(const String &SceneName, const String &groupName, SceneManager *yourSceneMgr, SceneNode *pAttachNode = NULL, const String &sPrependNode = "");
    SceneManager *getSceneManager() { return mSceneMgr; }
    String getProperty(const String &ndNm, const String &prop);

    std::vector<nodeProperty> nodeProperties;
    std::vector<String> staticObjects;
    std::vector<String> dynamicObjects;

    std::vector<String> directionalLights;
    std::vector<String> pointLights;
    std::vector<String> spotLights;
    EnvironmentProperty environment;

protected:
    void processScene(TiXmlElement *XMLRoot);

    void processNodes(TiXmlElement *XMLNode);
    void processExternals(TiXmlElement *XMLNode);
    void processEnvironment(TiXmlElement *XMLNode);
    void processTerrain(TiXmlElement *XMLNode);
    void processUserDataReference(TiXmlElement *XMLNode, SceneNode *pParent = 0);
    void processUserDataReference(TiXmlElement *XMLNode, Entity *pEntity);
    void processOctree(TiXmlElement *XMLNode);
    void processLight(TiXmlElement *XMLNode, SceneNode *pParent = 0);
    void processCamera(TiXmlElement *XMLNode, SceneNode *pParent = 0);

    void processNode(TiXmlElement *XMLNode, SceneNode *pParent = 0);
    void processLookTarget(TiXmlElement *XMLNode, SceneNode *pParent);
    void processTrackTarget(TiXmlElement *XMLNode, SceneNode *pParent);
    void processLookTarget(TiXmlElement *XMLNode, Camera *pParent);
    void processTrackTarget(TiXmlElement *XMLNode, Camera *pParent);
    void processEntity(TiXmlElement *XMLNode, SceneNode *pParent);
    void processParticleSystem(TiXmlElement *XMLNode, SceneNode *pParent);
    void processBillboardSet(TiXmlElement *XMLNode, SceneNode *pParent);
    void processPlane(TiXmlElement *XMLNode, SceneNode *pParent);

    void processFog(TiXmlElement *XMLNode);
    void processSkyBox(TiXmlElement *XMLNode);
    void processSkyDome(TiXmlElement *XMLNode);
    void processSkyPlane(TiXmlElement *XMLNode);
    void processClipping(TiXmlElement *XMLNode, Camera* pCamera = 0);

    void processLightRange(TiXmlElement *XMLNode, Light *pLight);
    void processLightAttenuation(TiXmlElement *XMLNode, Light *pLight);

    String getAttrib(TiXmlElement *XMLNode, const String &parameter, const String &defaultValue = "");
    Real getAttribReal(TiXmlElement *XMLNode, const String &parameter, Real defaultValue = 0);
    int getAttribInt(TiXmlElement *XMLNode, const String &parameter, int defaultValue = 0);
    bool getAttribBool(TiXmlElement *XMLNode, const String &parameter, bool defaultValue = false);

    Vector3 parseVector3(TiXmlElement *XMLNode);
    Quaternion parseQuaternion(TiXmlElement *XMLNode);
    ColourValue parseColour(TiXmlElement *XMLNode);


    SceneManager *mSceneMgr;
    SceneNode *mAttachNode;
    String m_sGroupName;
    String m_sPrependNode;
};
}

#endif // DOT_SCENELOADER_H
