/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include "DotSceneLoader.h"
#include <Ogre.h>

using namespace std;
using namespace Ogre;

DotSceneLoader::DotSceneLoader()
    : mSceneMgr(0)
{
    environment.nearClipDistance = 0.2;
    environment.farClipDistance = 1000;
    environment.backgroundColour = ColourValue(0,0,0,0);
    environment.ambientColour = ColourValue(0,0,0,1);
}

void DotSceneLoader::parseDotScene(const String &SceneName, const String &groupName, SceneManager *yourSceneMgr, SceneNode *pAttachNode, const String &sPrependNode)
{
    // set up shared object values
    m_sGroupName = groupName;
    mSceneMgr = yourSceneMgr;
    m_sPrependNode = sPrependNode;
    staticObjects.clear();
    dynamicObjects.clear();

    TiXmlDocument   *XMLDoc = 0;
    TiXmlElement   *XMLRoot;

    try
    {
        DataStreamPtr pStream = ResourceGroupManager::getSingleton().
                openResource( SceneName, groupName );

        String data = pStream->getAsString();
        // Open the .scene File
        XMLDoc = new TiXmlDocument();
        XMLDoc->Parse( data.c_str() );
        pStream->close();
        pStream.setNull();

        if( XMLDoc->Error() )
        {
            //We'll just log, and continue on gracefully
            LogManager::getSingleton().logMessage("[DotSceneLoader] The TiXmlDocument reported an error");
            delete XMLDoc;
            return;
        }
    }
    catch(...)
    {
        //We'll just log, and continue on gracefully
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error creating TiXmlDocument");
        delete XMLDoc;
        return;
    }

    // Validate the File
    XMLRoot = XMLDoc->RootElement();
    if( String( XMLRoot->Value()) != "scene"  )
    {
        LogManager::getSingleton().logMessage( "[DotSceneLoader] Error: Invalid .scene File. Missing <scene>" );
        delete XMLDoc;
        return;
    }

    // figure out where to attach any nodes we create
    mAttachNode = pAttachNode;
    if(!mAttachNode && mSceneMgr)
        mAttachNode = mSceneMgr->getRootSceneNode();

    // Process the scene
    processScene(XMLRoot);

    // Close the XML File
    delete XMLDoc;
}

void DotSceneLoader::processScene(TiXmlElement *XMLRoot)
{
    // Process the scene parameters
    String version = getAttrib(XMLRoot, "formatVersion", "unknown");

    String message = "[DotSceneLoader] Parsing dotScene file with version " + version;
    if(XMLRoot->Attribute("ID"))
        message += ", id " + String(XMLRoot->Attribute("ID"));
    if(XMLRoot->Attribute("sceneManager"))
        message += ", scene manager " + String(XMLRoot->Attribute("sceneManager"));
    if(XMLRoot->Attribute("minOgreVersion"))
        message += ", min. Ogre version " + String(XMLRoot->Attribute("minOgreVersion"));
    if(XMLRoot->Attribute("author"))
        message += ", author " + String(XMLRoot->Attribute("author"));

    LogManager::getSingleton().logMessage(message);

    if(!mSceneMgr)
    {
        if (XMLRoot->Attribute("sceneManager"))
            mSceneMgr = Root::getSingleton().createSceneManager(getAttrib(XMLRoot,"sceneManager"));
        else
            mSceneMgr = Root::getSingleton().createSceneManager(ST_GENERIC);
        if(!mAttachNode)
            mAttachNode = mSceneMgr->getRootSceneNode();
    }

    // do this first so we generate edge lists
    //mSceneMgr->setShadowTechnique(SHADOWTYPE_STENCIL_ADDITIVE);

    TiXmlElement *pElement;

    // Process nodes (?)
    pElement = XMLRoot->FirstChildElement("nodes");
    if(pElement)
        processNodes(pElement);

    // Process externals (?)
    pElement = XMLRoot->FirstChildElement("externals");
    if(pElement)
        processExternals(pElement);

    // Process environment (?)
    pElement = XMLRoot->FirstChildElement("environment");
    if(pElement)
        processEnvironment(pElement);

    // Process terrain (?)
    pElement = XMLRoot->FirstChildElement("terrain");
    if(pElement)
        processTerrain(pElement);

    // Process userDataReference (?)
    pElement = XMLRoot->FirstChildElement("userDataReference");
    if(pElement)
        processUserDataReference(pElement);

    // Process octree (?)
    pElement = XMLRoot->FirstChildElement("octree");
    if(pElement)
        processOctree(pElement);

    // Process light (?)
    pElement = XMLRoot->FirstChildElement("light");
    if(pElement)
        processLight(pElement);

    // Process camera (?)
    pElement = XMLRoot->FirstChildElement("camera");
    if(pElement)
        processCamera(pElement);
}

void DotSceneLoader::processNodes(TiXmlElement *XMLNode)
{
    TiXmlElement *pElement;

    // Process node (*)
    pElement = XMLNode->FirstChildElement("node");
    while(pElement)
    {
        processNode(pElement);
        pElement = pElement->NextSiblingElement("node");
    }

    // Process position (?)
    pElement = XMLNode->FirstChildElement("position");
    if(pElement)
    {
        mAttachNode->setPosition(parseVector3(pElement));
        mAttachNode->setInitialState();
    }

    // Process rotation (?)
    pElement = XMLNode->FirstChildElement("rotation");
    if(pElement)
    {
        mAttachNode->setOrientation(parseQuaternion(pElement));
        mAttachNode->setInitialState();
    }

    // Process scale (?)
    pElement = XMLNode->FirstChildElement("scale");
    if(pElement)
    {
        mAttachNode->setScale(parseVector3(pElement));
        mAttachNode->setInitialState();
    }
}

void DotSceneLoader::processExternals(TiXmlElement *XMLNode)
{
    //! @todo Implement this
}

void DotSceneLoader::processEnvironment(TiXmlElement *XMLNode)
{
    TiXmlElement *pElement;

    // Process fog (?)
    pElement = XMLNode->FirstChildElement("fog");
    if(pElement)
        processFog(pElement);

    // Process skyBox (?)
    pElement = XMLNode->FirstChildElement("skyBox");
    if(pElement)
        processSkyBox(pElement);

    // Process skyDome (?)
    pElement = XMLNode->FirstChildElement("skyDome");
    if(pElement)
        processSkyDome(pElement);

    // Process skyPlane (?)
    pElement = XMLNode->FirstChildElement("skyPlane");
    if(pElement)
        processSkyPlane(pElement);

    // Process clipping (?)
    pElement = XMLNode->FirstChildElement("clipping");
    if(pElement)
        processClipping(pElement);

    // Process colourAmbient (?)
    pElement = XMLNode->FirstChildElement("colourAmbient");
    if(pElement)
    {
        environment.ambientColour = parseColour(pElement);
        mSceneMgr->setAmbientLight(environment.ambientColour);
    }

    // Process colourBackground (?)
    //! @todo Set the background colour of all viewports (RenderWindow has to be provided then)
    pElement = XMLNode->FirstChildElement("colourBackground");
    if(pElement)
        environment.backgroundColour = parseColour(pElement);

    // Process userDataReference (?)
    pElement = XMLNode->FirstChildElement("userDataReference");
    if(pElement)
        processUserDataReference(pElement);
}

void DotSceneLoader::processTerrain(TiXmlElement *XMLNode)
{
    //! @todo Implement this
}

void DotSceneLoader::processUserDataReference(TiXmlElement *XMLNode, SceneNode *pParent)
{
    //! @todo Implement this
}

void DotSceneLoader::processOctree(TiXmlElement *XMLNode)
{
    //! @todo Implement this
}

void DotSceneLoader::processLight(TiXmlElement *XMLNode, SceneNode *pParent)
{
    // Process attributes
    String name = getAttrib(XMLNode, "name");
    String id = getAttrib(XMLNode, "id");

    // Create the light
    Light *pLight = mSceneMgr->createLight(name);
    if(pParent)
        pParent->attachObject(pLight);

    String sValue = getAttrib(XMLNode, "type");
    if(sValue == "point")
    {
        pLight->setType(Light::LT_POINT);
        pointLights.push_back(name);
    }
    else if(sValue == "directional")
    {
        pLight->setType(Light::LT_DIRECTIONAL);
        directionalLights.push_back(name);
    }
    else if(sValue == "spot")
    {
        pLight->setType(Light::LT_SPOTLIGHT);
        spotLights.push_back(name);
    }
    else if(sValue == "radPoint")
    {
        pLight->setType(Light::LT_POINT);
        pointLights.push_back(name);
    }

    pLight->setVisible(getAttribBool(XMLNode, "visible", true));
    pLight->setCastShadows(getAttribBool(XMLNode, "castShadows", true));

    TiXmlElement *pElement;

    // Process position (?)
    pElement = XMLNode->FirstChildElement("position");
    if(pElement)
        pLight->setPosition(parseVector3(pElement));

    // Process normal (?)
    pElement = XMLNode->FirstChildElement("normal");
    if(pElement)
        pLight->setDirection(parseVector3(pElement));

    // Process colourDiffuse (?)
    pElement = XMLNode->FirstChildElement("colourDiffuse");
    if(pElement)
        pLight->setDiffuseColour(parseColour(pElement));

    // Process colourSpecular (?)
    pElement = XMLNode->FirstChildElement("colourSpecular");
    if(pElement)
        pLight->setSpecularColour(parseColour(pElement));

    // Process lightRange (?)
    pElement = XMLNode->FirstChildElement("lightRange");
    if(pElement)
        processLightRange(pElement, pLight);

    // Process lightAttenuation (?)
    pElement = XMLNode->FirstChildElement("lightAttenuation");
    if(pElement)
        processLightAttenuation(pElement, pLight);

    // Process userDataReference (?)
    pElement = XMLNode->FirstChildElement("userDataReference");
//	if(pElement)
//		;//processUserDataReference(pElement, pLight);
}

void DotSceneLoader::processCamera(TiXmlElement *XMLNode, SceneNode *pParent)
{
    // Process attributes
    String name = getAttrib(XMLNode, "name");
    String id = getAttrib(XMLNode, "id");

    // Create the camera
    Camera *pCamera = mSceneMgr->createCamera(name);
    if(pParent)
        pParent->attachObject(pCamera);

    if (XMLNode->Attribute("fov"))
        pCamera->setFOVy(Ogre::Angle(getAttribReal(XMLNode, "fov")));
    if (XMLNode->Attribute("aspectRatio"))
        pCamera->setAspectRatio(getAttribReal(XMLNode, "aspectRatio"));

    String sValue = getAttrib(XMLNode, "projectionType");
    if(sValue == "perspective")
        pCamera->setProjectionType(Ogre::PT_PERSPECTIVE);
    else if(sValue == "orthographic")
        pCamera->setProjectionType(Ogre::PT_ORTHOGRAPHIC);

    TiXmlElement *pElement;

    // Process clipping (?)
    pElement = XMLNode->FirstChildElement("clipping");
    if(pElement)
        processClipping(pElement, pCamera);

    // Process position (?)
    pElement = XMLNode->FirstChildElement("position");
    if(pElement)
        pCamera->setPosition(parseVector3(pElement));

    // Process rotation (?)
    pElement = XMLNode->FirstChildElement("rotation");
    if(pElement)
        pCamera->setOrientation(parseQuaternion(pElement));

    // Process normal (?)
    pElement = XMLNode->FirstChildElement("normal");
    if(pElement)
        pCamera->setDirection(parseVector3(pElement));

    // Process lookTarget (?)
    pElement = XMLNode->FirstChildElement("lookTarget");
    if(pElement)
        processLookTarget(pElement, pCamera);

    // Process trackTarget (?)
    pElement = XMLNode->FirstChildElement("trackTarget");
    if(pElement)
        processTrackTarget(pElement, pCamera);

    // Process userDataReference (?)
    pElement = XMLNode->FirstChildElement("userDataReference");
//	if(pElement)
//		;//processUserDataReference(pElement, pCamera);
}

void DotSceneLoader::processNode(TiXmlElement *XMLNode, SceneNode *pParent)
{
    // Construct the node's name
    String name = m_sPrependNode + getAttrib(XMLNode, "name");

    // Create the scene node
    SceneNode *pNode;
    if(name.empty())
    {
        // Let Ogre choose the name
        if(pParent)
            pNode = pParent->createChildSceneNode();
        else
            pNode = mAttachNode->createChildSceneNode();
    }
    else
    {
        // Provide the name
        if(pParent)
            pNode = pParent->createChildSceneNode(name);
        else
            pNode = mAttachNode->createChildSceneNode(name);
    }

    // Process other attributes
    String id = getAttrib(XMLNode, "id");
    bool isTarget = getAttribBool(XMLNode, "isTarget");

    TiXmlElement *pElement;

    // Process position (?)
    pElement = XMLNode->FirstChildElement("position");
    if(pElement)
    {
        pNode->setPosition(parseVector3(pElement));
        pNode->setInitialState();
    }

    // Process rotation (?)
    pElement = XMLNode->FirstChildElement("rotation");
    if(pElement)
    {
        pNode->setOrientation(parseQuaternion(pElement));
        pNode->setInitialState();
    }

    // Process scale (?)
    pElement = XMLNode->FirstChildElement("scale");
    if(pElement)
    {
        pNode->setScale(parseVector3(pElement));
        pNode->setInitialState();
    }

    // Process lookTarget (?)
    pElement = XMLNode->FirstChildElement("lookTarget");
    if(pElement)
        processLookTarget(pElement, pNode);

    // Process trackTarget (?)
    pElement = XMLNode->FirstChildElement("trackTarget");
    if(pElement)
        processTrackTarget(pElement, pNode);

    // Process node (*)
    pElement = XMLNode->FirstChildElement("node");
    while(pElement)
    {
        processNode(pElement, pNode);
        pElement = pElement->NextSiblingElement("node");
    }

    // Process entity (*)
    pElement = XMLNode->FirstChildElement("entity");
    while(pElement)
    {
        processEntity(pElement, pNode);
        pElement = pElement->NextSiblingElement("entity");
    }

    // Process light (*)
    pElement = XMLNode->FirstChildElement("light");
    while(pElement)
    {
        processLight(pElement, pNode);
        pElement = pElement->NextSiblingElement("light");
    }

    // Process camera (*)
    pElement = XMLNode->FirstChildElement("camera");
    while(pElement)
    {
        processCamera(pElement, pNode);
        pElement = pElement->NextSiblingElement("camera");
    }

    // Process particleSystem (*)
    pElement = XMLNode->FirstChildElement("particleSystem");
    while(pElement)
    {
        processParticleSystem(pElement, pNode);
        pElement = pElement->NextSiblingElement("particleSystem");
    }

    // Process billboardSet (*)
    pElement = XMLNode->FirstChildElement("billboardSet");
    while(pElement)
    {
        processBillboardSet(pElement, pNode);
        pElement = pElement->NextSiblingElement("billboardSet");
    }

    // Process plane (*)
    pElement = XMLNode->FirstChildElement("plane");
    while(pElement)
    {
        processPlane(pElement, pNode);
        pElement = pElement->NextSiblingElement("plane");
    }

    // Process userDataReference (?)
    pElement = XMLNode->FirstChildElement("userDataReference");
    if(pElement)
        processUserDataReference(pElement, pNode);
}

void DotSceneLoader::processLookTarget(TiXmlElement *XMLNode, SceneNode *pParent)
{
    //! @todo Is this correct? Cause I don't have a clue actually

    // Process attributes
    String nodeName = getAttrib(XMLNode, "nodeName");

    Node::TransformSpace relativeTo = Node::TS_PARENT;
    String sValue = getAttrib(XMLNode, "relativeTo");
    if(sValue == "local")
        relativeTo = Node::TS_LOCAL;
    else if(sValue == "parent")
        relativeTo = Node::TS_PARENT;
    else if(sValue == "world")
        relativeTo = Node::TS_WORLD;

    TiXmlElement *pElement;

    // Process position (?)
    Vector3 position;
    pElement = XMLNode->FirstChildElement("position");
    if(pElement)
        position = parseVector3(pElement);

    // Process localDirection (?)
    Vector3 localDirection = Vector3::NEGATIVE_UNIT_Z;
    pElement = XMLNode->FirstChildElement("localDirection");
    if(pElement)
        localDirection = parseVector3(pElement);

    // Setup the look target
    try
    {
        if(!nodeName.empty())
        {
            SceneNode *pLookNode = mSceneMgr->getSceneNode(nodeName);
            position = pLookNode->_getDerivedPosition();
        }

        pParent->lookAt(position, relativeTo, localDirection);
    }
    catch(Ogre::Exception &/*e*/)
    {
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error processing a look target!");
    }
}

void DotSceneLoader::processTrackTarget(TiXmlElement *XMLNode, SceneNode *pParent)
{
    // Process attributes
    String nodeName = getAttrib(XMLNode, "nodeName");

    TiXmlElement *pElement;

    // Process localDirection (?)
    Vector3 localDirection = Vector3::NEGATIVE_UNIT_Z;
    pElement = XMLNode->FirstChildElement("localDirection");
    if(pElement)
        localDirection = parseVector3(pElement);

    // Process offset (?)
    Vector3 offset = Vector3::ZERO;
    pElement = XMLNode->FirstChildElement("offset");
    if(pElement)
        offset = parseVector3(pElement);

    // Setup the track target
    try
    {
        SceneNode *pTrackNode = mSceneMgr->getSceneNode(nodeName);
        pParent->setAutoTracking(true, pTrackNode, localDirection, offset);
    }
    catch(Ogre::Exception &/*e*/)
    {
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error processing a track target!");
    }
}

void DotSceneLoader::processLookTarget(TiXmlElement *XMLNode, Camera *pParent)
{
    //! @todo Is this correct? Cause I don't have a clue actually
    //! @todo Merge/factorize code with SceneNode version

    // Process attributes
    String nodeName = getAttrib(XMLNode, "nodeName");

    Node::TransformSpace relativeTo = Node::TS_PARENT;
    String sValue = getAttrib(XMLNode, "relativeTo");
    if(sValue == "local")
        relativeTo = Node::TS_LOCAL;
    else if(sValue == "parent")
        relativeTo = Node::TS_PARENT;
    else if(sValue == "world")
        relativeTo = Node::TS_WORLD;

    TiXmlElement *pElement;

    // Process position (?)
    Vector3 position;
    pElement = XMLNode->FirstChildElement("position");
    if(pElement)
        position = parseVector3(pElement);

    // Process localDirection (?)
    Vector3 localDirection = Vector3::NEGATIVE_UNIT_Z;
    pElement = XMLNode->FirstChildElement("localDirection");
    if(pElement)
        localDirection = parseVector3(pElement);

    // Setup the look target
    try
    {
        if(!nodeName.empty())
        {
            SceneNode *pLookNode = mSceneMgr->getSceneNode(nodeName);
            position = pLookNode->_getDerivedPosition();
        }

        pParent->lookAt(position /*, relativeTo, localDirection*/);
    }
    catch(Ogre::Exception &/*e*/)
    {
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error processing a look target!");
    }
}

void DotSceneLoader::processTrackTarget(TiXmlElement *XMLNode, Camera *pParent)
{
    //! @todo Merge/factorize code with SceneNode version
    // Process attributes
    String nodeName = getAttrib(XMLNode, "nodeName");

    TiXmlElement *pElement;

    // Process localDirection (?)
    Vector3 localDirection = Vector3::NEGATIVE_UNIT_Z;
    pElement = XMLNode->FirstChildElement("localDirection");
    if(pElement)
        localDirection = parseVector3(pElement);

    // Process offset (?)
    Vector3 offset = Vector3::ZERO;
    pElement = XMLNode->FirstChildElement("offset");
    if(pElement)
        offset = parseVector3(pElement);

    // Setup the track target
    try
    {
        SceneNode *pTrackNode = mSceneMgr->getSceneNode(nodeName);
        pParent->setAutoTracking(true, pTrackNode, /*localDirection,*/ offset);
    }
    catch(Ogre::Exception &/*e*/)
    {
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error processing a track target!");
    }
}

void DotSceneLoader::processEntity(TiXmlElement *XMLNode, SceneNode *pParent)
{


    // Process attributes
    String name = getAttrib(XMLNode, "name");

    String id = getAttrib(XMLNode, "id");
    String meshFile = getAttrib(XMLNode, "meshFile");
    String materialFile = getAttrib(XMLNode, "materialFile");
    bool isStatic = getAttribBool(XMLNode, "static", false);;
    bool castShadows = getAttribBool(XMLNode, "castShadows", true);

    // TEMP: Maintain a list of static and dynamic objects
    if(isStatic)
        staticObjects.push_back(name);
    else
        dynamicObjects.push_back(name);

    TiXmlElement *pElement;

    // Process vertexBuffer (?)
    pElement = XMLNode->FirstChildElement("vertexBuffer");
//	if(pElement)
//		;//processVertexBuffer(pElement);

    // Process indexBuffer (?)
    pElement = XMLNode->FirstChildElement("indexBuffer");
//	if(pElement)
//		;//processIndexBuffer(pElement);

    // Create the entity
    Entity *pEntity = 0;
    try
    {
        MeshManager::getSingleton().load(meshFile, m_sGroupName);
        pEntity = mSceneMgr->createEntity(name, meshFile);
        pEntity->setCastShadows(castShadows);
        pParent->attachObject(pEntity);

        if(!materialFile.empty())
            pEntity->setMaterialName(materialFile);
    }
    catch(Ogre::Exception &/*e*/)
    {
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error loading an entity!");
    }

    // Process userDataReference (?)
    pElement = XMLNode->FirstChildElement("userDataReference");
    if(pElement)
        processUserDataReference(pElement, pEntity);


}

void DotSceneLoader::processParticleSystem(TiXmlElement *XMLNode, SceneNode *pParent)
{
    // Process attributes
    String name = getAttrib(XMLNode, "name");
    String id = getAttrib(XMLNode, "id");
    String file = getAttrib(XMLNode, "file");

    // Create the particle system
    try
    {
        ParticleSystem *pParticles = mSceneMgr->createParticleSystem(name, file);
        pParent->attachObject(pParticles);
    }
    catch(Ogre::Exception &/*e*/)
    {
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error creating a particle system!");
    }
}

void DotSceneLoader::processBillboardSet(TiXmlElement *XMLNode, SceneNode *pParent)
{
    //! @todo Implement this
}

void DotSceneLoader::processPlane(TiXmlElement *XMLNode, SceneNode *pParent)
{
    String name = getAttrib(XMLNode, "name");
    String id = getAttrib(XMLNode, "id");
    Real distance = getAttribReal(XMLNode, "distance");
    Real width = getAttribReal(XMLNode, "width");
    Real height = getAttribReal(XMLNode, "height");
    int xSegments = getAttribInt(XMLNode, "xSegments",1);
    int ySegments = getAttribInt(XMLNode, "ySegments",1);
    int numTexCoordSets = getAttribInt(XMLNode, "numTexCoordSets",1);
    Real uTile = getAttribReal(XMLNode, "uTile", 1);
    Real vTile = getAttribReal(XMLNode, "vTile", 1);
    String material = getAttrib(XMLNode, "material", "");
    bool normals = getAttribBool(XMLNode, "normals", true);

    TiXmlElement *pElement;

    Plane plane;
    pElement = XMLNode->FirstChildElement("normal");
    if(pElement)
        plane.normal = parseVector3(pElement);
    plane.d = distance;

    Vector3 upVector = Vector3::UNIT_Y;
    pElement = XMLNode->FirstChildElement("upVector");
    if(pElement)
        upVector = parseVector3(pElement);

    Entity *pEntity = 0;
    try
    {
        MeshPtr pPlaneMesh = MeshManager::getSingleton().createPlane(name, m_sGroupName, plane, width, height, xSegments, ySegments, normals, numTexCoordSets, uTile, vTile, upVector);
        pEntity = mSceneMgr->createEntity(name, name);
        pParent->attachObject(pEntity);

        if(!material.empty())
            pEntity->setMaterialName(material);
    }
    catch(Ogre::Exception &e)
    {
        LogManager::getSingleton().logMessage("[DotSceneLoader] Error creating a plane!");
        LogManager::getSingleton().logMessage(e.getFullDescription());
    }
}

void DotSceneLoader::processFog(TiXmlElement *XMLNode)
{
    // Process attributes
    Real expDensity = getAttribReal(XMLNode, "expDensity", 0.001);
    Real linearStart = getAttribReal(XMLNode, "linearStart", 0.0);
    Real linearEnd = getAttribReal(XMLNode, "linearEnd", 1.0);

    FogMode mode = FOG_NONE;
    String sMode = getAttrib(XMLNode, "mode");
    if(sMode == "none")
        mode = FOG_NONE;
    else if(sMode == "exp")
        mode = FOG_EXP;
    else if(sMode == "exp2")
        mode = FOG_EXP2;
    else if(sMode == "linear")
        mode = FOG_LINEAR;

    TiXmlElement *pElement;

    // Process colourDiffuse (?)
    ColourValue colourDiffuse = ColourValue::White;
    pElement = XMLNode->FirstChildElement("colourDiffuse");
    if(pElement)
        colourDiffuse = parseColour(pElement);

    // Setup the fog
    mSceneMgr->setFog(mode, colourDiffuse, expDensity, linearStart, linearEnd);
}

void DotSceneLoader::processSkyBox(TiXmlElement *XMLNode)
{
    // Process attributes
    String material = getAttrib(XMLNode, "material");
    Real distance = getAttribReal(XMLNode, "distance", 5000);
    bool drawFirst = getAttribBool(XMLNode, "drawFirst", true);

    TiXmlElement *pElement;

    // Process rotation (?)
    Quaternion rotation = Quaternion::IDENTITY;
    pElement = XMLNode->FirstChildElement("rotation");
    if(pElement)
        rotation = parseQuaternion(pElement);

    // Setup the sky box
    mSceneMgr->setSkyBox(true, material, distance, drawFirst, rotation, m_sGroupName);
}

void DotSceneLoader::processSkyDome(TiXmlElement *XMLNode)
{
    // Process attributes
    String material = XMLNode->Attribute("material");
    Real curvature = getAttribReal(XMLNode, "curvature", 10);
    Real tiling = getAttribReal(XMLNode, "tiling", 8);
    Real distance = getAttribReal(XMLNode, "distance", 4000);
    bool drawFirst = getAttribBool(XMLNode, "drawFirst", true);

    TiXmlElement *pElement;

    // Process rotation (?)
    Quaternion rotation = Quaternion::IDENTITY;
    pElement = XMLNode->FirstChildElement("rotation");
    if(pElement)
        rotation = parseQuaternion(pElement);

    // Setup the sky dome
    mSceneMgr->setSkyDome(true, material, curvature, tiling, distance, drawFirst, rotation, 16, 16, -1, m_sGroupName);
}

void DotSceneLoader::processSkyPlane(TiXmlElement *XMLNode)
{
    // Process attributes
    String material = getAttrib(XMLNode, "material");
    Real planeX = getAttribReal(XMLNode, "planeX", 0);
    Real planeY = getAttribReal(XMLNode, "planeY", -1);
    Real planeZ = getAttribReal(XMLNode, "planeX", 0);
    Real planeD = getAttribReal(XMLNode, "planeD", 5000);
    Real scale = getAttribReal(XMLNode, "scale", 1000);
    Real bow = getAttribReal(XMLNode, "bow", 0);
    Real tiling = getAttribReal(XMLNode, "tiling", 10);
    bool drawFirst = getAttribBool(XMLNode, "drawFirst", true);

    // Setup the sky plane
    Plane plane;
    plane.normal = Vector3(planeX, planeY, planeZ);
    plane.d = planeD;
    mSceneMgr->setSkyPlane(true, plane, material, scale, tiling, drawFirst, bow, 1, 1, m_sGroupName);
}

void DotSceneLoader::processClipping(TiXmlElement *XMLNode, Camera* pCamera)
{
    //! @todo Implement this

    // Process attributes
    Real fNear = getAttribReal(XMLNode, "near", 0);
    Real fFar = getAttribReal(XMLNode, "far", 1);

    if (pCamera)
    {
        pCamera->setNearClipDistance(fNear);
        pCamera->setFarClipDistance(fFar);
    }
    else
    {
        environment.nearClipDistance = fNear;
        environment.farClipDistance = fFar;
    }
}

void DotSceneLoader::processLightRange(TiXmlElement *XMLNode, Light *pLight)
{
    // Process attributes
    Real inner = getAttribReal(XMLNode, "inner");
    Real outer = getAttribReal(XMLNode, "outer");
    Real falloff = getAttribReal(XMLNode, "falloff", 1.0);

    // Setup the light range
    pLight->setSpotlightRange(Angle(inner), Angle(outer), falloff);
}

void DotSceneLoader::processLightAttenuation(TiXmlElement *XMLNode, Light *pLight)
{
    // Process attributes
    Real range = getAttribReal(XMLNode, "range");
    Real constant = getAttribReal(XMLNode, "constant");
    Real linear = getAttribReal(XMLNode, "linear");
    Real quadratic = getAttribReal(XMLNode, "quadratic");

    // Setup the light attenuation
    pLight->setAttenuation(range, constant, linear, quadratic);
}


String DotSceneLoader::getAttrib(TiXmlElement *XMLNode, const String &attrib, const String &defaultValue)
{
    if(XMLNode->Attribute(attrib.c_str()))
        return XMLNode->Attribute(attrib.c_str());
    else
        return defaultValue;
}

Real DotSceneLoader::getAttribReal(TiXmlElement *XMLNode, const String &attrib, Real defaultValue)
{
    if(XMLNode->Attribute(attrib.c_str()))
        return StringConverter::parseReal(XMLNode->Attribute(attrib.c_str()));
    else
        return defaultValue;
}

int DotSceneLoader::getAttribInt(TiXmlElement *XMLNode, const String &attrib, int defaultValue)
{
    if(XMLNode->Attribute(attrib.c_str()))
        return StringConverter::parseInt(XMLNode->Attribute(attrib.c_str()));
    else
        return defaultValue;
}

bool DotSceneLoader::getAttribBool(TiXmlElement *XMLNode, const String &attrib, bool defaultValue)
{
    if(!XMLNode->Attribute(attrib.c_str()))
        return defaultValue;

    if(String(XMLNode->Attribute(attrib.c_str())) == "true")
        return true;

    return false;
}


Vector3 DotSceneLoader::parseVector3(TiXmlElement *XMLNode)
{
    return Vector3(
            StringConverter::parseReal(XMLNode->Attribute("x")),
            StringConverter::parseReal(XMLNode->Attribute("y")),
            StringConverter::parseReal(XMLNode->Attribute("z"))
            );
}

Quaternion DotSceneLoader::parseQuaternion(TiXmlElement *XMLNode)
{
    Quaternion orientation;

    if(XMLNode->Attribute("qx"))
    {
        orientation.x = StringConverter::parseReal(XMLNode->Attribute("qx"));
        orientation.y = StringConverter::parseReal(XMLNode->Attribute("qy"));
        orientation.z = StringConverter::parseReal(XMLNode->Attribute("qz"));
        orientation.w = StringConverter::parseReal(XMLNode->Attribute("qw"));
    }
    else if(XMLNode->Attribute("axisX"))
    {
        Vector3 axis;
        axis.x = StringConverter::parseReal(XMLNode->Attribute("axisX"));
        axis.y = StringConverter::parseReal(XMLNode->Attribute("axisY"));
        axis.z = StringConverter::parseReal(XMLNode->Attribute("axisZ"));
        Real angle = StringConverter::parseReal(XMLNode->Attribute("angle"));;
        orientation.FromAngleAxis(Ogre::Angle(angle), axis);
    }
    else if(XMLNode->Attribute("angleX"))
    {
        Vector3 angle;
        angle.x = StringConverter::parseReal(XMLNode->Attribute("angleX"));
        angle.y = StringConverter::parseReal(XMLNode->Attribute("angleY"));
        angle.z = StringConverter::parseReal(XMLNode->Attribute("angleZ"));
        Quaternion qx,qy,qz;
        qx.FromAngleAxis(Ogre::Angle(angle.x), Vector3::UNIT_X);
        qy.FromAngleAxis(Ogre::Angle(angle.y), Vector3::UNIT_Y);
        qz.FromAngleAxis(Ogre::Angle(angle.z), Vector3::UNIT_Z);
        //! @todo Check order
        orientation = qx * qy * qz;
    }

    return orientation;
}

ColourValue DotSceneLoader::parseColour(TiXmlElement *XMLNode)
{
    return ColourValue(
            StringConverter::parseReal(XMLNode->Attribute("r")),
            StringConverter::parseReal(XMLNode->Attribute("g")),
            StringConverter::parseReal(XMLNode->Attribute("b")),
            XMLNode->Attribute("a") != NULL ? StringConverter::parseReal(XMLNode->Attribute("a")) : 1
            );
}

String DotSceneLoader::getProperty(const String &ndNm, const String &prop)
{
    for ( unsigned int i = 0 ; i < nodeProperties.size(); i++ )
    {
        if ( nodeProperties[i].nodeName == ndNm && nodeProperties[i].propertyNm == prop )
        {
            return nodeProperties[i].valueName;
        }
    }

    return "";
}

void DotSceneLoader::processUserDataReference(TiXmlElement *XMLNode, Entity *pEntity)
{
    String str = XMLNode->Attribute("id");
    pEntity->setUserAny(Any(str));
}
