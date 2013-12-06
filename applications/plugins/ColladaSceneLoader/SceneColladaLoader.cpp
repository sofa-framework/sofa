/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SceneColladaLoader.h"
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/mapping/SkinningMapping.inl>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/projectiveconstraintset/SkeletalMotionConstraint.inl>
//#include <sofa/component/typedef/Particles_float.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/SetDirectory.h>

#include <stack>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;
using namespace sofa::component::container;
using namespace sofa::component::mass;
using namespace sofa::component::topology;
using namespace sofa::component::visualmodel;
using namespace sofa::component::mapping;
using namespace sofa::component::collision;
using namespace sofa::component::projectiveconstraintset;

SOFA_DECL_CLASS(SceneColladaLoader)

int SceneColladaLoaderClass = core::RegisterObject("Specific scene loader for Collada file format.")
        .add< SceneColladaLoader >()
        ;

SceneColladaLoader::SceneColladaLoader() : SceneLoader()
    , subSceneRoot()
	, importer()
	, animationSpeed(initData(&animationSpeed, 1.0f, "animationSpeed", "animation speed"))
{
	
}

SceneColladaLoader::~SceneColladaLoader()
{
    importer.FreeScene();
}

void SceneColladaLoader::init()
{
    if(subSceneRoot)
    {
        // retrieving parent node
        BaseContext* currentContext = getContext();
        GNode* parentNode = dynamic_cast<GNode*>(currentContext);
        if(!parentNode)
        {
            sout << "Error: SceneColladaLoader::init, this loader has no parentNode : " << name.getValue() << sendl;
            return;
        }

        // placing root node of the loaded sub scene
        std::string subSceneName(name.getValue());
        if(!subSceneName.empty())
            subSceneName += "_";
        subSceneName += "scene";
        subSceneRoot->setName(subSceneName);
        parentNode->addChild(subSceneRoot);

        // find how many siblings scene loaders there are upward the current one
        int sceneLoaderNum = 0;

        GNode::ObjectIterator objectIt;
        for(objectIt = parentNode->object.begin(); objectIt != parentNode->object.end(); ++objectIt)
        {
            if(dynamic_cast<SceneLoader*>(objectIt->get()))
                ++sceneLoaderNum;

            if(this == *objectIt)
                break;
        }

        // place an iterator on the last scene loader generated node
        int sceneLoaderNodeNum = 0;

        GNode::ChildIterator childIt = parentNode->child.begin();
        if(1 != sceneLoaderNum)
        {
            for(; childIt != parentNode->child.end() - 1; ++childIt)
            {
                ++sceneLoaderNodeNum;
                if(subSceneRoot == *childIt || sceneLoaderNum == sceneLoaderNodeNum)
                    break;
            }
        }

        // swap our generated node position till it is at the right place
        for (GNode::ChildIterator it = parentNode->child.end() - 1; it != childIt; --it)
            parentNode->child.swap(it, it - 1);
    }
}

bool SceneColladaLoader::load()
{
    sout << "Loading Collada (.dae) file: " << m_filename << sendl;

    bool fileRead = false;

    // loading file
    const char* filename = m_filename.getFullPath().c_str();
    std::ifstream file(filename);

    if(!file.good())
    {
        serr << "Error: SceneColladaLoader: Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    // reading file
    fileRead = readDAE (file,filename);
    file.close();

    return fileRead;
}

bool SceneColladaLoader::readDAE (std::ifstream &file, const char* filename)
{
    sout << "SceneColladaLoader::readDAE" << sendl;

    // if a scene is already loaded with this importer, free it
    importer.FreeScene();

    // importing scene
    const aiScene* currentAiScene = importer.ReadFile(m_filename.getValue(), 0);

    if(!currentAiScene)
    {
        sout << "Collada import failed : " << importer.GetErrorString() << sendl;
        return false;
    }

    // traversing the scene graph
    if(currentAiScene->mRootNode)
    {
        // use a stack to process the nodes of the scene graph in the right order, we link an assimp node with a GNode with a NodeInfo
        std::stack<NodeInfo> nodes;

        subSceneRoot = sofa::core::objectmodel::New<GNode>();
        nodes.push(NodeInfo(currentAiScene->mRootNode, subSceneRoot));

        int articulationIndex = 0;
        int parentIndex = 0;
        int index = 1;

        int meshId = 0;

        // processing each node of the scene graph
        while(!nodes.empty())
        {
            // fast access node parent pointer
            NodeInfo& currentNode = nodes.top();
            NodeInfo* parentNode = currentNode.mParentNode;
            aiNode* currentAiNode = currentNode.mAiNode;
            GNode::SPtr currentGNode = currentNode.mGNode;
            int& childIndex = currentNode.mChildIndex;
            aiMatrix4x4& currentTransformation = currentNode.mTransformation;

            // process the node just one time
            if(0 == childIndex)
            {
                {
                    // if the aiNode contains a name do not change it because we will need it to retrieve the node when processing bones
                    std::stringstream nameStream(std::string(currentAiNode->mName.data, currentAiNode->mName.length));
                    if(nameStream.str().empty())
                        nameStream << childIndex++;
                    currentGNode->setName(nameStream.str());

                    //std::cout << currentGNode->getName() << std::endl;
                }

                // extract the node transformation to apply them later on its meshes
                aiVector3D aiScale, aiTranslation;
                aiQuaternion aiRotation;
                currentTransformation.Decompose(aiScale, aiRotation, aiTranslation);
                Quaternion quat(aiRotation.x, aiRotation.y, aiRotation.z, aiRotation.w);

                Vec3d translation(aiTranslation.x, aiTranslation.y, aiTranslation.z);
                Vec3d rotation(quat.toEulerVector() / (M_PI / 180.0));
                Vec3d scale(aiScale.x, aiScale.y, aiScale.z);

                // useful to generate a unique index for each component of a node
                int componentIndex = 0;

                // for each mesh in the node
                for(unsigned int j = 0; j < currentAiNode->mNumMeshes; ++j, ++meshId)
                {
                    GNode::SPtr meshGNode = sofa::core::objectmodel::New<GNode>();
                    currentGNode->addChild(meshGNode);

                    std::stringstream meshNameStream;
                    meshNameStream << "mesh " << (int)meshId;
                    meshGNode->setName(meshNameStream.str());

                    aiMesh* currentAiMesh = currentAiScene->mMeshes[currentAiNode->mMeshes[j]];

                    // generating a name
                    std::string meshName(currentAiMesh->mName.data, currentAiMesh->mName.length);

                    // node used for collision
                    GNode::SPtr collisionGNode;

                    // generating a MechanicalObject and a SkinningMapping if the mesh contains bones and filling up theirs properties
                    MechanicalObject<Rigid3dTypes>::SPtr currentBoneMechanicalObject;
                    if(currentAiMesh->HasBones())
                    {
                        /*std::cout << "animation num : " << currentAiScene->mNumAnimations << std::endl;
                        std::cout << "animation duration : " << currentAiScene->mAnimations[0]->mDuration << std::endl;
                        std::cout << "animation ticks per second : " << currentAiScene->mAnimations[0]->mTicksPerSecond << std::endl;
                        std::cout << "animation channel num : " << currentAiScene->mAnimations[0]->mNumChannels << std::endl;*/

                        currentBoneMechanicalObject = sofa::core::objectmodel::New<MechanicalObject<Rigid3dTypes> >();
                        {
                            // adding the generated MechanicalObject to its parent GNode
                            meshGNode->addObject(currentBoneMechanicalObject);

                            std::stringstream nameStream(meshName);
                            if(meshName.empty())
                                nameStream << componentIndex++;
                            currentBoneMechanicalObject->setName(nameStream.str());

                            // filling up position coordinate array
                            currentBoneMechanicalObject->resize(currentAiMesh->mNumBones);

                            {
                                Data<vector<Rigid3dTypes::Coord> >* d_x = currentBoneMechanicalObject->write(core::VecCoordId::position());
                                vector<Rigid3dTypes::Coord> &x = *d_x->beginEdit();
                                for(unsigned int k = 0; k < currentAiMesh->mNumBones; ++k)
                                {
                                    aiMatrix4x4 offsetMatrix = currentAiMesh->mBones[k]->mOffsetMatrix;
                                    offsetMatrix.Inverse();

                                    // mesh space to world space
                                    offsetMatrix = currentTransformation * offsetMatrix;

                                    // extract the bone transformation
                                    aiVector3D aiBoneScale, aiBoneTranslation;
                                    aiQuaternion aiBoneRotation;
                                    offsetMatrix.Decompose(aiBoneScale, aiBoneRotation, aiBoneTranslation);

                                    Vec3d boneTranslation(aiBoneTranslation.x, aiBoneTranslation.y, aiBoneTranslation.z);
                                    Quaternion boneQuat(aiBoneRotation.x, aiBoneRotation.y, aiBoneRotation.z, aiBoneRotation.w);

                                    x[k] = Rigid3dTypes::Coord(boneTranslation, boneQuat);
                                }
                            }
                        }

                        // generating a SkeletalMotionConstraint and filling up its properties
                        SkeletalMotionConstraint<Rigid3dTypes>::SPtr currentSkeletalMotionConstraint = sofa::core::objectmodel::New<SkeletalMotionConstraint<Rigid3dTypes> >();
                        {
                            // adding the generated SkeletalMotionConstraint to its parent GNode
                            meshGNode->addObject(currentSkeletalMotionConstraint);

                            std::stringstream nameStream(meshName);
                            if(meshName.empty())
                                nameStream << componentIndex++;
                            currentSkeletalMotionConstraint->setName(nameStream.str());

//							currentSkeletalMotionConstraint->setAnimationSpeed(animationSpeed.getValue());

                            aiNode* parentAiNode = NULL;
                            if(parentNode)
                                parentAiNode = parentNode->mAiNode;

                            helper::vector<SkeletonJoint<Rigid3dTypes> > skeletonJoints;
                            helper::vector<SkeletonBone> skeletonBones;
                            fillSkeletalInfo(currentAiScene, parentAiNode, currentAiNode, currentTransformation, currentAiMesh, skeletonJoints, skeletonBones);
                            currentSkeletalMotionConstraint->setSkeletalMotion(skeletonJoints, skeletonBones);
                        }

                        collisionGNode = sofa::core::objectmodel::New<GNode>();
                        meshGNode->addChild(collisionGNode);

                        std::stringstream collisionNameStream;
                        collisionNameStream << "collision " << (int)meshId;
                        collisionGNode->setName(collisionNameStream.str());
                    }
                    else
                    {
                        collisionGNode = meshGNode;
                    }

                    // generating a MechanicalObject and filling up its properties
                    MechanicalObject<Vec3dTypes>::SPtr currentMechanicalObject = sofa::core::objectmodel::New<MechanicalObject<Vec3dTypes> >();
                    {
                        // adding the generated MechanicalObject to its parent GNode
                        collisionGNode->addObject(currentMechanicalObject);

                        std::stringstream nameStream(meshName);
                        if(meshName.empty())
                            nameStream << componentIndex++;
                        currentMechanicalObject->setName(nameStream.str());

                        currentMechanicalObject->setTranslation(translation.x(), translation.y(), translation.z());
                        currentMechanicalObject->setRotation(rotation.x(), rotation.y(), rotation.z());
                        currentMechanicalObject->setScale(scale.x(), scale.y(), scale.z());

                        // filling up position coordinate array
                        if(0 != currentAiMesh->mNumVertices)
                        {
                            currentMechanicalObject->resize(currentAiMesh->mNumVertices);

                            {
                                Data<vector<Vec3dTypes::Coord> >* d_x = currentMechanicalObject->write(core::VecCoordId::position());
                                vector<Vec3dTypes::Coord> &x = *d_x->beginEdit();
                                for(unsigned int k = 0; k < currentAiMesh->mNumVertices; ++k)
                                    x[k] = Vec3dTypes::Coord(Vec3d(currentAiMesh->mVertices[k][0], currentAiMesh->mVertices[k][1], currentAiMesh->mVertices[k][2]));
                            }
                        }
                    }

                    // generating a MeshTopology and filling up its properties
                    MeshTopology::SPtr currentMeshTopology = sofa::core::objectmodel::New<MeshTopology>();
                    {
                        // adding the generated MeshTopology to its parent GNode
                        collisionGNode->addObject(currentMeshTopology);

                        std::stringstream nameStream(meshName);
                        if(meshName.empty())
                            nameStream << componentIndex++;
                        currentMeshTopology->setName(nameStream.str());

                        // filling up position array
                        currentMeshTopology->seqPoints.setParent(&currentMechanicalObject->x);

                        // filling up triangle array
                        vector<topology::Triangle> triangles;
                        unsigned int numTriangles = 0;
                        for(unsigned int k = 0; k < currentAiMesh->mNumFaces; ++k)
                            if(3 == currentAiMesh->mFaces[k].mNumIndices)
                                ++numTriangles;

                        if(0 != numTriangles)
                        {
                            triangles.resize(numTriangles);

                            unsigned int triangleOffset = 0;
                            for(unsigned int k = 0; k < currentAiMesh->mNumFaces; ++k)
                            {
                                if(3 != currentAiMesh->mFaces[k].mNumIndices)
                                    continue;

                                memcpy(&triangles[0] + triangleOffset, currentAiMesh->mFaces[k].mIndices, sizeof(topology::Triangle));
                                ++triangleOffset;
                            }

                            {
                                vector<topology::Triangle>& seqTriangles = *currentMeshTopology->seqTriangles.beginEdit();
                                seqTriangles.reserve(triangles.size());

                                for(unsigned int k = 0; k < triangles.size(); ++k)
                                    seqTriangles.push_back(triangles[k]);
                            }
                        }

                        // filling up quad array
                        vector<topology::Quad> quads;
                        unsigned int numQuads = 0;
                        for(unsigned int k = 0; k < currentAiMesh->mNumFaces; ++k)
                            if(4 == currentAiMesh->mFaces[k].mNumIndices)
                                ++numQuads;

                        if(0 != numQuads)
                        {
                            quads.resize(numQuads);

                            unsigned int quadOffset = 0;
                            for(unsigned int k = 0; k < currentAiMesh->mNumFaces; ++k)
                            {
                                if(4 != currentAiMesh->mFaces[k].mNumIndices)
                                    continue;

                                memcpy(&quads[0] + quadOffset, currentAiMesh->mFaces[k].mIndices, sizeof(topology::Quad));
                                ++quadOffset;
                            }

                            {
                                vector<topology::Quad>& seqQuads = *currentMeshTopology->seqQuads.beginEdit();
                                seqQuads.reserve(quads.size());

                                for(unsigned int k = 0; k < quads.size(); ++k)
                                    seqQuads.push_back(quads[k]);
                            }
                        }
                    }

//					std::string xnodeName(currentAiNode->mName.data);
//					std::string xmeshName(currentAiMesh->mName.data);
//					std::cout << "nodeName: " << xnodeName << std::endl;
//					std::cout << " - meshName: " << xmeshName << std::endl;
//					std::cout << std::endl;

					UniformMass<defaulttype::Vec3dTypes, double>::SPtr currentUniformMass = sofa::core::objectmodel::New<UniformMass<defaulttype::Vec3dTypes, double> >();
					{
						// adding the generated UniformMass to its parent GNode
						collisionGNode->addObject(currentUniformMass);

						std::stringstream nameStream(meshName);
						if(meshName.empty())
							nameStream << componentIndex++;
						currentUniformMass->setName(nameStream.str());

						currentUniformMass->setTotalMass(10.0);
					}

					TTriangleModel<defaulttype::Vec3dTypes>::SPtr currentTTriangleModel = sofa::core::objectmodel::New<TTriangleModel<defaulttype::Vec3dTypes> >();
					{
						// adding the generated TTriangleModel to its parent GNode
						collisionGNode->addObject(currentTTriangleModel);

						std::stringstream nameStream(meshName);
						if(meshName.empty())
							nameStream << componentIndex++;
						currentTTriangleModel->setName(nameStream.str());
					}

					TLineModel<defaulttype::Vec3dTypes>::SPtr currentTLineModel = sofa::core::objectmodel::New<TLineModel<defaulttype::Vec3dTypes> >();
					{
						// adding the generated TLineModel to its parent GNode
						collisionGNode->addObject(currentTLineModel);

						std::stringstream nameStream(meshName);
						if(meshName.empty())
							nameStream << componentIndex++;
						currentTLineModel->setName(nameStream.str());
					}

					TPointModel<defaulttype::Vec3dTypes>::SPtr currentTPointModel = sofa::core::objectmodel::New<TPointModel<defaulttype::Vec3dTypes> >();
					{
						// adding the generated TPointModel to its parent GNode
						collisionGNode->addObject(currentTPointModel);

						std::stringstream nameStream(meshName);
						if(meshName.empty())
							nameStream << componentIndex++;
						currentTPointModel->setName(nameStream.str());
					}

					if(currentAiMesh->HasBones())
                    {
                        SkinningMapping<Rigid3dTypes, Vec3dTypes>::SPtr currentSkinningMapping = sofa::core::objectmodel::New<SkinningMapping<Rigid3dTypes, Vec3dTypes> >();
                        {
                            // adding the generated SkinningMapping to its parent GNode
                            collisionGNode->addObject(currentSkinningMapping);

                            std::stringstream nameStream(meshName);
                            if(meshName.empty())
                                nameStream << componentIndex++;
                            currentSkinningMapping->setName(nameStream.str());

                            currentSkinningMapping->setModels(currentBoneMechanicalObject.get(), currentMechanicalObject.get());

                            vector<SVector<SkinningMapping<Rigid3dTypes, Vec3dTypes>::InReal> > weights;
                            vector<SVector<unsigned int> > indices;
                            vector<unsigned int> nbref;

                            indices.resize(currentAiMesh->mNumVertices);
                            weights.resize(currentAiMesh->mNumVertices);
                            nbref.resize(currentAiMesh->mNumVertices);
                            for(unsigned int k = 0; k < nbref.size(); ++k)
                                nbref[k] = 0;

                            for(unsigned int k = 0; k < currentAiMesh->mNumBones; ++k)
                            {
                                aiBone*& bone = currentAiMesh->mBones[k];

                                for(unsigned int l = 0; l < bone->mNumWeights; ++l)
                                {
                                    unsigned int id = bone->mWeights[l].mVertexId;
                                    float weight = bone->mWeights[l].mWeight;

                                    if(id >= currentAiMesh->mNumVertices)
                                    {
                                        sout << "Error: SceneColladaLoader::readDAE, a mesh could not be load : " << nameStream.str() << " - in node : " << currentGNode->getName() << sendl;
                                        return false;
                                    }

                                    weights[id].push_back(weight);
                                    indices[id].push_back(k);
                                    ++nbref[id];
                                }
                            }

                            currentSkinningMapping->setWeights(weights, indices, nbref);

                            //currentSkeletonMapping->setModels(currentMechanicalObject.get(), currentOglModel.get());

                            //for(int k = 0; k < currentAiMesh->mNumBones; ++k)
                            //{
                            //	currentAiMesh->mBones[k];
                            //}
                        }
                    }

					// node used for visualization
                    GNode::SPtr visuGNode = sofa::core::objectmodel::New<GNode>();
                    collisionGNode->addChild(visuGNode);

                    std::stringstream visuNameStream;
                    visuNameStream << "visu " << (int)meshId;
                    visuGNode->setName(visuNameStream.str());
					
                    // generating an OglModel and filling up its properties
                    OglModel::SPtr currentOglModel = sofa::core::objectmodel::New<OglModel>();
                    {
                        // adding the generated OglModel to its parent GNode
                        visuGNode->addObject(currentOglModel);

                        std::stringstream nameStream(meshName);
                        if(meshName.empty())
                            nameStream << componentIndex++;
                        currentOglModel->setName(nameStream.str());

                        if(0 != currentAiMesh->mNumVertices)
                        {
                            ResizableExtVector<Vec3f> normals;
                            normals.resize(currentAiMesh->mNumVertices);
                            memcpy(&normals[0], currentAiMesh->mNormals, currentAiMesh->mNumVertices * sizeof(aiVector3D));
                            currentOglModel->setVnormals(&normals);
                        }

                        // filling up position array (by default : done by an IdentityMapping, but necessary if we use a SkinningMapping)
                        //if(currentAiMesh->HasBones())
                        //{
                        //currentOglModel->m_positions.setParent(&currentMechanicalObject->x);
                        //currentOglModel->m_positions.setReadOnly(true);
                        //}

                        // vertex / triangle / quad array are automatically filled up with the MeshTopology data

                        // filling up vertex array (redundancy with positions coordinates but required ?!)
                        /*if(0 != currentAiMesh->mNumVertices)
                        {
                        	ResizableExtVector<Vec3d> vertices;
                        	vertices.resize(currentAiMesh->mNumVertices);
                        	memcpy(&vertices[0], currentAiMesh->mVertices, currentAiMesh->mNumVertices * sizeof(aiVector3D));
                        	currentOglModel->setVertices(&vertices);
                        }

                        // filling up triangle array
                        ResizableExtVector<OglModel::Triangle> triangles;
                        unsigned int numTriangles = 0;
                        for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
                        	if(3 == currentAiMesh->mFaces[k].mNumIndices)
                        		++numTriangles;

                        if(0 != numTriangles)
                        {
                        	triangles.resize(numTriangles);

                        	unsigned int triangleOffset = 0;
                        	for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
                        	{
                        		if(3 != currentAiMesh->mFaces[k].mNumIndices)
                        			continue;

                        		memcpy(&triangles[0] + triangleOffset, currentAiMesh->mFaces[k].mIndices, sizeof(Triangle));
                        		++triangleOffset;
                        	}
                        	currentOglModel->setTriangles(&triangles);
                        }

                        // filling up quad array
                        ResizableExtVector<OglModel::Quad> quads;
                        unsigned int numQuads = 0;
                        for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
                        	if(4 == currentAiMesh->mFaces[k].mNumIndices)
                        		++numQuads;

                        if(0 != numQuads)
                        {
                        	quads.resize(numQuads);

                        	unsigned int quadOffset = 0;
                        	for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
                        	{
                        		if(4 != currentAiMesh->mFaces[k].mNumIndices)
                        			continue;

                        		memcpy(&quads[0] + quadOffset, currentAiMesh->mFaces[k].mIndices, sizeof(Quad));
                        		++quadOffset;
                        	}

                        	currentOglModel->setQuads(&quads);
                        }*/
                    }

					IdentityMapping<Vec3dTypes, ExtVec3fTypes>::SPtr currentIdentityMapping = sofa::core::objectmodel::New<IdentityMapping<Vec3dTypes, ExtVec3fTypes> >();
					{
						// adding the generated IdentityMapping to its parent GNode
						visuGNode->addObject(currentIdentityMapping);

						std::stringstream nameStream(meshName);
						if(meshName.empty())
							nameStream << componentIndex++;
						currentIdentityMapping->setName(nameStream.str());

						currentIdentityMapping->setModels(currentMechanicalObject.get(), currentOglModel.get());
					}
                }
            }

            // pop the current node when each one of its children have been processed
            if(childIndex == currentAiNode->mNumChildren)
            {
                nodes.pop();
            }
            // process next sub node
            else
            {
                // generating sub GNode and filling up its properties
                // store it in the stack to process its children later
                NodeInfo subNode(currentAiNode->mChildren[childIndex], sofa::core::objectmodel::New<GNode>(), &currentNode);
                nodes.push(subNode);

                // adding the generated node to its parent GNode
                currentGNode->addChild(subNode.mGNode);

                // this child will be processed, go to the next one
                ++childIndex;
            }
        }
    }

    removeEmptyNodes();

    return true;
}

bool SceneColladaLoader::fillSkeletalInfo(const aiScene* scene, aiNode* meshParentNode, aiNode* meshNode, aiMatrix4x4 meshTransformation, aiMesh* mesh, helper::vector<SkeletonJoint<Rigid3dTypes> >& skeletonJoints, helper::vector<SkeletonBone>& skeletonBones) const
{
    //std::cout << "fillSkeletalInfo : begin" << std::endl;

    // return now if their is no scene, no mesh or no skeletonBones
    if(!scene || !mesh || !mesh->HasBones())
    {
        sout << "no mesh to load !" << sendl;
        return false;
    }

    std::map<aiNode*, int> aiNodeToSkeletonJointIndex;

    // compute the mesh transformation into a rigid
    Mat4x4d meshWorldTranformation(meshTransformation[0]);
//	for(int j = 0; j < 4; ++j)
//		for(int i = 0; i < 4; ++i)
//			meshWorldTranformation[j][i] = meshTransformation[j][i];

    Rigid3dTypes::Coord meshTransformationRigid;
    meshTransformationRigid.getCenter()[0] = meshWorldTranformation[0][3];
    meshTransformationRigid.getCenter()[1] = meshWorldTranformation[1][3];
    meshTransformationRigid.getCenter()[2] = meshWorldTranformation[2][3];
    Mat3x3d rot; rot = meshWorldTranformation;
    meshTransformationRigid.getOrientation().fromMatrix(rot);

    //std::cout << "ANIMATION" << std::endl;

    // register every SkeletonJoint
    for(unsigned int j = 0; j < scene->mNumAnimations; ++j)
    {
        // for now we just want to handle one animation
        if(1 == j)
            break;

        //std::cout << "num channels : " << scene->mAnimations[j]->mNumChannels << std::endl;

        aiAnimation*& animation = scene->mAnimations[j];
        for(unsigned int k = 0; k < animation->mNumChannels; ++k)
        {
            aiNodeAnim*& channel = animation->mChannels[k];
            aiString& nodeName = channel->mNodeName;
            aiNode* node = scene->mRootNode->FindNode(nodeName);

            // create the corresponding SkeletonJoint if it does not exist
            std::map<aiNode*, int>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            if(aiNodeToSkeletonJointIndex.end() == aiNodeToSkeletonJointIndexIterator)
            {
                skeletonJoints.push_back(SkeletonJoint<Rigid3dTypes>());
                aiNodeToSkeletonJointIndex[node] = skeletonJoints.size() - 1;
                aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            }
            else
            {
                return false;
            }
            SkeletonJoint<Rigid3dTypes>& skeletonJoint = skeletonJoints[aiNodeToSkeletonJointIndexIterator->second];

            aiVectorKey positionKey, scaleKey;
            aiQuatKey	rotationKey;

            int numKey = std::max(channel->mNumPositionKeys, channel->mNumRotationKeys);
            //int numKey = std::max(channel->mNumScalingKeys , std::max(channel->mNumPositionKeys, channel->mNumRotationKeys));

            skeletonJoint.mTimes.resize(numKey);
            skeletonJoint.mChannels.resize(numKey);
            for(int l = 0; l < numKey; ++l)
            {
                double time = 0.0;
                aiMatrix4x4 transformation;

                if(l < channel->mNumPositionKeys)
                {
                    positionKey = channel->mPositionKeys[l];
                    time = positionKey.mTime;
                    aiMatrix4x4 position;
                    aiMatrix4x4::Translation(positionKey.mValue, position);
                    transformation = position;
                }

                if(l < channel->mNumRotationKeys)
                {
                    rotationKey = channel->mRotationKeys[l];
                    time = rotationKey.mTime;
                    aiMatrix4x4 rotation(rotationKey.mValue.GetMatrix());
                    transformation *= rotation;
                }

                // 							if(l < channel->mNumScalingKeys)
                // 							{
                // 								scaleKey = channel->mScalingKeys[l];
                // 								time = scaleKey.mTime;
                // 								aiMatrix4x4 scale;
                // 								aiMatrix4x4::Scaling(scaleKey.mValue, scale);
                // 								transformation *= scale;
                // 							}

                Mat4x4d localTranformation(transformation[0]);

                Rigid3dTypes::Coord localRigid;
                localRigid.getCenter()[0] = localTranformation[0][3];
                localRigid.getCenter()[1] = localTranformation[1][3];
                localRigid.getCenter()[2] = localTranformation[2][3];
                Mat3x3d rot; rot = localTranformation;
                localRigid.getOrientation().fromMatrix(rot);

                skeletonJoint.mTimes[l] = time;
                skeletonJoint.mChannels[l] = localRigid;
            }
        }
    }

    // register every bone and link them to their SkeletonJoint (or create it if it has not been created)
    skeletonBones.resize(mesh->mNumBones);
    for(unsigned int i = 0; i < mesh->mNumBones; ++i)
    {
        aiBone*& bone = mesh->mBones[i];
        const aiString& boneName = bone->mName;

        // register the parents SkeletonJoint for each bone
        aiNode* node = scene->mRootNode->FindNode(boneName);

        // create the corresponding SkeletonJoint if it does not exist
        std::map<aiNode*, int>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
        if(aiNodeToSkeletonJointIndex.end() == aiNodeToSkeletonJointIndexIterator)
        {
            skeletonJoints.push_back(SkeletonJoint<Rigid3dTypes>());
            aiNodeToSkeletonJointIndex[node] = skeletonJoints.size() - 1;
            aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
        }

        skeletonBones[i] = aiNodeToSkeletonJointIndexIterator->second;
    }

    // register every SkeletonJoint and their parents and fill up theirs properties
    for(int i = 0; i < skeletonJoints.size(); ++i)
    {
        SkeletonJoint<Rigid3dTypes>& skeletonJoint = skeletonJoints[i];

        aiNode*	node = NULL;

        // find the ai node corresponding to the SkeletonJoint
        for(std::map<aiNode*, int>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.begin(); aiNodeToSkeletonJointIndexIterator != aiNodeToSkeletonJointIndex.end(); ++aiNodeToSkeletonJointIndexIterator)
        {
            if(i == aiNodeToSkeletonJointIndexIterator->second)
            {
                node = aiNodeToSkeletonJointIndexIterator->first;
                break;
            }
        }

        if(NULL == node)
            return false;

        int previousSkeletonJointIndex = -1;

        // find parents node
        while(NULL != node)
        {
            // stop if we reach the mesh node or its parent
            if(meshNode == node || meshParentNode == node)
                break;

            // create the corresponding SkeletonJoint if it does not exist
            std::map<aiNode*, int>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            if(aiNodeToSkeletonJointIndex.end() == aiNodeToSkeletonJointIndexIterator)
            {
                skeletonJoints.push_back(SkeletonJoint<Rigid3dTypes>());
                aiNodeToSkeletonJointIndex[node] = skeletonJoints.size() - 1;
                aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            }
            SkeletonJoint<Rigid3dTypes>& currentSkeletonJoint = skeletonJoints[aiNodeToSkeletonJointIndexIterator->second];

            // register the current node
            aiMatrix4x4 aiLocalTransformation = node->mTransformation;

            // compute the rigid corresponding to the SkeletonJoint
            Mat4x4d localTranformation(aiLocalTransformation[0]);

            Rigid3dTypes::Coord localRigid;
            localRigid.getCenter()[0] = localTranformation[0][3];
            localRigid.getCenter()[1] = localTranformation[1][3];
            localRigid.getCenter()[2] = localTranformation[2][3];
            Mat3x3d rot; rot = localTranformation;
            localRigid.getOrientation().fromMatrix(rot);

            // apply the mesh transformation to the skeleton root joint only
            // we know that this joint is the root if the corresponding aiNode is the mesh node or its parent
            aiNode* parentNode = node->mParent;
            if(meshNode == parentNode || meshParentNode == parentNode)
            {
                // compute the mesh transformation
                localRigid = meshTransformationRigid.mult(localRigid);

                // apply the mesh transformation to each channel if the skeleton root joint contains animation
                for(int i = 0; i < currentSkeletonJoint.mChannels.size(); ++i)
                    currentSkeletonJoint.mChannels[i] = meshTransformationRigid.mult(currentSkeletonJoint.mChannels[i]);
            }

            currentSkeletonJoint.setRestPosition(localRigid);

            if(-1 != previousSkeletonJointIndex)
                skeletonJoints[previousSkeletonJointIndex].mParentIndex = aiNodeToSkeletonJointIndexIterator->second;

            previousSkeletonJointIndex = aiNodeToSkeletonJointIndexIterator->second;

            node = node->mParent;
        }
    }

    return true;
}

void SceneColladaLoader::removeEmptyNodes()
{
    // remove intermediary or empty nodes
    {
        std::stack<std::pair<GNode::SPtr, int> > nodes;

        nodes.push(std::pair<GNode::SPtr, int>(subSceneRoot, 0));
        while(!nodes.empty())
        {
            GNode::SPtr& node = nodes.top().first;
            int& index = nodes.top().second;

            if(node->getChildren().size() <= index)
            {
                nodes.pop();

                if(nodes.empty())
                    break;

                GNode::SPtr& parentNode = nodes.top().first;
                int& parentIndex = nodes.top().second;

                // remove the node if it has no objects
                if(node->object.empty())
                {
                    if(0 != node->getChildren().size())
                    {
                        // links its child nodes directly to its parent node before remove the current intermediary node
                        while(!node->getChildren().empty())
                        {
                            GNode::SPtr childNode = static_cast<GNode*>(node->getChildren()[0]);
                            parentNode->moveChild(childNode);
                        }
                    }

                    parentNode->removeChild(node);
                }
                else
                {
                    ++parentIndex;
                }
            }
            else
            {
                GNode::SPtr child = static_cast<GNode*>(node->getChildren()[index]);
                nodes.push(std::pair<GNode::SPtr, int>(child, 0));
            }
        }
    }
}

} // namespace loader

} // namespace component

} // namespace sofa

