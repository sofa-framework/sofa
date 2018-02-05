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
#include "SceneColladaLoader.h"
#include <sofa/simulation/Simulation.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaGeneralRigid/SkinningMapping.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaBoundaryCondition/FixedConstraint.h>
#include <SofaBoundaryCondition/SkeletalMotionConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/SetDirectory.h>
#include <stack>
#include <algorithm>

#ifdef SOFA_HAVE_PLUGIN_FLEXIBLE
#include <Flexible/deformationMapping/LinearMapping.h>
#endif


#ifdef SOFA_HAVE_PLUGIN_IMAGE
#include <image/ImageContainer.h>
#include <image/MeshToImageEngine.h>
#include <image/ImageFilter.h>
#include <image/ImageViewer.h>
#endif

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
using namespace sofa::simulation;

SOFA_DECL_CLASS(SceneColladaLoader)

int SceneColladaLoaderClass = core::RegisterObject("Specific scene loader for Collada file format.")
        .add< SceneColladaLoader >()
        ;

SceneColladaLoader::SceneColladaLoader() : SceneLoader()
    , subSceneRoot()
	, importer()
	, animationSpeed(initData(&animationSpeed, 1.0f, "animationSpeed", "animation speed"))
	, generateCollisionModels(initData(&generateCollisionModels, true, "generateCollisionModels", "generate point/line/triangle collision models for imported meshes"))
#ifdef SOFA_HAVE_PLUGIN_FLEXIBLE
    , useFlexible(initData(&useFlexible, false, "useFlexible", "Use the Flexible plugin (it will replace the SkinningMapping with a LinearMapping)"))
#endif
#ifdef SOFA_HAVE_PLUGIN_IMAGE
    , generateShapeFunction(initData(&generateShapeFunction, false, "generateShapeFunction", "Generate a shape function that could be used in another simulation"))
    , voxelSize(initData(&voxelSize, (SReal)0.02, "voxelSize", "voxelSize used for shape function generation"))
#endif
{
	
}

SceneColladaLoader::~SceneColladaLoader()
{
    importer.FreeScene();
}

void SceneColladaLoader::init()
{
	if(0 == subSceneRoot)
		return;

    // retrieving parent node
    core::objectmodel::BaseContext* currentContext = getContext();
    Node* parentNode = dynamic_cast<Node*>(currentContext);
    if(!parentNode)
    {
        sout << "Error: SceneColladaLoader::init, loader " << name.getValue() << "has no parentNode" << sendl;
		if(currentContext)
			sout << "Context is : " << currentContext->getName() << sendl;

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

    Node::ObjectIterator objectIt;
    for(objectIt = parentNode->object.begin(); objectIt != parentNode->object.end(); ++objectIt)
    {
        if(dynamic_cast<SceneLoader*>(objectIt->get()))
            ++sceneLoaderNum;

        if(this == *objectIt)
            break;
    }

    // place an iterator on the last scene loader generated node
    int sceneLoaderNodeNum = 0;

    Node::ChildIterator childIt = parentNode->child.begin();
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
    for (Node::ChildIterator it = parentNode->child.end() - 1; it != childIt; --it)
        parentNode->child.swap(it, it - 1);
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

bool SceneColladaLoader::readDAE (std::ifstream &/*file*/, const char* /*filename*/)
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
        // use a stack to process the nodes of the scene graph in the right order, we link an assimp node with a Node with a NodeInfo
        std::stack<NodeInfo> nodes;

        subSceneRoot = getSimulation()->createNewNode("subroot");
        nodes.push(NodeInfo(currentAiScene->mRootNode, subSceneRoot));

        int meshId = 0;

        // processing each node of the scene graph
        while(!nodes.empty())
        {
            // fast access node parent pointer
            NodeInfo& currentNodeInfo = nodes.top();
            NodeInfo* parentNodeInfo = currentNodeInfo.mParentNode;
            aiNode* currentAiNode = currentNodeInfo.mAiNode;
            Node::SPtr currentNode = currentNodeInfo.mNode;
            std::size_t& childIndex = currentNodeInfo.mChildIndex;
            aiMatrix4x4& currentTransformation = currentNodeInfo.mTransformation;

            // process the node just one time
            if(0 == childIndex)
            {
                {
                    // if the aiNode contains a name do not change it because we will need it to retrieve the node when processing bones
                    std::stringstream nameStream(std::string(currentAiNode->mName.data, currentAiNode->mName.length));
                    if(nameStream.str().empty())
                        nameStream << childIndex++;
                    currentNode->setName(nameStream.str());

                    //std::cout << currentNode->getName() << std::endl;
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


                // TODO: have only one rigidMechanicalObject with no redundancy and same indices for all meshes
                // so all meshes can be mapped to the same rigidMechanicalObject


                // for each mesh in the node
                for(unsigned int j = 0; j < currentAiNode->mNumMeshes; ++j, ++meshId)
                {
                    std::stringstream meshNameStream;
                    meshNameStream << "mesh_" << (int)meshId;
                    
                    Node::SPtr meshNode = getSimulation()->createNewNode(meshNameStream.str());
                    currentNode->addChild(meshNode);

                    aiMesh* currentAiMesh = currentAiScene->mMeshes[currentAiNode->mMeshes[j]];

                    // generating a name
                    std::string meshName(currentAiMesh->mName.data, currentAiMesh->mName.length);

					// the node representing a part of the current mesh construction (skinning, collision, visualization ...)
                    Node::SPtr currentSubNode = meshNode;

                    // generating a MechanicalObject and a SkinningMapping if the mesh contains bones and filling up theirs properties
                    MechanicalObject<Rigid3Types>::SPtr currentBoneMechanicalObject;
                    if(currentAiMesh->HasBones())
                    {
                        /*std::cout << "animation num : " << currentAiScene->mNumAnimations << std::endl;
                        std::cout << "animation duration : " << currentAiScene->mAnimations[0]->mDuration << std::endl;
                        std::cout << "animation ticks per second : " << currentAiScene->mAnimations[0]->mTicksPerSecond << std::endl;
                        std::cout << "animation channel num : " << currentAiScene->mAnimations[0]->mNumChannels << std::endl;*/

                        currentBoneMechanicalObject = sofa::core::objectmodel::New<MechanicalObject<Rigid3Types> >();
                        {
                            // adding the generated MechanicalObject to its parent Node
                            currentSubNode->addObject(currentBoneMechanicalObject);

                            std::stringstream nameStream(meshName);
                            if(meshName.empty())
                                nameStream << componentIndex++;
                            currentBoneMechanicalObject->setName(nameStream.str());

                            // filling up position coordinate array
                            currentBoneMechanicalObject->resize(currentAiMesh->mNumBones);

                            {
                                Data<Rigid3Types::VecCoord>* d_x = currentBoneMechanicalObject->write(core::VecCoordId::position());
                                Rigid3Types::VecCoord &x = *d_x->beginEdit();
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

                                    x[k] = Rigid3Types::Coord(boneTranslation, boneQuat);
                                }
                                d_x->endEdit();
                            }
                        }

                        if(generateCollisionModels.getValue())
                        {
                            UniformMass<Rigid3Types, Rigid3Mass>::SPtr currentUniformMass = sofa::core::objectmodel::New<UniformMass<Rigid3Types, Rigid3Mass> >();
                            {
                                // adding the generated UniformMass to its parent Node
                                currentSubNode->addObject(currentUniformMass);

                                std::stringstream nameStream(meshName);
                                if(meshName.empty())
                                    nameStream << componentIndex++;
                                currentUniformMass->setName(nameStream.str());

                                currentUniformMass->setTotalMass(80.0);
                            }
                        }

                        // generating a SkeletalMotionConstraint and filling up its properties
                        SkeletalMotionConstraint<Rigid3Types>::SPtr currentSkeletalMotionConstraint = sofa::core::objectmodel::New<SkeletalMotionConstraint<Rigid3Types> >();
                        {
                            // adding the generated SkeletalMotionConstraint to its parent Node
                            currentSubNode->addObject(currentSkeletalMotionConstraint);

                            std::stringstream nameStream(meshName);
                            if(meshName.empty())
                                nameStream << componentIndex++;
                            currentSkeletalMotionConstraint->setName(nameStream.str());

							currentSkeletalMotionConstraint->setAnimationSpeed(animationSpeed.getValue());

                            aiNode* parentAiNode = NULL;
                            if(parentNodeInfo)
                                parentAiNode = parentNodeInfo->mAiNode;

                            helper::vector<SkeletonJoint<Rigid3Types> > skeletonJoints;
                            helper::vector<SkeletonBone> skeletonBones;
                            fillSkeletalInfo(currentAiScene, parentAiNode, currentAiNode, currentTransformation, currentAiMesh, skeletonJoints, skeletonBones);
                            currentSkeletalMotionConstraint->setSkeletalMotion(skeletonJoints, skeletonBones);
                        }
                    }
                    else
                    {
                        currentBoneMechanicalObject = sofa::core::objectmodel::New<MechanicalObject<Rigid3Types> >();
						{
							// adding the generated MechanicalObject to its parent Node
							currentSubNode->addObject(currentBoneMechanicalObject);

							std::stringstream nameStream(meshName);
							if(meshName.empty())
								nameStream << componentIndex++;
							currentBoneMechanicalObject->setName(nameStream.str());

							// filling up position coordinate array
							currentBoneMechanicalObject->resize(1);

							{
                                Data<helper::vector<Rigid3Types::Coord> >* d_x = currentBoneMechanicalObject->write(core::VecCoordId::position());
                                helper::vector<Rigid3Types::Coord> &x = *d_x->beginEdit();
								
								Vec3d boneTranslation(0.0, 0.0, 0.0);
								Quaternion boneQuat(0.0, 0.0, 1.0, 1.0);

                                x[0] = Rigid3Types::Coord(boneTranslation, boneQuat);

								d_x->endEdit();
							}
						}

                        UniformMass<Rigid3Types, Rigid3Mass>::SPtr currentUniformMass = sofa::core::objectmodel::New<UniformMass<Rigid3Types, Rigid3Mass> >();
						{
							// adding the generated UniformMass to its parent Node
							currentSubNode->addObject(currentUniformMass);

							std::stringstream nameStream(meshName);
							if(meshName.empty())
								nameStream << componentIndex++;
							currentUniformMass->setName(nameStream.str());
						}

                        FixedConstraint<Rigid3Types>::SPtr currentFixedConstraint = sofa::core::objectmodel::New<FixedConstraint<Rigid3Types> >();
						{
							// adding the generated FixedConstraint to its parent Node
							currentSubNode->addObject(currentFixedConstraint);

							std::stringstream nameStream(meshName);
							if(meshName.empty())
								nameStream << componentIndex++;
							currentFixedConstraint->setName(nameStream.str());

                                                        currentFixedConstraint->d_fixAll.setValue(true);
						}
                    }

					std::stringstream rigidNameStream;
					if(currentAiMesh->HasBones())
						rigidNameStream << "skinning_" << (int)meshId;
					else
						rigidNameStream << "rigid_" << (int)meshId;

					Node::SPtr rigidNode = getSimulation()->createNewNode(rigidNameStream.str());
					currentSubNode->addChild(rigidNode);

					currentSubNode = rigidNode;

                    std::map<Vec3d,unsigned> vertexMap; // no to superimpose identical vertices

                    // generating a MechanicalObject and filling up its properties
                    MechanicalObject<Vec3Types>::SPtr currentMechanicalObject = sofa::core::objectmodel::New<MechanicalObject<Vec3Types> >();
                    {
                        // adding the generated MechanicalObject to its parent Node
                        currentSubNode->addObject(currentMechanicalObject);

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
                            int vertexIdx=0;
                            for(unsigned int k = 0; k < currentAiMesh->mNumVertices; ++k)
                            {
                                Vec3d v(currentAiMesh->mVertices[k][0], currentAiMesh->mVertices[k][1], currentAiMesh->mVertices[k][2]);
                                if( vertexMap.find(v) == vertexMap.end() ) vertexMap[v] = vertexIdx++;
                            }

                            currentMechanicalObject->resize(vertexMap.size());

                            {
                                Data<helper::vector<Vec3Types::Coord> >* d_x = currentMechanicalObject->write(core::VecCoordId::position());
                                helper::vector<Vec3Types::Coord> &x = *d_x->beginEdit();
                                for( std::map<Vec3d,unsigned>::iterator it=vertexMap.begin() , itend=vertexMap.end() ; it!=itend ; ++it )
                                    x[it->second] = it->first;

                                d_x->endEdit();
                            }
                        }
                    }

                    // generating a MeshTopology and filling up its properties
                    MeshTopology::SPtr currentMeshTopology = sofa::core::objectmodel::New<MeshTopology>();
                    {
                        // adding the generated MeshTopology to its parent Node
                        currentSubNode->addObject(currentMeshTopology);

                        std::stringstream nameStream(meshName);
                        if(meshName.empty())
                            nameStream << componentIndex++;
                        currentMeshTopology->setName(nameStream.str());

                        // filling up position array
                        currentMeshTopology->seqPoints.setParent(&currentMechanicalObject->x);

                        unsigned int numTriangles = 0, numQuads = 0;
                        for(unsigned int k = 0; k < currentAiMesh->mNumFaces; ++k)
                            if( 3 == currentAiMesh->mFaces[k].mNumIndices )
                                ++numTriangles;
                            else if( 4 == currentAiMesh->mFaces[k].mNumIndices )
                                ++numQuads;


                        helper::vector<core::topology::BaseMeshTopology::Triangle>& seqTriangles = *currentMeshTopology->seqTriangles.beginEdit();
#ifdef SOFA_HAVE_PLUGIN_IMAGE
                        if( generateShapeFunction.getValue() )
                        {
                            if( numTriangles || numQuads ) seqTriangles.reserve(numTriangles+numQuads*2);

                            for( unsigned int k = 0 ; k < currentAiMesh->mNumFaces ; ++k )
                            {
                                if( currentAiMesh->mFaces[k].mNumIndices==3 )
                                {
                                    unsigned int *faceIndices = currentAiMesh->mFaces[k].mIndices;

                                    const unsigned int &faceIndex0 = faceIndices[0];
                                    const unsigned int &faceIndex1 = faceIndices[1];
                                    const unsigned int &faceIndex2 = faceIndices[2];

                                    Vec3d v0(currentAiMesh->mVertices[faceIndex0][0], currentAiMesh->mVertices[faceIndex0][1], currentAiMesh->mVertices[faceIndex0][2]);
                                    Vec3d v1(currentAiMesh->mVertices[faceIndex1][0], currentAiMesh->mVertices[faceIndex1][1], currentAiMesh->mVertices[faceIndex1][2]);
                                    Vec3d v2(currentAiMesh->mVertices[faceIndex2][0], currentAiMesh->mVertices[faceIndex2][1], currentAiMesh->mVertices[faceIndex2][2]);

                                    seqTriangles.push_back( core::topology::BaseMeshTopology::Triangle( vertexMap[v0], vertexMap[v1], vertexMap[v2] ) );
                                }
                                else if( currentAiMesh->mFaces[k].mNumIndices==4 )
                                {
                                    unsigned int *faceIndices = currentAiMesh->mFaces[k].mIndices;

                                    const unsigned int &faceIndex0 = faceIndices[0];
                                    const unsigned int &faceIndex1 = faceIndices[1];
                                    const unsigned int &faceIndex2 = faceIndices[2];
                                    const unsigned int &faceIndex3 = faceIndices[3];

                                    Vec3d v0(currentAiMesh->mVertices[faceIndex0][0], currentAiMesh->mVertices[faceIndex0][1], currentAiMesh->mVertices[faceIndex0][2]);
                                    Vec3d v1(currentAiMesh->mVertices[faceIndex1][0], currentAiMesh->mVertices[faceIndex1][1], currentAiMesh->mVertices[faceIndex1][2]);
                                    Vec3d v2(currentAiMesh->mVertices[faceIndex2][0], currentAiMesh->mVertices[faceIndex2][1], currentAiMesh->mVertices[faceIndex2][2]);
                                    Vec3d v3(currentAiMesh->mVertices[faceIndex3][0], currentAiMesh->mVertices[faceIndex3][1], currentAiMesh->mVertices[faceIndex3][2]);

                                    unsigned int i0 = vertexMap[v0];
                                    unsigned int i1 = vertexMap[v1];
                                    unsigned int i2 = vertexMap[v2];
                                    unsigned int i3 = vertexMap[v3];

                                    seqTriangles.push_back( core::topology::BaseMeshTopology::Triangle( i0, i1, i2 ) );
                                    seqTriangles.push_back( core::topology::BaseMeshTopology::Triangle( i0, i2, i3 ) );
                                }
                            }
                        }
                        else
#endif
                        {
                            if( numTriangles ) seqTriangles.reserve(numTriangles);
                            helper::vector<core::topology::BaseMeshTopology::Quad>& seqQuads = *currentMeshTopology->seqQuads.beginEdit();
                            if( numQuads ) seqQuads.reserve(numQuads);

                            for( unsigned int k = 0 ; k < currentAiMesh->mNumFaces ; ++k )
                            {
                                if( currentAiMesh->mFaces[k].mNumIndices==3 )
                                {
                                    unsigned int *faceIndices = currentAiMesh->mFaces[k].mIndices;

                                    const unsigned int &faceIndex0 = faceIndices[0];
                                    const unsigned int &faceIndex1 = faceIndices[1];
                                    const unsigned int &faceIndex2 = faceIndices[2];

                                    Vec3d v0(currentAiMesh->mVertices[faceIndex0][0], currentAiMesh->mVertices[faceIndex0][1], currentAiMesh->mVertices[faceIndex0][2]);
                                    Vec3d v1(currentAiMesh->mVertices[faceIndex1][0], currentAiMesh->mVertices[faceIndex1][1], currentAiMesh->mVertices[faceIndex1][2]);
                                    Vec3d v2(currentAiMesh->mVertices[faceIndex2][0], currentAiMesh->mVertices[faceIndex2][1], currentAiMesh->mVertices[faceIndex2][2]);

                                    seqTriangles.push_back( core::topology::BaseMeshTopology::Triangle( vertexMap[v0], vertexMap[v1], vertexMap[v2] ) );
                                }
                                else if( currentAiMesh->mFaces[k].mNumIndices==4 )
                                {
                                    unsigned int *faceIndices = currentAiMesh->mFaces[k].mIndices;

                                    const unsigned int &faceIndex0 = faceIndices[0];
                                    const unsigned int &faceIndex1 = faceIndices[1];
                                    const unsigned int &faceIndex2 = faceIndices[2];
                                    const unsigned int &faceIndex3 = faceIndices[3];

                                    Vec3d v0(currentAiMesh->mVertices[faceIndex0][0], currentAiMesh->mVertices[faceIndex0][1], currentAiMesh->mVertices[faceIndex0][2]);
                                    Vec3d v1(currentAiMesh->mVertices[faceIndex1][0], currentAiMesh->mVertices[faceIndex1][1], currentAiMesh->mVertices[faceIndex1][2]);
                                    Vec3d v2(currentAiMesh->mVertices[faceIndex2][0], currentAiMesh->mVertices[faceIndex2][1], currentAiMesh->mVertices[faceIndex2][2]);
                                    Vec3d v3(currentAiMesh->mVertices[faceIndex3][0], currentAiMesh->mVertices[faceIndex3][1], currentAiMesh->mVertices[faceIndex3][2]);

                                    seqQuads.push_back( core::topology::BaseMeshTopology::Quad( vertexMap[v0], vertexMap[v1], vertexMap[v2], vertexMap[v3] ) );
                                }
                            }

                            currentMeshTopology->seqQuads.endEdit();
                        }

                        currentMeshTopology->seqTriangles.endEdit();
                    }


					if(generateCollisionModels.getValue())
					{
                        TTriangleModel<defaulttype::Vec3Types>::SPtr currentTTriangleModel = sofa::core::objectmodel::New<TTriangleModel<defaulttype::Vec3Types> >();
						{
							// adding the generated TTriangleModel to its parent Node
							currentSubNode->addObject(currentTTriangleModel);

							std::stringstream nameStream(meshName);
							if(meshName.empty())
								nameStream << componentIndex++;
							currentTTriangleModel->setName(nameStream.str());
						}

                        TLineModel<defaulttype::Vec3Types>::SPtr currentTLineModel = sofa::core::objectmodel::New<TLineModel<defaulttype::Vec3Types> >();
						{
							// adding the generated TLineModel to its parent Node
                            currentSubNode->addObject(currentTLineModel);

							std::stringstream nameStream(meshName);
							if(meshName.empty())
								nameStream << componentIndex++;
							currentTLineModel->setName(nameStream.str());
						}

                        TPointModel<defaulttype::Vec3Types>::SPtr currentTPointModel = sofa::core::objectmodel::New<TPointModel<defaulttype::Vec3Types> >();
						{
							// adding the generated TPointModel to its parent Node
                            currentSubNode->addObject(currentTPointModel);

							std::stringstream nameStream(meshName);
							if(meshName.empty())
								nameStream << componentIndex++;
							currentTPointModel->setName(nameStream.str());
                        }
					}

//					std::string xnodeName(currentAiNode->mName.data);
//					std::string xmeshName(currentAiMesh->mName.data);
//					std::cout << "nodeName: " << xnodeName << std::endl;
//					std::cout << " - meshName: " << xmeshName << std::endl;
//					std::cout << std::endl;

					if(currentAiMesh->HasBones())
                    {
#ifdef SOFA_HAVE_PLUGIN_IMAGE
                        if( generateShapeFunction.getValue() )
                        {
                            SReal vsize = this->voxelSize.getValue();

                            // rasterized mesh
                            Node::SPtr labelNode = currentSubNode->createChild("label");
                            engine::MeshToImageEngine<defaulttype::ImageB>::SPtr M2I = sofa::core::objectmodel::New<engine::MeshToImageEngine<defaulttype::ImageB> >();
                            M2I->setName( "rasterizer" );
                            M2I->voxelSize.setValue( helper::vector<SReal>(1,vsize) );
                            M2I->padSize.setValue(2);
                            M2I->rotateImage.setValue(false);
                            M2I->f_nbMeshes.setValue(1);
//                            M2I->createInputMeshesData();
                            M2I->backgroundValue.setValue(0);
                            engine::MeshToImageEngine<defaulttype::ImageB>::SeqValues values(1,1);
                            (*M2I->vf_values[0]).setValue(values);
                            (*M2I->vf_positions[0]).setParent( &currentMechanicalObject->x );
                            (*M2I->vf_triangles[0]).setParent( &currentMeshTopology->seqTriangles );
                            labelNode->addObject(M2I);

                            ImageContainer<defaulttype::ImageB>::SPtr IC0 = sofa::core::objectmodel::New<ImageContainer<defaulttype::ImageB> >();
                            IC0->setName( "image" );
                            IC0->image.setParent(&M2I->image);
                            IC0->transform.setParent(&M2I->transform);
                            labelNode->addObject(IC0);

//                            misc::ImageViewer<defaulttype::ImageB>::SPtr IV0 = sofa::core::objectmodel::New<misc::ImageViewer<defaulttype::ImageB> >();
//                            IV0->setName( "viewer" );
//                            IV0->image.setParent( &M2I->image );
//                            IV0->transform.setParent( &M2I->transform );
//                            labelNode->addObject(IV0);

                            // rasterized weights on surface
                            for( unsigned int b = 0 ; b < currentAiMesh->mNumBones /*&& b<1*/ ; ++b )
                            {
                                aiBone*& bone = currentAiMesh->mBones[b];

                                std::stringstream nodeName;
                                nodeName << "dof " << b;
                                Node::SPtr dofNode = currentSubNode->createChild(nodeName.str());

                                engine::MeshToImageEngine<defaulttype::ImageD>::SPtr M2I = sofa::core::objectmodel::New<engine::MeshToImageEngine<defaulttype::ImageD> >();
                                M2I->setName( "rasterizer" );
                                M2I->voxelSize.setValue( helper::vector<SReal>(1,vsize) );
                                M2I->padSize.setValue(2);
                                M2I->rotateImage.setValue(false);
                                M2I->f_nbMeshes.setValue(1);
//                                M2I->createInputMeshesData();

                                std::stringstream nameStream(meshName);
                                if(meshName.empty())
                                    nameStream << componentIndex++;



                                engine::MeshToImageEngine<defaulttype::ImageD>::SeqValues values(vertexMap.size());


                                for(unsigned int l = 0; l < bone->mNumWeights; ++l)
                                {

                                    const unsigned int& vertexid = bone->mWeights[l].mVertexId;

                                    if(vertexid >= currentAiMesh->mNumVertices)
                                    {
                                        sout << "Error: SceneColladaLoader::readDAE, a mesh could not be load : " << nameStream.str() << " - in node : " << currentNode->getName() << sendl;
                                        return false;
                                    }

                                    Vec3d v(currentAiMesh->mVertices[vertexid][0], currentAiMesh->mVertices[vertexid][1], currentAiMesh->mVertices[vertexid][2]);

                                    unsigned int id = vertexMap[v];
                                    float weight = bone->mWeights[l].mWeight;

                                    values[id] = weight;
                                }



                                (*M2I->vf_values[0]).setValue(values);
                                M2I->backgroundValue.setValue(-1);
                                (*M2I->vf_positions[0]).setParent( &currentMechanicalObject->x );
                                (*M2I->vf_triangles[0]).setParent( &currentMeshTopology->seqTriangles );


                                dofNode->addObject(M2I);

                                ImageContainer<defaulttype::ImageD>::SPtr IC0 = sofa::core::objectmodel::New<ImageContainer<defaulttype::ImageD> >();
                                IC0->setName( "image" );
                                IC0->image.setParent(&M2I->image);
                                IC0->transform.setParent(&M2I->transform);
                                dofNode->addObject(IC0);

//                                misc::ImageViewer<defaulttype::ImageD>::SPtr IV0 = sofa::core::objectmodel::New<misc::ImageViewer<defaulttype::ImageD> >();
//                                IV0->setName( "viewer" );
//                                IV0->image.setParent( &M2I->image );
//                                IV0->transform.setParent( &M2I->transform );
//                                dofNode->addObject(IV0);

//                                engine::ImageFilter<defaulttype::ImageD,defaulttype::ImageD>::SPtr IF = sofa::core::objectmodel::New<engine::ImageFilter<defaulttype::ImageD,defaulttype::ImageD> >();
//                                IF->setName( "diffusion" );
//                                IF->inputImage.setParent(&M2I->image);
//                                IF->inputTransform.setParent(&M2I->transform);
//                                IF->filter.beginEdit()->setSelectedItem( 23 ); IF->filter.endEdit();
//                                engine::ImageFilter<defaulttype::ImageD,defaulttype::ImageD>::ParamTypes params(4);
//                                params[0] = 0; params[1] = params[2] = 1; params[3] = 1e-5;
//                                IF->param.setValue(params);
//                                IF->f_printLog.setValue(true);
//                                dofNode->addObject(IF);

//                                ImageContainer<defaulttype::ImageD>::SPtr IC = sofa::core::objectmodel::New<ImageContainer<defaulttype::ImageD> >();
//                                IC->setName( "image" );
//                                IC->image.setParent(&IF->outputImage);
//                                IC->transform.setParent(&IF->outputTransform);
//                                dofNode->addObject(IC);

//                                misc::ImageViewer<defaulttype::ImageD>::SPtr IV = sofa::core::objectmodel::New<misc::ImageViewer<defaulttype::ImageD> >();
//                                IV->setName( "viewer" );
//                                IV->image.setParent( &IF->outputImage );
//                                IV->transform.setParent( &IF->outputTransform );
//                                dofNode->addObject(IV);

                            }
                        } else
#endif
#ifdef SOFA_HAVE_PLUGIN_FLEXIBLE
                        if(useFlexible.getValue())
                        {
                            LinearMapping<Rigid3Types, Vec3Types>::SPtr currentLinearMapping = sofa::core::objectmodel::New<LinearMapping<Rigid3Types, Vec3Types> >();

                            // adding the generated LinearMapping to its parent Node
                            currentSubNode->addObject(currentLinearMapping);

                            std::stringstream nameStream(meshName);
                            if(meshName.empty())
                                nameStream << componentIndex++;
                            currentLinearMapping->setName(nameStream.str());

                            currentLinearMapping->setModels(currentBoneMechanicalObject.get(), currentMechanicalObject.get());

                            LinearMapping<Rigid3Types, Vec3Types>::VecVReal& weights = *currentLinearMapping->f_w.beginEdit();
                            LinearMapping<Rigid3Types, Vec3Types>::VecVRef& indices = *currentLinearMapping->f_index.beginEdit();

                            indices.resize(vertexMap.size());
                            weights.resize(vertexMap.size());

                            for(unsigned int k = 0; k < currentAiMesh->mNumBones; ++k)
                            {
                                aiBone*& bone = currentAiMesh->mBones[k];


//                                helper:vector<float> boneW((size_t)currentAiMesh->mNumVertices,.0f);

                                for(unsigned int l = 0; l < bone->mNumWeights; ++l)
                                {
                                    const unsigned int& vertexid = bone->mWeights[l].mVertexId;

                                    if(vertexid >= currentAiMesh->mNumVertices)
                                    {
                                        sout << "Error: SceneColladaLoader::readDAE, a mesh could not be load : " << nameStream.str() << " - in node : " << currentNode->getName() << sendl;
                                        return false;
                                    }

                                    Vec3d v(currentAiMesh->mVertices[vertexid][0], currentAiMesh->mVertices[vertexid][1], currentAiMesh->mVertices[vertexid][2]);

                                    unsigned int id = vertexMap[v];
                                    float weight = bone->mWeights[l].mWeight;

//                                    boneW[id]=weight;

                                    if( std::find( indices[id].begin(), indices[id].end(), k ) == indices[id].end() )
                                    {
                                        indices[id].push_back(k);
                                        weights[id].push_back(weight);
                                    }
                                }

//                                std::cerr<<"MESH "<<meshId<<" - DOF "<<k<<": "<<boneW<<std::endl;

                            }

                            currentLinearMapping->f_w.endEdit();
                            currentLinearMapping->f_index.endEdit();
						}
                        else
#endif
						{
                            SkinningMapping<Rigid3Types, Vec3Types>::SPtr currentSkinningMapping = sofa::core::objectmodel::New<SkinningMapping<Rigid3Types, Vec3Types> >();
							{
								// adding the generated SkinningMapping to its parent Node
								currentSubNode->addObject(currentSkinningMapping);

								std::stringstream nameStream(meshName);
								if(meshName.empty())
									nameStream << componentIndex++;
								currentSkinningMapping->setName(nameStream.str());

								currentSkinningMapping->setModels(currentBoneMechanicalObject.get(), currentMechanicalObject.get());

                                helper::vector<helper::SVector<SkinningMapping<Rigid3Types, Vec3Types>::InReal> > weights;
                                helper::vector<helper::SVector<unsigned int> > indices;
                                helper::vector<unsigned int> nbref;

                                indices.resize(vertexMap.size());
                                weights.resize(vertexMap.size());
                                nbref.resize(vertexMap.size(),0);

								for(unsigned int k = 0; k < currentAiMesh->mNumBones; ++k)
								{
									aiBone*& bone = currentAiMesh->mBones[k];

									for(unsigned int l = 0; l < bone->mNumWeights; ++l)
									{

                                        const unsigned int& vertexid = bone->mWeights[l].mVertexId;

                                        if(vertexid >= currentAiMesh->mNumVertices)
										{
											sout << "Error: SceneColladaLoader::readDAE, a mesh could not be load : " << nameStream.str() << " - in node : " << currentNode->getName() << sendl;
											return false;
										}

                                        Vec3d v(currentAiMesh->mVertices[vertexid][0], currentAiMesh->mVertices[vertexid][1], currentAiMesh->mVertices[vertexid][2]);

                                        unsigned int id = vertexMap[v];
                                        float weight = bone->mWeights[l].mWeight;

                                        if( std::find( indices[id].begin(), indices[id].end(), k ) == indices[id].end() )
                                        {
                                            weights[id].push_back(weight);
                                            indices[id].push_back(k);
                                            ++nbref[id];
                                        }

									}
								}

								currentSkinningMapping->setWeights(weights, indices, nbref);
							}
						}
                    }
					else
					{
                        RigidMapping<Rigid3Types, Vec3Types>::SPtr currentRigidMapping = sofa::core::objectmodel::New<RigidMapping<Rigid3Types, Vec3Types> >();
						{
							// adding the generated RigidMapping to its parent Node
							currentSubNode->addObject(currentRigidMapping);

							std::stringstream nameStream(meshName);
							if(meshName.empty())
								nameStream << componentIndex++;
							currentRigidMapping->setName(nameStream.str());

							currentRigidMapping->setModels(currentBoneMechanicalObject.get(), currentMechanicalObject.get());
						}
					}

					// node used for visualization
					std::stringstream visuNameStream;
					visuNameStream << "visualization " << (int)meshId;

					Node::SPtr visuNode = getSimulation()->createNewNode(visuNameStream.str());
					currentSubNode->addChild(visuNode);

					currentSubNode = visuNode;
					
                    // generating an OglModel and filling up its properties
                    OglModel::SPtr currentOglModel = sofa::core::objectmodel::New<OglModel>();
                    {
                        // adding the generated OglModel to its parent Node
                        currentSubNode->addObject(currentOglModel);

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
//                         if(0 != currentAiMesh->mNumVertices)
//                         {
//                         	ResizableExtVector<Vec3d> vertices;
//                         	vertices.resize(currentAiMesh->mNumVertices);
//                         	memcpy(&vertices[0], currentAiMesh->mVertices, currentAiMesh->mNumVertices * sizeof(aiVector3D));
//                         	currentOglModel->setVertices(&vertices);
//                         }
// 
//                         // filling up triangle array
//                         ResizableExtVector<OglModel::Triangle> triangles;
//                         unsigned int numTriangles = 0;
//                         for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
//                         	if(3 == currentAiMesh->mFaces[k].mNumIndices)
//                         		++numTriangles;
// 
//                         if(0 != numTriangles)
//                         {
//                         	triangles.resize(numTriangles);
// 
//                         	unsigned int triangleOffset = 0;
//                         	for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
//                         	{
//                         		if(3 != currentAiMesh->mFaces[k].mNumIndices)
//                         			continue;
// 
//                         		memcpy(&triangles[0] + triangleOffset, currentAiMesh->mFaces[k].mIndices, sizeof(Triangle));
//                         		++triangleOffset;
//                         	}
//                         	currentOglModel->setTriangles(&triangles);
//                         }
// 
//                         // filling up quad array
//                         ResizableExtVector<OglModel::Quad> quads;
//                         unsigned int numQuads = 0;
//                         for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
//                         	if(4 == currentAiMesh->mFaces[k].mNumIndices)
//                         		++numQuads;
// 
//                         if(0 != numQuads)
//                         {
//                         	quads.resize(numQuads);
// 
//                         	unsigned int quadOffset = 0;
//                         	for(int k = 0; k < currentAiMesh->mNumFaces; ++k)
//                         	{
//                         		if(4 != currentAiMesh->mFaces[k].mNumIndices)
//                         			continue;
// 
//                         		memcpy(&quads[0] + quadOffset, currentAiMesh->mFaces[k].mIndices, sizeof(Quad));
//                         		++quadOffset;
//                         	}
// 
//                         	currentOglModel->setQuads(&quads);
//                         }
                    }

                    IdentityMapping<Vec3Types, ExtVec3fTypes>::SPtr currentIdentityMapping = sofa::core::objectmodel::New<IdentityMapping<Vec3Types, ExtVec3fTypes> >();
					{
						// adding the generated IdentityMapping to its parent Node
						currentSubNode->addObject(currentIdentityMapping);

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
                // generating sub Node and filling up its properties
                // store it in the stack to process its children later
                NodeInfo subNodeInfo(currentAiNode->mChildren[childIndex], getSimulation()->createNewNode(""), &currentNodeInfo);
                nodes.push(subNodeInfo);

                // adding the generated node to its parent Node
                currentNode->addChild(subNodeInfo.mNode);

                // this child will be processed, go to the next one
                ++childIndex;
            }
        }
    }

    removeEmptyNodes();

    return true;
}

bool SceneColladaLoader::fillSkeletalInfo(const aiScene* scene, aiNode* meshParentNode, aiNode* meshNode, aiMatrix4x4 meshTransformation, aiMesh* mesh, helper::vector<projectiveconstraintset::SkeletonJoint<Rigid3Types> > &skeletonJoints, helper::vector<SkeletonBone>& skeletonBones) const
{
    //std::cout << "fillSkeletalInfo : begin" << std::endl;

    // return now if their is no scene, no mesh or no skeletonBones
    if(!scene || !mesh || !mesh->HasBones())
    {
        sout << "no mesh to load !" << sendl;
        return false;
    }

    std::map<aiNode*, std::size_t> aiNodeToSkeletonJointIndex;

    // compute the mesh transformation into a rigid
    Mat4x4d meshWorldTranformation(meshTransformation[0]);
//	for(int j = 0; j < 4; ++j)
//		for(int i = 0; i < 4; ++i)
//			meshWorldTranformation[j][i] = meshTransformation[j][i];

    Rigid3Types::Coord meshTransformationRigid;
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
            std::map<aiNode*, std::size_t>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            if(aiNodeToSkeletonJointIndex.end() == aiNodeToSkeletonJointIndexIterator)
            {
                skeletonJoints.push_back(SkeletonJoint<Rigid3Types>());
                aiNodeToSkeletonJointIndex[node] = skeletonJoints.size() - 1;
                aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            }
            else
            {
                return false;
            }
            SkeletonJoint<Rigid3Types>& skeletonJoint = skeletonJoints[aiNodeToSkeletonJointIndexIterator->second];

            aiVectorKey positionKey, scaleKey;
            aiQuatKey	rotationKey;

            unsigned int numKey = std::max(channel->mNumPositionKeys, channel->mNumRotationKeys);
            //int numKey = std::max(channel->mNumScalingKeys , std::max(channel->mNumPositionKeys, channel->mNumRotationKeys));

            skeletonJoint.mTimes.resize(numKey);
            skeletonJoint.mChannels.resize(numKey);
            for(unsigned int l = 0; l < numKey; ++l)
            {
                SReal time = 0.0;
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

                Rigid3Types::Coord localRigid;
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
        std::map<aiNode*, std::size_t>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
        if(aiNodeToSkeletonJointIndex.end() == aiNodeToSkeletonJointIndexIterator)
        {
            skeletonJoints.push_back(SkeletonJoint<Rigid3Types>());
            aiNodeToSkeletonJointIndex[node] = skeletonJoints.size() - 1;
            aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
        }

        skeletonBones[i] = aiNodeToSkeletonJointIndexIterator->second;
    }

    // register every SkeletonJoint and their parents and fill up theirs properties
    for(std::size_t i = 0; i < skeletonJoints.size(); ++i)
    {
//        SkeletonJoint<Rigid3Types>& skeletonJoint = skeletonJoints[i];

        aiNode*	node = NULL;

        // find the ai node corresponding to the SkeletonJoint
        for(std::map<aiNode*, std::size_t>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.begin(); aiNodeToSkeletonJointIndexIterator != aiNodeToSkeletonJointIndex.end(); ++aiNodeToSkeletonJointIndexIterator)
        {
            if(i == aiNodeToSkeletonJointIndexIterator->second)
            {
                node = aiNodeToSkeletonJointIndexIterator->first;
                break;
            }
        }

        if(NULL == node)
            return false;

        std::size_t previousSkeletonJointIndex;
        bool firstIteration = true;

        // find parents node
        while(NULL != node)
        {
            // stop if we reach the mesh node or its parent
            if(meshNode == node || meshParentNode == node)
                break;

            // create the corresponding SkeletonJoint if it does not exist
            std::map<aiNode*, std::size_t>::iterator aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            if(aiNodeToSkeletonJointIndex.end() == aiNodeToSkeletonJointIndexIterator)
            {
                skeletonJoints.push_back(SkeletonJoint<Rigid3Types>());
                aiNodeToSkeletonJointIndex[node] = skeletonJoints.size() - 1;
                aiNodeToSkeletonJointIndexIterator = aiNodeToSkeletonJointIndex.find(node);
            }
            SkeletonJoint<Rigid3Types>& currentSkeletonJoint = skeletonJoints[aiNodeToSkeletonJointIndexIterator->second];

            // register the current node
            aiMatrix4x4 aiLocalTransformation = node->mTransformation;

            // compute the rigid corresponding to the SkeletonJoint
            Mat4x4d localTranformation(aiLocalTransformation[0]);

            Rigid3Types::Coord localRigid;
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
                for(std::size_t i = 0; i < currentSkeletonJoint.mChannels.size(); ++i)
                    currentSkeletonJoint.mChannels[i] = meshTransformationRigid.mult(currentSkeletonJoint.mChannels[i]);
            }

            currentSkeletonJoint.setRestPosition(localRigid);

            if(!firstIteration)
                skeletonJoints[previousSkeletonJointIndex].mParentIndex = aiNodeToSkeletonJointIndexIterator->second;

            firstIteration = false;
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
        std::stack<std::pair<Node::SPtr, std::size_t> > nodes;

        nodes.push(std::pair<Node::SPtr, std::size_t>(subSceneRoot, 0));
        while(!nodes.empty())
        {
            Node::SPtr& node = nodes.top().first;
            std::size_t& index = nodes.top().second;

            if(node->getChildren().size() <= index)
            {
                nodes.pop();

                if(nodes.empty())
                    break;

                Node::SPtr& parentNode = nodes.top().first;
                std::size_t& parentIndex = nodes.top().second;

                // remove the node if it has no objects
                if(node->object.empty())
                {
                    if(0 != node->getChildren().size())
                    {
                        // links its child nodes directly to its parent node before remove the current intermediary node
                        while(!node->getChildren().empty())
                        {
                            Node::SPtr childNode = static_cast<Node*>(node->getChildren()[0]);
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
                Node::SPtr child = static_cast<Node*>(node->getChildren()[index]);
                nodes.push(std::pair<Node::SPtr, std::size_t>(child, 0));
            }
        }
    }
}

} // namespace loader

} // namespace component

} // namespace sofa

