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
#ifndef SOFA_COMPONENT_LOADER_SceneColladaLoader_H
#define SOFA_COMPONENT_LOADER_SceneColladaLoader_H

#include <ColladaSceneLoader/config.h>
#include <sofa/core/loader/SceneLoader.h>
#include <sofa/helper/SVector.h>
#include <sofa/simulation/Node.h>
#include <SofaBoundaryCondition/SkeletalMotionConstraint.h>

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

namespace sofa
{

namespace component
{

namespace loader
{

/**
	current limitation : one animation per scene
*/
class SOFA_COLLADASCENELOADER_API SceneColladaLoader : public sofa::core::loader::SceneLoader
{
public:
    SOFA_CLASS(SceneColladaLoader,sofa::core::loader::SceneLoader);

    struct NodeInfo;

    // describing a link between Assimp Node and Sofa Node allowing us to build a node hierarchy
    struct NodeInfo
    {
        std::size_t			mChildIndex;		// index of the current child node to process
        aiNode*				mAiNode;		// aiNode being processed
        simulation::Node::SPtr			mNode;			// corresponding Node created in the sofa scene graph
        NodeInfo*			mParentNode;		// parent node (useful to retrieve mesh skeleton and to compute world transformation matrix)
        aiMatrix4x4			mTransformation;	// matrix that transforms from node space to world space

        NodeInfo(aiNode* pAiNode, simulation::Node::SPtr pNode, NodeInfo* mParentNode = NULL) :
            mChildIndex(0),
            mAiNode(pAiNode),
            mNode(pNode),
            mParentNode(mParentNode),
            mTransformation()
        {
            if(mParentNode)
                mTransformation = mParentNode->mTransformation;

            if(pAiNode)
                mTransformation *= pAiNode->mTransformation;

            /*if(root)
            {
            	std::cout << pAiNode->mTransformation.a1 << " - " << pAiNode->mTransformation.b1 << " - " << pAiNode->mTransformation.c1 << " - " << pAiNode->mTransformation.d1 << std::endl;
            	std::cout << pAiNode->mTransformation.a2 << " - " << pAiNode->mTransformation.b2 << " - " << pAiNode->mTransformation.c2 << " - " << pAiNode->mTransformation.d2 << std::endl;
            	std::cout << pAiNode->mTransformation.a3 << " - " << pAiNode->mTransformation.b3 << " - " << pAiNode->mTransformation.c3 << " - " << pAiNode->mTransformation.d3 << std::endl;
            	std::cout << pAiNode->mTransformation.a4 << " - " << pAiNode->mTransformation.b4 << " - " << pAiNode->mTransformation.c4 << " - " << pAiNode->mTransformation.d4 << std::endl;
            }*/
        }

        NodeInfo(const NodeInfo& nodeInfo) :
            mChildIndex(nodeInfo.mChildIndex),
            mAiNode(nodeInfo.mAiNode),
            mNode(nodeInfo.mNode),
            mParentNode(nodeInfo.mParentNode),
            mTransformation(nodeInfo.mTransformation)
        {

        }
    };

    // describing a link between a Node and an Assimp Mesh
    struct MeshInfo
    {
        aiMesh*		mAiMesh;	// mesh being processed
        NodeInfo	mNodeInfo;		// its owner node

        MeshInfo(aiMesh* pAiMesh, NodeInfo pNodeInfo) :
            mAiMesh(pAiMesh),
            mNodeInfo(pNodeInfo)
        {

        }
    };

protected:
    SceneColladaLoader();
    ~SceneColladaLoader();
public:

    virtual void init();
    virtual bool load();

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        return BaseLoader::canCreate (obj, context, arg);
    }

	float getAnimationSpeed() const			{return animationSpeed.getValue();}
	void setAnimationSpeed(float speed)		{animationSpeed.setValue(speed);}

protected:

    bool readDAE (std::ifstream &file, const char* filename);

private:

    // build the joints and bones array used in the SkeletalMotionConstraint
    bool fillSkeletalInfo(const aiScene* scene, aiNode* meshParentNode, aiNode* meshNode, aiMatrix4x4 meshTransformation, aiMesh* mesh, helper::vector<projectiveconstraintset::SkeletonJoint<defaulttype::Rigid3Types> >& skeletonJoints, helper::vector<projectiveconstraintset::SkeletonBone>& skeletonBones) const;

    // clean the scene graph of its empty and useless intermediary nodes
    void removeEmptyNodes();

public:

    virtual std::string type() { return "The format of this scene is Collada (.dae)."; }

private:
    simulation::Node::SPtr subSceneRoot;		// the Node containing the whole Collada loaded scene

    Assimp::Importer importer;		// the Assimp importer used to easily load the Collada scene

	Data<float> animationSpeed;
	Data<bool> generateCollisionModels;

#ifdef SOFA_HAVE_PLUGIN_FLEXIBLE
	Data<bool> useFlexible;
#endif
#ifdef SOFA_HAVE_PLUGIN_IMAGE
    Data<bool> generateShapeFunction;
    Data<SReal> voxelSize;
#endif

};

} // namespace loader

} // namespace component

} // namespace sofa

#endif
