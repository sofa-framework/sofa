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
#ifndef SOFA_BULLET_COLLISION_DETECTION
#define SOFA_BULLET_COLLISION_DETECTION

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionElement.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/defaulttype/Vec.h>

#include "BulletCollisionModel.h"
#include "BulletTriangleModel.h"
#include <boost/unordered_map.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "BtInterManager.h"
#include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#pragma GCC diagnostic pop


#include <iostream>
//#include <sofa/helper/system/gl.h>
//#include <sofa/helper/system/glut.h>

#include "BulletSphereModel.h"
#include "BulletOBBModel.h"
#include "BulletCapsuleModel.h"
#include "BulletCylinderModel.h"
#include "BulletConvexHullModel.h"

namespace sofa
{

namespace component
{

namespace collision
{

//template <class T1,class T2>
//class BtInterManager;

//using namespace sofa::defaulttype;

class SOFA_BULLETCOLLISIONDETECTION_API BulletIntersection : public core::collision::Intersection,public core::collision::BaseIntersector
{
public:
    SOFA_CLASS(BulletIntersection,core::collision::Intersection);

    typedef BtInterManager<BulletIntersection> MyBtInterManager;

    Data<SReal> contactDistance;

    virtual SReal getContactDistance()const{
        return contactDistance.getValue();
    }

//    virtual SReal getAlarmDistance()const{
//        return contactDistance.getValue();
//    }

    virtual void setContactDistance(SReal dist){
        contactDistance.setValue(dist);
    }

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    /// @param swapModel output value set to true if the collision models must be swapped before calling the intersector.
    const MyBtInterManager* findIntersectionManager(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels){
        const MyBtInterManager* btim = intersectors.go(*object1,*object2);
        if(btim != 0x0){
            swapModels = false;
            return btim;
        }
        else{
            swapModels = true;
            return intersectors.go(*object2,*object1);
        }
    }    

    BtInterDispatcher<sofa::core::CollisionModel, const MyBtInterManager *> intersectors;

    template <class T1,class T2>
    bool testIntersection(T1 &,T2 &){return false;}

    template <class T1,class T2,class T3>
    int computeIntersection(T1 &,T2 &,T3){return 0;}

    template <class CModel1,class CModel2>
    static const MyBtInterManager* internalGetInterManager(sofa::core::CollisionModel&,sofa::core::CollisionModel&){
        static BtMemberInterManager<CModel1,CModel2,BulletIntersection> manager;
        return &manager;
    }

    template <class CModel1,class CModel2>
    void addCollisionDetection(){
        //const BtInterManager & manager = internalGetInterManager<CModel1,CModel2>();

        intersectors.add(typeid(CModel1), typeid(CModel2), &(internalGetInterManager<CModel1,CModel2>));
    }

private:
    virtual sofa::core::collision::ElementIntersector* findIntersector(core::CollisionModel* , core::CollisionModel* , bool& ){
        std::cerr<<"BulletIntersector::findIntersector should not be used"<<std::endl;
        return 0x0;
    }

protected:
    BulletIntersection() : contactDistance(initData(&contactDistance, (SReal)(0.2), "contactDistance", "Maximum distance between points when contact is created")){
        //intersectors.add<BulletTriangleModel,BulletTriangleModel>(this);
        addCollisionDetection<BulletTriangleModel,BulletTriangleModel>();
        addCollisionDetection<BulletSphereModel,BulletSphereModel>();
        addCollisionDetection<BulletOBBModel,BulletOBBModel>();
        addCollisionDetection<BulletTriangleModel,BulletOBBModel>();
        addCollisionDetection<BulletTriangleModel,BulletSphereModel>();
        addCollisionDetection<BulletSphereModel,BulletOBBModel>();
        addCollisionDetection<BulletCapsuleModel,BulletCapsuleModel>();
        addCollisionDetection<BulletCapsuleModel,BulletOBBModel>();
        addCollisionDetection<BulletCapsuleModel,BulletTriangleModel>();
        addCollisionDetection<BulletCapsuleModel,BulletSphereModel>();

        addCollisionDetection<BulletCapsuleModel,BulletRigidCapsuleModel>();
        addCollisionDetection<BulletRigidCapsuleModel,BulletRigidCapsuleModel>();
        addCollisionDetection<BulletRigidCapsuleModel,BulletOBBModel>();
        addCollisionDetection<BulletRigidCapsuleModel,BulletTriangleModel>();
        addCollisionDetection<BulletRigidCapsuleModel,BulletSphereModel>();

        //if you want to take account of one of any intersection below, uncoment it,
        //I wrote it because order is important for the collision response
        addCollisionDetection<BulletCylinderModel,BulletCylinderModel>();
        addCollisionDetection<BulletCapsuleModel,BulletCylinderModel>();
        addCollisionDetection<BulletCylinderModel,BulletRigidCapsuleModel>();
        addCollisionDetection<BulletCylinderModel,BulletOBBModel>();
        addCollisionDetection<BulletCylinderModel,BulletTriangleModel>();
        addCollisionDetection<BulletCylinderModel,BulletSphereModel>();

        addCollisionDetection<BulletConvexHullModel,BulletConvexHullModel>();
        addCollisionDetection<BulletConvexHullModel,BulletCylinderModel>();
        addCollisionDetection<BulletConvexHullModel,BulletCapsuleModel>();
        addCollisionDetection<BulletConvexHullModel,BulletRigidCapsuleModel>();
        addCollisionDetection<BulletConvexHullModel,BulletOBBModel>();
        addCollisionDetection<BulletConvexHullModel,BulletTriangleModel>();
        addCollisionDetection<BulletConvexHullModel,BulletSphereModel>();
    }
};

class SOFA_BULLETCOLLISIONDETECTION_API BulletCollisionDetection :
    public core::collision::BroadPhaseDetection,
    public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS2(BulletCollisionDetection, core::collision::BroadPhaseDetection, core::collision::NarrowPhaseDetection);

    Data<bool> useSimpleBroadPhase;
    Data<bool> useSAP;
    //Data<bool> useMultiSAP;
    //Data<bool> useBdvt;

    Data< helper::fixed_array<defaulttype::Vector3,2> > box;

private:
    //sofa::set< > collisionModels;

protected:
    BulletCollisionDetection();

    ~BulletCollisionDetection();


    btBroadphaseInterface * _bt_broadphase;
    btDiscreteDynamicsWorld * _bt_world;
    btDefaultCollisionConfiguration * _bt_collision_configuration;
    btCollisionDispatcher * _bt_dispatcher;

    boost::unordered_map<const btCollisionObject*,sofa::core::CollisionModel*> _bt2sofa_cm;

    BulletIntersection * _bt_inter_method;
public:
    void init();

    void addCollisionModel (core::CollisionModel *cm);
    void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>&){}

//    virtual void beginBroadPhase()
//    {
//        core::collision::BroadPhaseDetection::beginBroadPhase();
//        //collisionModels.clear();
//    }

    virtual void beginNarrowPhase();

    /* for debugging */
    void draw(const core::visual::VisualParams* vparams);

    inline virtual bool needsDeepBoundingTree()const{return false;}
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
