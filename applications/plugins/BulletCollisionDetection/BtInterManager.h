#ifndef BT_INTER_MANAGER_H
#define BT_INTER_MANAGER_H

#include <sofa/core/collision/Intersection.inl>
#include "btBulletCollisionCommon.h"
#include <sofa/helper/FnDispatcher.inl>
#include "BulletOBBModel.h"
#include "BulletConvexHullModel.h"
#include "BulletCapsuleModel.h"
#include <sofa/defaulttype/Vec.h>
#include <SofaBaseCollision/IntrUtility3.h>

namespace sofa{namespace component{namespace collision{

template <class I>
class BtInterManager{
public:

//    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2,  sofa::core::collision::DetectionOutputVector* contacts,int i = 0)
//    {
//        ++i;
//        Elem1 e1(elem1);
//        Elem2 e2(elem2);
//        return impl->computeIntersection(e1, e2, impl->getOutputVector(e1.getCollisionModel(), e2.getCollisionModel(), contacts));
//    }

    virtual void fillContacts(core::CollisionModel * cm1_base, core::CollisionModel * cm2_base,const btPersistentManifold & contPts,I & intersectionMethod,
                              sofa::core::collision::DetectionOutputVector *& contacts,bool swapContPts = false)const = 0;
};


template <class T1,class T2>
static int getId(T1 & t1,T2 & t2){
    return t1.getCollisionModel()->getSize() > t2.getCollisionModel()->getSize() ? t1.getIndex() : t2.getIndex();
}

template <class T1>
static int getId(T1 & t1,T1 & t2){
    return t1.getIndex();
}

template <class Vec>
void display(const Vec & vec){
    std::cout<<vec[0]<<" "<<vec[1]<<" "<<vec[2]<<std::endl;
}

template <class CModel>
void correctContactPoint0(SReal margin,sofa::core::collision::DetectionOutput & dec_out,typename CModel::Element &){
    dec_out.point[0] -= dec_out.normal * margin;
}

template <class CModel>
void correctContactPoint1(SReal margin,sofa::core::collision::DetectionOutput & dec_out,typename CModel::Element &){
    dec_out.point[1] += dec_out.normal * margin;
}

template <>
void correctContactPoint0<BulletOBBModel>(SReal,sofa::core::collision::DetectionOutput & dec_out,OBB & e0){
    OBB obb(e0);
    IntrUtil<OBB>::project(dec_out.point[0],obb);
}

template <>
void correctContactPoint1<BulletOBBModel>(SReal,sofa::core::collision::DetectionOutput & dec_out,OBB & e1){
    OBB obb(e1);
    IntrUtil<OBB>::project(dec_out.point[1],obb);
}


template <class CModel1,class CModel2,class I>//CModel1 and CModel2 are CollisionModels inheriting from BulletCollisionModel
class BtMemberInterManager : public BtInterManager<I>{
public:
    virtual void fillContacts(core::CollisionModel * cm1_base, core::CollisionModel * cm2_base,const btPersistentManifold & contPts,I & intersectionMethod,
                              sofa::core::collision::DetectionOutputVector *& contacts,bool swapContPts = false)const{
        static int _id = 0;

        CModel1 * cm1 = static_cast<CModel1*>(cm1_base);
        CModel2 * cm2 = static_cast<CModel2*>(cm2_base);

        if (contacts == NULL)
        {
            contacts = intersectionMethod.createOutputVector(cm1,cm2);
        }

        sofa::core::collision::TDetectionOutputVector<CModel1,CModel2>& cast_contacts = *(static_cast<sofa::core::collision::TDetectionOutputVector<CModel1,CModel2>*>(contacts));

        intersectionMethod.beginIntersection(cm1, cm2, &cast_contacts);

        int i = cast_contacts.size();
        cast_contacts.resize(i + contPts.getNumContacts());

        //std::cout<<"contPts.getNumContacts() "<<contPts.getNumContacts()<<std::endl;

        if(swapContPts){
            for(int j = 0 ; j < contPts.getNumContacts() ; ++j){
                const btManifoldPoint& manpt = contPts.getContactPoint(j);
                typename CModel1::Element e1(cm1,manpt.m_index1);
                typename CModel2::Element e2(cm2,manpt.m_index0);
//                typename CModel1::Element e1(cm1,manpt);
//                typename CModel2::Element e2(cm2,manpt.m_index0);

                sofa::core::collision::DetectionOutput & dec_out = cast_contacts[i];
                //dec_out.id = getId(e1,e2);
                dec_out.id = _id;++_id;
                dec_out.elem = std::pair<typename CModel1::Element,typename CModel2::Element>(e1,e2);
                //manpt.M
//                std::cout<<"world 1 "<<std::endl;
//                display(manpt.m_positionWorldOnA);
//                std::cout<<"world 2 "<<std::endl;
//                display(manpt.m_positionWorldOnB);
//                std::cout<<"point 1 "<<std::endl;
//                display(manpt.m_localPointA);
//                std::cout<<"point 2 "<<std::endl;
//                display(manpt.m_localPointB);
//                std::cout<<"the normal ";display(manpt.m_normalWorldOnB);
//                std::cout<<"distance "<<manpt.m_distance1<<std::endl;

                dec_out.point[1].set(manpt.m_positionWorldOnA[0],manpt.m_positionWorldOnA[1],manpt.m_positionWorldOnA[2]);
                dec_out.value = (manpt.m_distance1) - intersectionMethod.getContactDistance();
                dec_out.normal.set(manpt.m_normalWorldOnB[0],manpt.m_normalWorldOnB[1],manpt.m_normalWorldOnB[2]);

                //dec_out.point[0] = dec_out.point[1] - dec_out.value * dec_out.normal;
                dec_out.point[0].set(manpt.m_positionWorldOnB[0],manpt.m_positionWorldOnB[1],manpt.m_positionWorldOnB[2]);

//                correctContactPoint0<CModel1>(cm1->getBtCollisionObject()->getCollisionShape()->getMargin(),dec_out);
//                correctContactPoint1<CModel2>(cm2->getBtCollisionObject()->getCollisionShape()->getMargin(),dec_out);
                correctContactPoint0<CModel1>(cm1->margin.getValue(),dec_out,e1);
                correctContactPoint1<CModel2>(cm2->margin.getValue(),dec_out,e2);

                dec_out.value = fabs(manpt.m_distance1) - intersectionMethod.getContactDistance();
                //dec_out.value = std::max<SReal>(fabs(manpt.m_distance1) - intersectionMethod.getContactDistance(),(SReal)0.0);
                //dec_out.value = (dec_out.point[0] - dec_out.point[1]).norm() - intersectionMethod.getContactDistance();

//                std::cout<<cm1->name.getValue()<<" infos"<<std::endl;
//                std::cout<<"point 0"<<std::endl;
//                display(dec_out.point[0]);
//                std::cout<<cm2->name.getValue()<<" infos"<<std::endl;
//                std::cout<<"point 1"<<std::endl;
//                display(dec_out.point[1]);
//                std::cout<<"norm"<<std::endl;
//                display(dec_out.normal);
//                std::cout<<"value "<<dec_out.value<<std::endl;
                //this is the end
//                //this is the end

                ++i;
            }
        }
        else{
            for(int j = 0 ; j < contPts.getNumContacts() ; ++j){
                const btManifoldPoint& manpt = contPts.getContactPoint(j);
                typename CModel1::Element e1(cm1,manpt.m_index0);
                typename CModel2::Element e2(cm2,manpt.m_index1);

                sofa::core::collision::DetectionOutput & dec_out = cast_contacts[i];
                //dec_out.id = getId(e1,e2);
                dec_out.id = _id;++_id;
                dec_out.elem = std::pair<typename CModel1::Element,typename CModel2::Element>(e1,e2);
    //            std::cout<<"world 1 "<<std::endl;
    //            display(manpt.m_positionWorldOnA);
    //            std::cout<<"world 2 "<<std::endl;
    //            display(manpt.m_positionWorldOnB);
    //            std::cout<<"point 1 "<<std::endl;
    //            display(manpt.m_localPointA);
    //            std::cout<<"point 2 "<<std::endl;
    //            display(manpt.m_localPointB);
    //            std::cout<<"the normal ";display(manpt.m_normalWorldOnB);
    //            std::cout<<"distance "<<manpt.m_distance1<<std::endl;


                dec_out.point[1].set(manpt.m_positionWorldOnB[0],manpt.m_positionWorldOnB[1],manpt.m_positionWorldOnB[2]);
                //dec_out.value = fabs(manpt.m_distance1) - intersectionMethod.getContactDistance();
                //dec_out.value = fabs(manpt.m_distance1) - intersectionMethod.getContactDistance();
                //dec_out.value = 1;
                //std::cout<<"dist "<<manpt.m_distance1<<std::endl;
                dec_out.normal.set(-manpt.m_normalWorldOnB[0],-manpt.m_normalWorldOnB[1],-manpt.m_normalWorldOnB[2]);

                //dec_out.point[0] = dec_out.point[1] - dec_out.value * dec_out.normal;
                //dec_out.point[0].set(manpt.m_localPointA[0],manpt.m_localPointA[1],manpt.m_localPointA[2]);
                dec_out.point[0].set(manpt.m_positionWorldOnA[0],manpt.m_positionWorldOnA[1],manpt.m_positionWorldOnA[2]);

                correctContactPoint0<CModel1>(cm1->margin.getValue(),dec_out,e1);
                correctContactPoint1<CModel2>(cm2->margin.getValue(),dec_out,e2);

                dec_out.value = fabs(manpt.m_distance1) - intersectionMethod.getContactDistance();
                //dec_out.value = std::max<SReal>(fabs(manpt.m_distance1) - intersectionMethod.getContactDistance(),(SReal)0.0);
                //dec_out.value = (dec_out.point[0] - dec_out.point[1]).norm() - intersectionMethod.getContactDistance();

//                std::cout<<cm1->name.getValue()<<" infos"<<std::endl;
//                std::cout<<"point 0"<<std::endl;
//                display(dec_out.point[0]);
//                std::cout<<cm2->name.getValue()<<" infos"<<std::endl;
//                std::cout<<"point 1"<<std::endl;
//                display(dec_out.point[1]);
//                std::cout<<"norm"<<std::endl;
//                display(dec_out.normal);
//                std::cout<<"value "<<dec_out.value<<std::endl;
//                //this is the end
                ++i;
            }
        }
    }
};


template <class BaseClass, class Res>
class BtInterDispatcher : public helper::BasicDispatcher<BaseClass,Res>{
protected:
    friend class BulletInternalIntersector;
    virtual Res defaultFn(BaseClass&, BaseClass&){
        return 0x0;
    }

    static Res ignoreFn(BaseClass&, BaseClass&){
        return 0x0;
    }
};

}
}
}


#endif
