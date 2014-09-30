#ifndef BT_DYNAMIC_MESH_H
#define BT_DYNAMIC_MESH_H
#include "BulletCollision/Gimpact/btGImpactShape.h"

class BtDynamicMeshPart : public btGImpactMeshShapePart{
public:
    BtDynamicMeshPart()  : btGImpactMeshShapePart() {}

    BtDynamicMeshPart(btStridingMeshInterface * meshInterface,	int part) : btGImpactMeshShapePart(meshInterface,part){}

    virtual void calcLocalAABB()
    {
        lockChildShapes();
//        if(m_box_set.getNodeCount() == 0)
//        {
            m_box_set.buildSet();
//        }
//        else
//        {
//            m_box_set.update();
//        }
        unlockChildShapes();

        m_localAABB = m_box_set.getGlobalBox();
    }
};

class BtDynamicMesh : public btGImpactMeshShape{
public:
    //BtDynamicMesh(btStridingMeshInterface * meshInterface) : btGImpactMeshShape(meshInterface){}

    virtual void calcLocalAABB()
    {
        m_localAABB.invalidate();
        int i = m_mesh_parts.size();
        while(i--)
        {

            m_mesh_parts[i]->updateBound();

            m_localAABB.merge(m_mesh_parts[i]->getLocalBox());
        }
    }

    void rebuildMeshParts(btStridingMeshInterface * meshInterface)
    {
        for (int i=0;i<meshInterface->getNumSubParts() ;++i )
        {
            delete m_mesh_parts[i];
            m_mesh_parts[i] = new BtDynamicMeshPart(meshInterface,i);
        }
    }

    BtDynamicMesh(btStridingMeshInterface * meshInterface) : btGImpactMeshShape(meshInterface){
        rebuildMeshParts(meshInterface);
    }
//    BtDynamicMesh(btStridingMeshInterface * meshInterface) : btGImpactMeshShape()
//    {
//        this->m_meshInterface = meshInterface;
//        buildMeshParts(meshInterface);
//    }
};

#endif

