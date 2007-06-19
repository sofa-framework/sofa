#include <iostream>
#include <fstream>
#include <stack>

#include <flowvr/module.h>
#include <flowvr/render/chunkwriter.h>
//#include <flowvr/interact/chunkwriter.h>

#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/tree/Action.h>
#include <sofa/simulation/tree/ParallelActionScheduler.h>
#include <sofa/simulation/tree/CactusStackStorage.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/tree/AnimateBeginEvent.h>
#include <sofa/simulation/tree/AnimateEndEvent.h>
#include <sofa/component/topology/MeshTopology.h>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/InitAction.h>
#include <sofa/simulation/tree/DeleteAction.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/MinProximityIntersection.h>
#include <sofa/component/collision/BruteForceDetection.h>

#include <sofa/gui/SofaGUI.h>

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;
using namespace sofa::simulation::tree;


//#define VERBOSE

// ---------------------------------------------------------------------
// --- SOFA+FLOWVR integration
// ---------------------------------------------------------------------

namespace ftl
{

namespace Type
{

template<int N, typename real>
Type get(const sofa::defaulttype::Vec<N,real>&)
{
    return (Type)vector(get(real()),N);
}

}

}

namespace SofaFlowVR
{

flowvr::ModuleAPI* module = NULL;

//using namespace flowvr::render;
//using namespace flowvr::interact;

//using namespace Sofa::Abstract;
//using namespace Sofa::Components;
using namespace sofa::defaulttype;

class FlowVREvent : public sofa::core::objectmodel::Event
{
public:
    virtual ~FlowVREvent() {}
};

class FlowVRPreInitEvent : public FlowVREvent
{
public:
    virtual ~FlowVRPreInitEvent() {}
    std::vector<flowvr::Port*>* ports;
};

class FlowVRInitEvent : public FlowVREvent
{
public:
    virtual ~FlowVRInitEvent() {}
    flowvr::ModuleAPI* module;
};

class FlowVRBeginIterationEvent : public FlowVREvent
{
public:
    virtual ~FlowVRBeginIterationEvent() {}
    flowvr::ModuleAPI* module;
};

class FlowVREndIterationEvent : public FlowVREvent
{
public:
    virtual ~FlowVREndIterationEvent() {}
    flowvr::ModuleAPI* module;
};

class FlowVRObject : public virtual sofa::core::objectmodel::BaseObject
{
public:
    FlowVRObject()
    {
        f_listening.setValue(true);
    }

    virtual ~FlowVRObject()
    {
    }

    virtual void animateBegin(double /*dt*/)
    {
    }

    virtual void animateEnd(double /*dt*/)
    {
    }

    virtual void flowvrPreInit(std::vector<flowvr::Port*>* /*ports*/)
    {
    }

    virtual void flowvrInit(flowvr::ModuleAPI* /*module*/)
    {
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* /*module*/)
    {
    }

    virtual void flowvrEndIteration(flowvr::ModuleAPI* /*module*/)
    {
    }

    virtual void handleEvent(sofa::core::objectmodel::Event* event)
    {
        if (AnimateBeginEvent* ev = dynamic_cast<AnimateBeginEvent*>(event))
            animateBegin(ev->getDt());
        if (AnimateEndEvent* ev = dynamic_cast<AnimateEndEvent*>(event))
            animateEnd(ev->getDt());
        if (dynamic_cast<FlowVREvent*>(event))
        {
            if (FlowVRPreInitEvent* ev = dynamic_cast<FlowVRPreInitEvent*>(event))
                flowvrPreInit(ev->ports);
            if (FlowVRInitEvent* ev = dynamic_cast<FlowVRInitEvent*>(event))
                flowvrInit(ev->module);
            if (FlowVRBeginIterationEvent* ev = dynamic_cast<FlowVRBeginIterationEvent*>(event))
                flowvrBeginIteration(ev->module);
            if (FlowVREndIterationEvent* ev = dynamic_cast<FlowVREndIterationEvent*>(event))
                flowvrEndIteration(ev->module);
        }
    }

};

std::map<std::string, flowvr::ModuleAPI*> moduleMap;
std::map<std::string, flowvr::InputPort*> inputPortMap;
std::map<std::string, flowvr::OutputPort*> outputPortMap;
std::map<std::string, flowvr::render::SceneOutputPort*> sceneOutputPortMap;
std::vector<flowvr::ID> prevP;
std::vector<flowvr::ID> prevVS;
std::vector<flowvr::ID> prevPS;
std::vector<flowvr::ID> prevVB;
std::vector<flowvr::ID> prevIB;
std::vector<flowvr::ID> prevT;

flowvr::ModuleAPI* createModule(const std::vector<flowvr::Port*>& ports, const char* name="")
{
    flowvr::ModuleAPI*& module = moduleMap[name];
    if (module==NULL)
    {
        std::cout << "SofaFlowVR: Creating module "<<name<<std::endl;
        module = flowvr::initModule(ports);
    }
    else
    {
        std::cout << "SofaFlowVR: Reusing module "<<name<<std::endl;
    }
    return module;
}

flowvr::InputPort* createInputPort(const char* name)
{
    flowvr::InputPort*& port = inputPortMap[name];
    if (port==NULL)
    {
        std::cout << "SofaFlowVR: Creating port "<<name<<std::endl;
        port = new flowvr::InputPort(name);
    }
    else
        std::cout << "SofaFlowVR: Reusing port "<<name<<std::endl;
    return port;
}

flowvr::OutputPort* createOutputPort(const char* name)
{
    flowvr::OutputPort*& port = outputPortMap[name];
    if (port==NULL)
    {
        std::cout << "SofaFlowVR: Creating port "<<name<<std::endl;
        port = new flowvr::OutputPort(name);
    }
    else
        std::cout << "SofaFlowVR: Reusing port "<<name<<std::endl;
    return port;
}

flowvr::render::SceneOutputPort* createSceneOutputPort(const char* name="scene")
{
    flowvr::render::SceneOutputPort*& port = sceneOutputPortMap[name];
    if (port==NULL)
    {
        std::cout << "SofaFlowVR: Creating port "<<name<<std::endl;
        port = new flowvr::render::SceneOutputPort(name);
    }
    else
        std::cout << "SofaFlowVR: Reusing port "<<name<<std::endl;

    return port;
}

class FlowVRModule : public FlowVRObject
{
public:
    flowvr::ModuleAPI* module;
    DataField<double> f_dt;
    int it;
    double lasttime;
    bool step;
    FlowVRModule()
        : module(NULL)
        , f_dt(dataField(&f_dt,0.0,"dt","simulation time interval between flowvr iteration"))
        , it(-1)
        , lasttime(0.0), step(false)
    {
    }

    virtual void init()
    {
        if (module!=NULL) return;
        std::vector<flowvr::Port*> ports;
        std::cout << "Sending FlowVRPreInit"<<std::endl;
        FlowVRPreInitEvent ev;
        ev.ports = &ports;
        getContext()->propagateEvent(&ev);
        //module = flowvr::initModule(ports);
        module = createModule(ports);
        if (module == NULL)
        {
            std::cerr << "SofaFlowVR: module creation failed. Exit."<<std::endl;
            std::exit(1);
        }
        std::cout << "Sending FlowVRInit"<<std::endl;
        FlowVRInitEvent ev2;
        ev2.module = module;
        getContext()->propagateEvent(&ev2);
    }

    virtual void animateBegin(double /*dt*/)
    {
        if (module==NULL) return;
        if (it!=-1 && f_dt.getValue()>0 && getContext()->getTime()<lasttime+f_dt.getValue()) return;
        if (!module->wait())
        {
            std::cerr << "SofaFlowVR: module wait method returned 0. Exit."<<std::endl;
            std::exit(1);
        }
        ++it; step = true;
        lasttime = getContext()->getTime();
        std::cout << "Sending FlowVRBeginIteration"<<std::endl;
        FlowVRBeginIterationEvent ev;
        ev.module = module;
        getContext()->propagateEvent(&ev);
    }

    virtual void animateEnd(double /*dt*/)
    {
        if (module==NULL) return;
        if (!step) return;
        step = false;
        std::cout << "Sending FlowVREndIteration"<<std::endl;
        FlowVREndIterationEvent ev;
        ev.module = module;
        getContext()->propagateEvent(&ev);
    }
};

SOFA_DECL_CLASS(FlowVRModule)
int FlowVRModuleClass = sofa::core::RegisterObject("FlowVR main module")
        .add<FlowVRModule>()
        ;

//flowvr::interact::ObjectsOutputPort pObjectsOut("objects");
//flowvr::interact::ChunkInteractWriter objects;

class FlowVRInputMesh : public FlowVRObject
{
public:

    flowvr::InputPort* pInFacets;
    flowvr::InputPort* pInPoints;
    flowvr::InputPort* pInMatrix;

    DataField<float> f_scale;
    DataField<Vec3f> f_trans;
    DataField<bool> computeV;
    DataField<double> maxVDist;

    // Velocity is estimated by searching the nearest primitive from each new point
    // To do it we need to create an additionnal PointModel collision model, as well as a Detection and Intersection class
    sofa::simulation::tree::GNode * newPointsNode;
    typedef sofa::simulation::tree::GNode::Sequence<sofa::core::CollisionModel>::iterator CMIterator;
    sofa::component::MechanicalObject<Vec3dTypes> * newPoints;
    sofa::component::collision::PointModel * newPointsCM;
    sofa::component::collision::MinProximityIntersection * intersection;
    sofa::component::collision::BruteForceDetection * detection;
    sofa::helper::vector<double> newPointsDist;

    Mat4x4f matrix;
    int facetsLastIt;
    int pointsLastIt;
    int matrixLastIt;
    double motionLastTime;

    FlowVRInputMesh()
        : pInFacets(createInputPort("facets")), pInPoints(createInputPort("points")), pInMatrix(createInputPort("matrix"))
        , f_scale(dataField(&f_scale,1.0f,"scale","scale"))
        , f_trans(dataField(&f_trans,Vec3f(0,0,0),"translation","translation"))
        , computeV( dataField(&computeV, false, "computeV", "estimate velocity by detecting nearest primitive of previous model") )
        , maxVDist( dataField(&maxVDist,   1.0, "maxVDist", "maximum distance to use for velocity estimation") )
        , newPointsNode(NULL), newPointsCM(NULL), intersection(NULL), detection(NULL)
        , facetsLastIt(-20), pointsLastIt(-20), matrixLastIt(-20), motionLastTime(-1000)
    {
    }

    ~FlowVRInputMesh()
    {
        if (newPointsNode != NULL)
        {
            newPointsNode->execute<sofa::simulation::tree::DeleteAction>();
            delete newPointsNode;
        }
    }

    virtual void flowvrPreInit(std::vector<flowvr::Port*>* ports)
    {
        std::cout << "Received FlowVRPreInit"<<std::endl;
        ports->push_back(pInFacets);
        ports->push_back(pInPoints);
        ports->push_back(pInMatrix);
    }

    virtual void init()
    {
        this->FlowVRObject::init();
        if (computeV.getValue())
        {
            newPointsNode = new sofa::simulation::tree::GNode("newPoints");
            newPointsNode->addObject ( newPoints = new sofa::component::MechanicalObject<Vec3dTypes> );
            newPointsNode->addObject ( newPointsCM = new sofa::component::collision::PointModel );

            newPointsNode->addObject ( intersection = new sofa::component::collision::MinProximityIntersection );
            intersection->setAlarmDistance(maxVDist.getValue());
            intersection->setContactDistance(0); //maxVDist.getValue());

            newPointsNode->addObject ( detection = new sofa::component::collision::BruteForceDetection );
            detection->setIntersectionMethod(intersection);

            newPointsNode->execute<sofa::simulation::tree::InitAction>();
        }
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* module)
    {
        std::cout << "Received FlowVRBeginIteration"<<std::endl;
        flowvr::Message points, facets;
        double time = getContext()->getTime();

        module->get(pInPoints, points);
        module->get(pInFacets, facets);

        bool newmatrix = false;
        if (pInMatrix->isConnected())
        {
            flowvr::Message msgmatrix;
            module->get(pInMatrix,msgmatrix);
            int matrixIt = -1;
            msgmatrix.stamps.read(pInMatrix->stamps->it,matrixIt);
            if (matrixIt != matrixLastIt && msgmatrix.data.getSize()>=(int)sizeof(Mat4x4f))
            {
                matrix = *msgmatrix.data.getRead<Mat4x4f>(0);
                matrixLastIt = matrixIt;
                newmatrix = true;
            }
        }

        int pointsIt = -1;
        points.stamps.read(pInPoints->stamps->it,pointsIt);
        const unsigned int nbv = points.data.getSize()/sizeof(Vec3f);
        if (pointsIt != pointsLastIt || newmatrix)
        {
            pointsLastIt = pointsIt;
            const Vec3f* vertices = points.data.getRead<Vec3f>(0);
            const Vec3f trans = f_trans.getValue();
            const float scale = f_scale.getValue();

            BaseObject* mmodel = getContext()->getMechanicalState();
            sofa::core::componentmodel::behavior::MechanicalState<Vec3fTypes>* mmodel3f;
            sofa::core::componentmodel::behavior::MechanicalState<Vec3dTypes>* mmodel3d;
            if ((mmodel3f = dynamic_cast<sofa::core::componentmodel::behavior::MechanicalState<Vec3fTypes>*>(mmodel))!=NULL)
            {
                std::cout << "Copying "<<nbv<<" vertices to mmodel3f"<<std::endl;
                mmodel3f->resize(nbv);
                Vec3fTypes::VecCoord& x = *mmodel3f->getX();
                if (matrixLastIt==-20)
                {
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        x[i] = vertices[i]*scale;
                        x[i] += trans;
                    }
                }
                else
                {
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        Vec3f v = vertices[i]*scale;
                        v += trans;
                        Vec4f tv = matrix * Vec4f(v[0],v[1],v[2],1.0f);
                        x[i] = Vec3f(tv[0],tv[1],tv[2]);
                    }
                }
            }
            else if ((mmodel3d = dynamic_cast<sofa::core::componentmodel::behavior::MechanicalState<Vec3dTypes>*>(mmodel))!=NULL)
            {
                bool doComputeV = (computeV.getValue() && newPoints != NULL && motionLastTime != -1000);

                sofa::core::componentmodel::behavior::MechanicalState<Vec3dTypes>* mm;
                if (doComputeV)
                {
                    std::cout << "Copying "<<nbv<<" vertices and estimate velocity"<<std::endl;
                    mm = newPoints; // put new data in newPoints state
                }
                else
                {
                    std::cout << "Copying "<<nbv<<" vertices to mmodel3d"<<std::endl;
                    mm = mmodel3d;
                }
                mm->resize(nbv);
                Vec3dTypes::VecCoord& x = *mm->getX();

                if (matrixLastIt==-20)
                {
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        x[i] = vertices[i]*scale;
                        x[i] += trans;
                    }
                }
                else
                {
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        Vec3f v = vertices[i]*scale;
                        v += trans;
                        Vec4f tv = matrix * Vec4f(v[0],v[1],v[2],1.0f);
                        x[i] = Vec3d(tv[0],tv[1],tv[2]);
                    }
                }
                if (doComputeV)
                {
                    sofa::simulation::tree::GNode* node = dynamic_cast<sofa::simulation::tree::GNode*>(getContext());
                    Vec3dTypes::VecDeriv& v = *newPoints->getV();
                    const double dmax = maxVDist.getValue();
                    newPointsDist.resize(nbv);
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        v[i] = Vec3dTypes::Deriv();
                        newPointsDist[i] = dmax;
                    }
                    intersection->setAlarmDistance(dmax); // make sure the distance is up-to-date
                    intersection->setContactDistance(0);
                    newPointsCM->computeBoundingTree( 6 ); // compute a bbox tree of depth 6
                    //std::cout << "computeV: "<<newPointsCM->end().getIndex()<<" points"<<std::endl;
                    detection->clearNarrowPhase();
                    for (CMIterator it = node->collisionModel.begin(), itend = node->collisionModel.end(); it != itend ; ++it)
                    {
                        sofa::core::CollisionModel* cm2 = *it;
                        std::cout << "computeV: narrow phase detection with "<<cm2->getClassName()<<std::endl;
                        detection->addCollisionPair(std::make_pair(newPointsCM->getFirst(), cm2->getFirst()));
                        //detection->addCollisionPair(std::make_pair(cm2, newPointsCM));
                    }
                    // then we start the real detection between primitives
                    {
                        std::vector<std::pair<sofa::core::CollisionElementIterator, sofa::core::CollisionElementIterator> >& vectElemPair = detection->getCollisionElementPairs();
                        std::vector<std::pair<sofa::core::CollisionElementIterator, sofa::core::CollisionElementIterator> >::iterator it4 = vectElemPair.begin();
                        std::vector<std::pair<sofa::core::CollisionElementIterator, sofa::core::CollisionElementIterator> >::iterator it4End = vectElemPair.end();

                        std::cout << "computeV: "<<vectElemPair.size()<<" colliding bbox pairs"<<std::endl;
                        // Cache the intersector used
                        sofa::core::componentmodel::collision::ElementIntersector* intersector = NULL;
                        sofa::core::CollisionModel* model1 = NULL;
                        sofa::core::CollisionModel* model2 = NULL;
                        int newPointsCMIndex = 0; // 0 or 1 depending if newPointsCM is the first or second CM
                        int ncollisions = 0;
                        for (; it4 != it4End; it4++)
                        {
                            sofa::core::CollisionElementIterator cm1 = it4->first;
                            sofa::core::CollisionElementIterator cm2 = it4->second;
                            if (cm1.getCollisionModel() != model1 || cm2.getCollisionModel() != model2)
                            {
                                model1 = cm1.getCollisionModel();
                                model2 = cm2.getCollisionModel();
                                intersector = intersection->findIntersector(model1, model2);
                                //newPointsCMIndex = (model2==newPointsCM)?1:0;
                            }
                            if (intersector != NULL)
                            {
                                sofa::core::componentmodel::collision::DetectionOutput *detection = intersector->intersect(cm1, cm2);
                                if (detection != NULL)
                                {
                                    ++ncollisions;
                                    newPointsCMIndex = (detection->elem.second.getCollisionModel()==newPointsCM)?1:0;
                                    int index = (&(detection->elem.first))[newPointsCMIndex].getIndex();
                                    double d = detection->distance;
                                    if ((unsigned)index >= nbv)
                                    {
                                        std::cerr << "computeV: invalid point index "<<index<<std::endl;
                                    }
                                    else if (d < newPointsDist[index])
                                    {
                                        newPointsDist[index] = d;
                                        v[index] = detection->point[newPointsCMIndex] - detection->point[1-newPointsCMIndex];
                                    }
                                    delete detection;
                                }
                            }
                        }
                        std::cout << "computeV: "<<ncollisions<<" collisions detected"<<std::endl;
                    }
                    // then we finalize the results
                    const double vscale = (time > motionLastTime) ? 1.0/(time-motionLastTime) : 1.0;
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        v[i] *= vscale;
                    }
                    mmodel3d->resize(nbv);
                    Vec3dTypes::VecCoord& x2 = *mmodel3d->getX();
                    Vec3dTypes::VecDeriv& v2 = *mmodel3d->getV();
                    x2.swap(x);
                    v2.swap(v);
                }
                motionLastTime = time;
            }
        }

        int facetsIt = -1;
        facets.stamps.read(pInFacets->stamps->it,facetsIt);
        if (facetsIt != facetsLastIt)
        {
            facetsLastIt = facetsIt;
            const unsigned int nbi = facets.data.getSize()/sizeof(unsigned short);
            const unsigned short* indices = facets.data.getRead<unsigned short>(0);

            // check indices
            bool valid = true;
            for (unsigned int i=0; i<nbi; i++)
            {
                if (indices[i] >= nbv)
                {
                    std::cerr << "ERROR: invalid indice "<<i<<" ("<<indices[i]<<">="<<nbv<<") it="<<facetsIt<<std::endl;
                    valid = false;
                }
            }
            BaseObject* topology = getContext()->getTopology();
            sofa::component::topology::MeshTopology* mesh;
            if ((mesh = dynamic_cast<sofa::component::topology::MeshTopology*>(topology))!=NULL)
            {
                mesh->clear();
                if (valid)
                {
                    std::cout << "Copying "<<nbi/3<<" triangles to mesh"<<std::endl;
                    for (unsigned int i=0; i<nbi; i+=3)
                    {
                        mesh->addTriangle(indices[i  ],indices[i+1],indices[i+2]);
                        // must create edges too
                        if (indices[i  ] < indices[i+1])
                            mesh->addLine(indices[i  ],indices[i+1]);
                        if (indices[i+1] < indices[i+2])
                            mesh->addLine(indices[i+1],indices[i+2]);
                        if (indices[i+2] < indices[i  ])
                            mesh->addLine(indices[i+2],indices[i  ]);
                    }
                    std::cout << "Copying "<<mesh->getNbLines()<<" edges to mesh"<<std::endl;
                }
            }
        }
    }
};

SOFA_DECL_CLASS(FlowVRInputMesh)
int FlowVRInputMeshClass = sofa::core::RegisterObject("Import a mesh from a FlowVR InputPort")
        .add< FlowVRInputMesh >()
        ;


class FlowVRRenderEvent : public FlowVREvent
{
public:
    virtual ~FlowVRRenderEvent() {}
};

class FlowVRRenderInitEvent : public FlowVRRenderEvent
{
public:
    virtual ~FlowVRRenderInitEvent() {}
    flowvr::render::ChunkRenderWriter* scene;
    bool* scratch;
};

class FlowVRRenderUpdateEvent : public FlowVRRenderEvent
{
public:
    virtual ~FlowVRRenderUpdateEvent() {}
    flowvr::render::ChunkRenderWriter* scene;
};

class FlowVRRenderObject : public FlowVRObject
{
    virtual void flowvrRenderInit(flowvr::render::ChunkRenderWriter* /*scene*/, bool* /*scratch*/)
    {
    }

    virtual void flowvrRenderUpdate(flowvr::render::ChunkRenderWriter* /*scene*/)
    {
    }

    virtual void handleEvent(sofa::core::objectmodel::Event* event)
    {
        FlowVRObject::handleEvent(event);
        if (dynamic_cast<FlowVRRenderEvent*>(event))
        {
            if (FlowVRRenderInitEvent* ev = dynamic_cast<FlowVRRenderInitEvent*>(event))
                flowvrRenderInit(ev->scene, ev->scratch);
            if (FlowVRRenderUpdateEvent* ev = dynamic_cast<FlowVRRenderUpdateEvent*>(event))
                flowvrRenderUpdate(ev->scene);
        }
    }
};

class FlowVRRenderWriter : public FlowVRRenderObject
{
public:
    flowvr::render::SceneOutputPort* pOutScene;
    flowvr::render::ChunkRenderWriter scene;
    bool scratch;
    bool init;
    FlowVRRenderWriter()
        : pOutScene(createSceneOutputPort()), init(false), scratch(false)
    {
        for(unsigned int i=0; i<prevP.size(); ++i)
            scene.delPrimitive(prevP[i]);
        prevP.clear();
        for(unsigned int i=0; i<prevVS.size(); ++i)
            scene.delVertexShader(prevVS[i]);
        prevVS.clear();
        for(unsigned int i=0; i<prevPS.size(); ++i)
            scene.delPixelShader(prevPS[i]);
        prevPS.clear();
        for(unsigned int i=0; i<prevVB.size(); ++i)
            scene.delVertexBuffer(prevVB[i]);
        prevVB.clear();
        for(unsigned int i=0; i<prevIB.size(); ++i)
            scene.delIndexBuffer(prevIB[i]);
        prevIB.clear();
    }

    virtual void flowvrPreInit(std::vector<flowvr::Port*>* ports)
    {
        std::cout << "Received FlowVRPreInit"<<std::endl;
        ports->push_back(pOutScene);
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* /*module*/)
    {
        if (!init)
        {
            FlowVRRenderInitEvent ev;
            ev.scene = &scene;
            ev.scratch = &scratch;
            getContext()->propagateEvent(&ev);
            init = true;
        }
    }

    virtual void flowvrEndIteration(flowvr::ModuleAPI* /*module*/)
    {
        FlowVRRenderUpdateEvent ev;
        ev.scene = &scene;
        getContext()->propagateEvent(&ev);
        scene.put(pOutScene, scratch);
        scratch = true;
    }

    virtual void flowvrInit(flowvr::ModuleAPI* module)
    {
        FlowVRRenderObject::flowvrInit(module);
        if (!init)
        {
            //FlowVRRenderInitEvent ev;
            //ev.scene = &scene;
            //getContext()->propagateEvent(&ev);
            scene.put(pOutScene);
            //init = true;
        }
    }

};

SOFA_DECL_CLASS(FlowVRRenderWriter)
int FlowVRRenderWriterClass = sofa::core::RegisterObject("FlowVRRender scene manager")
        .add<FlowVRRenderWriter>()
        ;

class FlowVRRenderVisualModel : public FlowVRRenderObject, public sofa::core::VisualModel
{
protected:
    DataField<float> f_scale;
    DataField<Vec3f> f_trans;
    flowvr::render::ChunkRenderWriter* scene;
    bool *scratch;
    flowvr::ModuleAPI* module;
public:
    FlowVRRenderVisualModel()
        : f_scale(dataField(&f_scale,1.0f,"scale","scale"))
        , f_trans(dataField(&f_trans,Vec3f(0,0,0),"translation","translation"))
        , scene(NULL), module(NULL)
    {
    }

    virtual void renderInit() = 0;
    virtual void renderUpdate()= 0;

    void flowvrInit(flowvr::ModuleAPI* module)
    {
        this->module = module;
        if (this->scene!=NULL)
            renderInit();
    }
    //void flowvrRenderInit(flowvr::render::ChunkRenderWriter* scene)
    void flowvrRenderInit(flowvr::render::ChunkRenderWriter* scene, bool* scratch)
    {
        this->scene = scene;
        this->scratch = scratch;
        if (this->module!=NULL)
            renderInit();
    }
    void flowvrRenderUpdate(flowvr::render::ChunkRenderWriter* /*scene*/)
    {
        renderUpdate();
    }

    void initTextures()
    {
    }

    void update()
    {
        //if (scene!=NULL)
        //    renderUpdate();
    }

    void draw()
    {
    }
};

template<class DataTypes>
class FlowVRRenderMesh : public FlowVRRenderVisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;

    DataField<std::string> vShader;
    DataField<std::string> pShader;
    DataField<Vec4f> color;

    sofa::component::topology::MeshTopology* topology;
    sofa::core::componentmodel::behavior::MechanicalState<DataTypes>* mmodel;

    flowvr::ID idP;
    flowvr::ID idVB;
    flowvr::ID idVBN;
    flowvr::ID idIB;
    flowvr::ID idVS;
    flowvr::ID idPS;

    int lastMeshRev;
    int lastPosRev;

    VecCoord n;
    Coord bbmin;
    Coord bbmax;

    FlowVRRenderMesh()
        : vShader(dataField(&vShader, std::string("shaders/default_v.cg"), "vshader", "vertex shader name"))
        , pShader(dataField(&pShader, std::string("shaders/default_p.cg"), "pshader", "pixel shader name"))
        , color(dataField(&color, Vec4f(1, 1, 1, 0.5f), "color", "RGBA color value"))
        , topology(NULL)
        , mmodel(NULL)
        , idP(0)
        , idVB(0)
        , idVBN(0)
        , idIB(0)
        , idVS(0)
        , idPS(0)
        , lastMeshRev(-1)
        , lastPosRev(-1)
    {
    }

    void renderInit()
    {
    }

    void computeNormals()
    {
        const VecCoord& x = *mmodel->getX();
        const sofa::component::topology::MeshTopology::SeqTriangles& triangles = topology->getTriangles();
        const sofa::component::topology::MeshTopology::SeqQuads& quads = topology->getQuads();
        n.resize(x.size());
        for(unsigned int i=0; i<x.size(); i++)
        {
            n[i] = Coord();
        }
        for(unsigned int i=0; i<triangles.size(); i++)
        {
            const Coord& pt1 = x[triangles[i][0]];
            const Coord& pt2 = x[triangles[i][1]];
            const Coord& pt3 = x[triangles[i][2]];
            Coord normal = cross(pt2-pt1,pt3-pt1);
            normal.normalize();
            n[triangles[i][0]] += normal;
            n[triangles[i][1]] += normal;
            n[triangles[i][2]] += normal;
        }
        for(unsigned int i=0; i<quads.size(); i++)
        {
            const Coord& pt1 = x[quads[i][0]];
            const Coord& pt2 = x[quads[i][1]];
            const Coord& pt3 = x[quads[i][2]];
            const Coord& pt4 = x[quads[i][3]];
            Coord normal = cross(pt2-pt1,pt3-pt1) + cross(pt3-pt1,pt4-pt1);
            normal.normalize();
            n[quads[i][0]] += normal;
            n[quads[i][1]] += normal;
            n[quads[i][2]] += normal;
            n[quads[i][3]] += normal;
        }
        for(unsigned int i=0; i<x.size(); i++)
        {
            n[i].normalize();
        }
    }

    void computeBBox()
    {
        const VecCoord& x = *mmodel->getX();
        if (x.size()==0)
        {
            bbmin = bbmax = Coord();
        }
        else
        {
            bbmin = bbmax = x[0];
            for(unsigned int i=1; i<x.size(); i++)
            {
                const Coord& c = x[i];
                for (unsigned int j=0; j<c.size(); j++)
                {
                    if (c[j] < bbmin[j]) bbmin[j] = c[j];
                    else if (c[j] > bbmax[j]) bbmax[j] = c[j];
                }
            }
        }
    }

    void init()
    {
        mmodel = dynamic_cast<sofa::core::componentmodel::behavior::MechanicalState<DataTypes>*>(getContext()->getMechanicalState());
        topology = dynamic_cast<sofa::component::topology::MeshTopology*>(getContext()->getTopology());
        if (!module || !scene || !mmodel) return;

        renderUpdate();
    }

    void renderUpdate()
    {
        if (!idP)
        {
            *scratch = false;
            idP = module->generateID();
            scene->addPrimitive(idP, getName().c_str());
            prevP.push_back(idP);
            if (!vShader.getValue().empty())
            {
                idVS = module->generateID();
                scene->loadVertexShader(idVS, vShader.getValue().c_str());
                prevP.push_back(idVS);
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::VSHADER, "", idVS);
            }
            if (!pShader.getValue().empty())
            {
                idPS = module->generateID();
                scene->loadPixelShader(idPS, pShader.getValue().c_str());
                prevP.push_back(idPS);
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::PSHADER, "", idPS);
            }
            scene->addParamEnum(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "Proj", flowvr::render::ChunkPrimParam::Projection);
            scene->addParamEnum(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "ModelViewProj", flowvr::render::ChunkPrimParam::ModelViewProjection);
            scene->addParamEnum(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "ModelViewIT", flowvr::render::ChunkPrimParam::ModelView_InvTrans);
            ftl::Vec3f light(1,1,2); light.normalize();
            scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, "lightdir", light);
            scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "color", ftl::Vec4f(color.getValue().ptr())); //ftl::Vec4f(1, 1, 1, 0.5));

            // SOFA = trans + scale * FLOWVR
            // FLOWVR = SOFA * 1/scale - trans * 1/scale

            scene->addParam(idP, flowvr::render::ChunkPrimParam::TRANSFORM_POSITION, "", ftl::Vec3f(f_trans.getValue().ptr())*(-1/f_scale.getValue()));
            scene->addParam(idP, flowvr::render::ChunkPrimParam::TRANSFORM_SCALE, "", ftl::Vec3f(1,1,1)/f_scale.getValue());
        }
        if (mmodel)
        {
            computeBBox();
            if (!idVB)
            {
                *scratch = false;
                idVB = module->generateID();
                prevP.push_back(idVB);
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::VBUFFER_ID, "position", idVB);
                scene->addParam(idP, flowvr::render::ChunkPrimParam::VBUFFER_NUMDATA, "position", 0);
            }
            VecCoord& x = *mmodel->getX();
            int types[1] = { ftl::Type::get(x[0]) };
            flowvr::render::BBox bb(ftl::Vec3f((float)bbmin[0],(float)bbmin[1],(float)bbmin[2]),ftl::Vec3f((float)bbmax[0],(float)bbmax[1],(float)bbmax[2]));
            flowvr::render::ChunkVertexBuffer* vb = scene->addVertexBuffer(idVB, x.size(), 1, types, bb);
            vb->gen = ++lastPosRev;
            memcpy(vb->data(), &(x[0]), vb->dataSize());
        }
        if (mmodel && topology)
        {
            computeNormals();
            if (!idVBN)
            {
                *scratch = false;
                idVBN = module->generateID();
                prevP.push_back(idVB);
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::VBUFFER_ID, "normal", idVBN);
                scene->addParam(idP, flowvr::render::ChunkPrimParam::VBUFFER_NUMDATA, "normal", 0);
            }
            int types[1] = { ftl::Type::get(n[0]) };
            flowvr::render::BBox bb(ftl::Vec3f((float)bbmin[0],(float)bbmin[1],(float)bbmin[2]),ftl::Vec3f((float)bbmax[0],(float)bbmax[1],(float)bbmax[2]));
            flowvr::render::ChunkVertexBuffer* vb = scene->addVertexBuffer(idVBN, n.size(), 1, types, bb);
            vb->gen = lastPosRev;
            memcpy(vb->data(), &(n[0]), vb->dataSize());

            if (!idIB)
            {
                *scratch = false;
                idIB = module->generateID();
                prevP.push_back(idIB);
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::IBUFFER_ID, "", idIB);
            }
            if (topology->getRevision() != lastMeshRev)
            {
                lastMeshRev = topology->getRevision();
                const sofa::component::topology::MeshTopology::SeqTriangles& triangles = topology->getTriangles();
                const sofa::component::topology::MeshTopology::SeqQuads& quads = topology->getQuads();
                if (quads.empty())
                {
                    // only triangles
                    flowvr::render::ChunkIndexBuffer* ib = scene->addIndexBuffer(idIB, triangles.size()*3, ftl::Type::Int, flowvr::render::ChunkIndexBuffer::Triangle);
                    ib->gen = lastMeshRev;
                    memcpy(ib->data(), &(triangles[0]), ib->dataSize());
                }
                else if (triangles.empty())
                {
                    // only quads
                    flowvr::render::ChunkIndexBuffer* ib = scene->addIndexBuffer(idIB, quads.size()*4, ftl::Type::Int, flowvr::render::ChunkIndexBuffer::Quad);
                    ib->gen = lastMeshRev;
                    memcpy(ib->data(), &(quads[0]), ib->dataSize());
                }
                else
                {
                    // both triangles and quads -> convert all to triangles
                    flowvr::render::ChunkIndexBuffer* ib = scene->addIndexBuffer(idIB, triangles.size()*3 + quads.size()*6, ftl::Type::Int, flowvr::render::ChunkIndexBuffer::Triangle);
                    ib->gen = lastMeshRev;
                    memcpy(ib->data(), &(triangles[0]), triangles.size()*3*sizeof(int));
                    int* dest = ((int*)ib->data()) + triangles.size()*3;
                    for (unsigned int i=0; i<quads.size(); i++)
                    {
                        *(dest++) = quads[i][0];
                        *(dest++) = quads[i][1];
                        *(dest++) = quads[i][2];
                        *(dest++) = quads[i][0];
                        *(dest++) = quads[i][2];
                        *(dest++) = quads[i][3];
                    }
                }
            }
        }
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<sofa::core::componentmodel::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const FlowVRRenderMesh<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
};

SOFA_DECL_CLASS(FlowVRRenderMesh)
int FlowVRRenderMesh3fClass = sofa::core::RegisterObject("FlowVRRender Visual Model")
        .add< FlowVRRenderMesh<Vec3fTypes> >()
        .add< FlowVRRenderMesh<Vec3dTypes> >()
        ;

} // namespace SofaFlowVR

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();

    std::string fileName = "/home/allardj/work/sig07et/data/test1.scn";
    //int nbIter = 500;

    if (argc>1)
        fileName = argv[1];

    sofa::gui::SofaGUI::Init(argv[0]);

    GNode* groot = NULL;

    if (!fileName.empty())
    {
        groot = Simulation::load(fileName.c_str());
    }

    if (groot==NULL)
    {
        groot = new GNode;
    }

    sofa::gui::SofaGUI::MainLoop(groot,fileName.c_str());

    return 0;
}
