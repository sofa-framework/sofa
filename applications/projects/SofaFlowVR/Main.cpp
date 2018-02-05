/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <fstream>
#include <stack>

#include <flowvr/module.h>
#include <flowvr/render/chunkwriter.h>
//#include <flowvr/interact/chunkwriter.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/tree/DeleteVisitor.h>
#include <SofaSimulationTree/init.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/MinProximityIntersection.h>
#include <SofaBaseCollision/BruteForceDetection.h>
#include <SofaComponentMain/init.h>

#include <SofaBaseVisual/VisualModelImpl.h>
#include <SofaOpenglVisual/OglModel.h>

#ifdef SOFA_GPU_CUDA
#include <sofa/gpu/cuda/CudaDistanceGridCollisionModel.h>
#endif

#include <sofa/gui/SofaGUI.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>

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

template<int L, int C, typename real>
Type get(const sofa::defaulttype::Mat<L,C,real>&)
{
    return (Type)matrix(get(real()),L,C);
}

}

}

namespace SofaFlowVR
{

//flowvr::ModuleAPI* module = NULL;

//using namespace flowvr::render;
//using namespace flowvr::interact;

//using namespace Sofa::Abstract;
//using namespace Sofa::Components;
using namespace sofa::defaulttype;
using namespace sofa::simulation;

class FlowVRModule;

class FlowVREvent : public sofa::core::objectmodel::Event
{
public:
    FlowVREvent() : from(NULL) {}
    virtual ~FlowVREvent() {}
    FlowVRModule* from;
};

class FlowVRPreInitEvent : public FlowVREvent
{
public:
    virtual ~FlowVRPreInitEvent() {}
    std::set<flowvr::Port*>* ports;
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
    Data<std::string> modName;
    FlowVRModule* mod;


    FlowVRObject()
        : modName(initData(&modName, "module", "Name of FlowVR Module"))
        , mod(NULL)
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

    virtual void flowvrPreInit(std::set<flowvr::Port*>* /*ports*/)
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

    virtual void handleEvent(sofa::core::objectmodel::Event* event);

};

std::map<std::string, flowvr::ModuleAPI*> moduleMap;
std::map<std::string, flowvr::InputPort*> inputPortMap;
std::map<std::string, flowvr::OutputPort*> outputPortMap;
std::map<std::string, flowvr::render::SceneOutputPort*> sceneOutputPortMap;

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
    Data<double> f_dt;
    Data<float> f_scale;
    Data<Vec3f> f_trans;
    Data< vector<std::string> > f_inputPorts;
    Data< vector<std::string> > f_outputPorts;
    int it;
    double lasttime;
    bool step;
    FlowVRModule()
        : module(NULL)
        , f_dt(initData(&f_dt,0.0,"dt","simulation time interval between flowvr iteration"))
        , f_scale(initData(&f_scale,1.0f,"scale","scale"))
        , f_trans(initData(&f_trans,Vec3f(0,0,0),"translation","translation"))
        , f_inputPorts(initData(&f_inputPorts,"inputPorts","additional input ports to be defined"))
        , f_outputPorts(initData(&f_outputPorts,"outputPorts","additional output ports to be defined"))
        , it(-1)
        , lasttime(0.0), step(false)
    {
        mod = this;
    }

    virtual void init()
    {
        if (module!=NULL) return;
        std::set<flowvr::Port*> ports;
        std::cout << "Sending FlowVRPreInit"<<std::endl;
        FlowVRPreInitEvent ev;
        ev.from = this;
        ev.ports = &ports;
        getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &ev);
        const vector<std::string>& inputPorts = f_inputPorts.getValue();
        for (vector<std::string>::const_iterator it = inputPorts.begin(); it != inputPorts.end(); ++it)
            ports.insert(createInputPort(it->c_str()));
        const vector<std::string>& outputPorts = f_outputPorts.getValue();
        for (vector<std::string>::const_iterator it = outputPorts.begin(); it != outputPorts.end(); ++it)
            ports.insert(createOutputPort(it->c_str()));
        std::vector<flowvr::Port*> vports;
        vports.insert(vports.end(), ports.begin(), ports.end());
        //module = flowvr::initModule(vports);
        module = createModule(vports);
        if (module == NULL)
        {
            std::cerr << "SofaFlowVR: module creation failed. Exit."<<std::endl;
            std::exit(1);
        }
        std::cout << "Sending FlowVRInit"<<std::endl;
        FlowVRInitEvent ev2;
        ev2.from = this;
        ev2.module = module;
        getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &ev2);
    }

    virtual void animateBegin(double /*dt*/)
    {
        if (module==NULL) return;
        if (it!=-1 && f_dt.getValue()>0 && getContext()->getTime()<lasttime+f_dt.getValue() && getContext()->getTime()>=lasttime) return;
        if (!module->wait())
        {
            std::cerr << "SofaFlowVR: module wait method returned 0. Exit."<<std::endl;
            std::exit(1);
        }
        ++it; step = true;
        lasttime = getContext()->getTime();
        //std::cout << "Sending FlowVRBeginIteration"<<std::endl;
        FlowVRBeginIterationEvent ev;
        ev.from = this;
        ev.module = module;
        getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &ev);
    }

    virtual void animateEnd(double /*dt*/)
    {
        if (module==NULL) return;
        if (!step) return;
        step = false;
        //std::cout << "Sending FlowVREndIteration"<<std::endl;
        FlowVREndIterationEvent ev;
        ev.from = this;
        ev.module = module;
        getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &ev);
    }
};

SOFA_DECL_CLASS(FlowVRModule)
int FlowVRModuleClass = sofa::core::RegisterObject("FlowVR main module")
        .add<FlowVRModule>()
        ;


void FlowVRObject::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (AnimateBeginEvent* ev = dynamic_cast<AnimateBeginEvent*>(event))
        animateBegin(ev->getDt());
    if (AnimateEndEvent* ev = dynamic_cast<AnimateEndEvent*>(event))
        animateEnd(ev->getDt());
    if (FlowVREvent* flev = dynamic_cast<FlowVREvent*>(event))
    {
        if (!mod)
        {
            if (modName.getValue().empty() || modName.getValue() == flev->from->modName.getValue())
                mod = flev->from;
            else
                return; // event for another module
        }
        else if (flev->from != mod)
            return; // event for another module
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

//flowvr::interact::ObjectsOutputPort pObjectsOut("objects");
//flowvr::interact::ChunkInteractWriter objects;

class FlowVRInputMesh : public FlowVRObject
{
public:

    flowvr::InputPort* pInFacets;
    flowvr::InputPort* pInPoints;
    flowvr::InputPort* pInMatrix;

    Data<bool> computeV;
    Data<double> maxVDist;

    // Velocity is estimated by searching the nearest primitive from each new point
    // To do it we need to create an additionnal PointModel collision model, as well as a Detection and Intersection class
    sofa::simulation::tree::GNode * newPointsNode;
    typedef sofa::simulation::tree::GNode::Sequence<sofa::core::CollisionModel>::iterator CMIterator;
    sofa::component::container::MechanicalObject<Vec3Types> * newPoints;
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
        , computeV( initData(&computeV, false, "computeV", "estimate velocity by detecting nearest primitive of previous model") )
        , maxVDist( initData(&maxVDist,   1.0, "maxVDist", "maximum distance to use for velocity estimation") )
        , newPointsNode(NULL), newPointsCM(NULL), intersection(NULL), detection(NULL)
        , facetsLastIt(-20), pointsLastIt(-20), matrixLastIt(-20), motionLastTime(-1000)
    {
        matrix.identity();
    }

    ~FlowVRInputMesh()
    {
        if (newPointsNode != NULL)
        {
            newPointsNode->execute<sofa::simulation::tree::DeleteVisitor>();
            delete newPointsNode;
        }
    }

    virtual void flowvrPreInit(std::set<flowvr::Port*>* ports)
    {
        std::cout << "Received FlowVRPreInit"<<std::endl;
        ports->insert(pInFacets);
        ports->insert(pInPoints);
        ports->insert(pInMatrix);
    }

    virtual void init()
    {
        this->FlowVRObject::init();
        if (computeV.getValue())
        {
            newPointsNode = new sofa::simulation::tree::GNode("newPoints");
            newPointsNode->addObject ( newPoints = new sofa::component::container::MechanicalObject<Vec3Types> );
            newPointsNode->addObject ( newPointsCM = new sofa::component::collision::PointModel );

            newPointsNode->addObject ( intersection = new sofa::component::collision::MinProximityIntersection );
            intersection->setAlarmDistance(maxVDist.getValue());
            intersection->setContactDistance(0); //maxVDist.getValue());

            newPointsNode->addObject ( detection = new sofa::component::collision::BruteForceDetection );
            detection->setIntersectionMethod(intersection);

            newPointsNode->execute<sofa::simulation::InitVisitor>();
        }
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* module)
    {
        //std::cout << "Received FlowVRBeginIteration"<<std::endl;
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
            const Vec3f trans = mod->f_trans.getValue();
            const float scale = mod->f_scale.getValue();

            BaseObject* mmodel = getContext()->getMechanicalState();
#ifndef SOFA_FLOAT
            sofa::core::behavior::MechanicalState<Vec3dTypes>* mmodel3d;
            if ((mmodel3d = dynamic_cast<sofa::core::behavior::MechanicalState<Vec3dTypes>*>(mmodel))!=NULL)
            {
                bool doComputeV = (computeV.getValue() && newPoints != NULL && motionLastTime != -1000);

                sofa::core::behavior::MechanicalState<Vec3dTypes>* mm;
                if (doComputeV)
                {
                    //std::cout << "Copying "<<nbv<<" vertices and estimate velocity"<<std::endl;
                    mm = newPoints; // put new data in newPoints state
                }
                else
                {
                    //std::cout << "Copying "<<nbv<<" vertices to mmodel3d"<<std::endl;
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
                    float scale2 = 1.0;
                    if (matrix[3][3] != 0) scale2 = 1/matrix[3][3];
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        Vec3f v = vertices[i]*scale;
                        v += trans;
                        Vec4f tv = matrix * Vec4f(v[0],v[1],v[2],1.0f);
                        x[i] = Vec3d(tv[0],tv[1],tv[2])*scale2;
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
                    detection->beginNarrowPhase();
                    for (CMIterator it = node->collisionModel.begin(), itend = node->collisionModel.end(); it != itend ; ++it)
                    {
                        sofa::core::CollisionModel* cm2 = *it;
                        std::cout << "computeV: narrow phase detection with "<<cm2->getClassName()<<std::endl;
                        detection->addCollisionPair(std::make_pair(newPointsCM->getFirst(), cm2->getFirst()));
                        //detection->addCollisionPair(std::make_pair(cm2, newPointsCM));
                    }
                    {
                        sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap& contactMap = detection->getDetectionOutputs();
                        int ncollisions = 0;
                        for (sofa::core::collision::NarrowPhaseDetection::DetectionOutputMap::iterator it1 = contactMap.begin(); it1 != contactMap.end(); ++it1)
                        {
                            sofa::core::collision::NarrowPhaseDetection::DetectionOutputVector& contacts = it1->second;
                            if (contacts.empty()) continue;
                            int newPointsCMIndex = (contacts[0].elem.second.getCollisionModel()==newPointsCM)?1:0;
                            for (sofa::core::collision::NarrowPhaseDetection::DetectionOutputVector::iterator it2 = contacts.begin(); it2 != contacts.end(); ++it2)
                            {
                                sofa::core::collision::DetectionOutput* detection = &*it2;
                                ++ncollisions;
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

#endif
#ifndef SOFA_DOUBLE
            sofa::core::behavior::MechanicalState<Vec3fTypes>* mmodel3f;

            if ((mmodel3f = dynamic_cast<sofa::core::behavior::MechanicalState<Vec3fTypes>*>(mmodel))!=NULL)
            {
                //std::cout << "Copying "<<nbv<<" vertices to mmodel3f"<<std::endl;
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
                    float scale2 = 1.0;
                    if (matrix[3][3] != 0) scale2 = 1/matrix[3][3];
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        Vec3f v = vertices[i]*scale;
                        v += trans;
                        Vec4f tv = matrix * Vec4f(v[0],v[1],v[2],1.0f);
                        x[i] = Vec3f(tv[0],tv[1],tv[2])*scale2;
                    }
                }
            }
#endif
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
            sofa::core::topology::BaseMeshTopology* mesh;
            if ((mesh = dynamic_cast<sofa::core::topology::BaseMeshTopology*>(topology))!=NULL)
            {
                mesh->clear();
                if (valid)
                {
                    //std::cout << "Copying "<<nbi/3<<" triangles to mesh"<<std::endl;
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
                    //std::cout << "Copying "<<mesh->getNbLines()<<" edges to mesh"<<std::endl;
                }
            }
        }
    }
};

SOFA_DECL_CLASS(FlowVRInputMesh)
int FlowVRInputMeshClass = sofa::core::RegisterObject("Import a mesh from a FlowVR InputPort")
        .add< FlowVRInputMesh >()
        ;

template<class T>
class SofaFlowVRAllocator : public sofa::defaulttype::ExtVectorAllocator<T>
{
public:
    typedef typename sofa::defaulttype::ExtVectorAllocator<T>::value_type value_type;
    typedef typename sofa::defaulttype::ExtVectorAllocator<T>::size_type size_type;
    virtual void close(value_type* /*data*/)
    {
        delete this;
    }
    virtual void resize(value_type*& data, size_type size, size_type& maxsize, size_type& cursize)
    {
        if (size > maxsize)
        {
            if (size > (size_type)bRead.getSize()) size = bRead.getSize();
            maxsize = bRead.getSize();
            data = const_cast<value_type*>(bRead.getRead<value_type>());
        }
        cursize = size;
    }
    flowvr::Buffer bRead;
    SofaFlowVRAllocator(flowvr::Buffer& data) : bRead(data) {}
};

//using sofa::component::collision::DistanceGrid;

template<class DistanceGridModel, class DistanceGrid>
class FlowVRInputDistanceGrid : public FlowVRObject
{
public:



    flowvr::InputPort* pInDistance;
    flowvr::InputPort* pInMatrix;
    flowvr::StampInfo stampSizes, stampP0, stampDP, stampBB;

    Data<bool> computeV;
    Data<double> maxVDist;

    Mat4x4f matrix, lastMatrix;
    float mscale; ///< scale part from input matrix
    int distanceLastIt;
    int matrixLastIt;
    double motionLastTime;
    DistanceGrid* curDistGrid;
    DistanceGrid* emptyGrid;
    //flowvr::Message curDistance, lastDistance;

    DistanceGridModel* grid;
    //sofa::core::behavior::MechanicalState<RigidTypes>* rigid;

    FlowVRInputDistanceGrid()
        : pInDistance(createInputPort("distance")), pInMatrix(createInputPort("matrix"))
        , stampSizes("Sizes", flowvr::TypeArray::create(3, flowvr::TypeInt::create()))
        , stampP0("P0", flowvr::TypeArray::create(3, flowvr::TypeFloat::create()))
        , stampDP("DP", flowvr::TypeArray::create(3, flowvr::TypeFloat::create()))
        , stampBB("BB", flowvr::TypeArray::create(6, flowvr::TypeInt::create()))
        , computeV( initData(&computeV, false, "computeV", "estimate velocity by detecting nearest primitive of previous model") )
        , maxVDist( initData(&maxVDist,   1.0, "maxVDist", "maximum distance to use for velocity estimation") )
        , mscale(1.0f), distanceLastIt(-20), matrixLastIt(-20), motionLastTime(-1000), curDistGrid(NULL), emptyGrid(NULL)
        , grid(NULL) //, rigid(NULL)
    {
        pInDistance->stamps->add(&stampSizes);
        pInDistance->stamps->add(&stampP0);
        pInDistance->stamps->add(&stampDP);
        pInDistance->stamps->add(&stampBB);
        matrix.identity();
    }

    ~FlowVRInputDistanceGrid()
    {
        if (emptyGrid!=NULL)
        {
            emptyGrid->release();
        }
        if (curDistGrid!=NULL)
        {
            curDistGrid->release();
        }
    }

    virtual void flowvrPreInit(std::set<flowvr::Port*>* ports)
    {
        std::cout << "Received FlowVRPreInit"<<std::endl;
        ports->insert(pInDistance);
        ports->insert(pInMatrix);
    }

    virtual void init()
    {
        this->FlowVRObject::init();
        sofa::simulation::tree::GNode* node = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
        if (node)
        {
            node->getNodeObject(grid);
            if (grid)
            {
                //rigid = grid->getRigidModel();

                // just create a dummy distance grid for now
                emptyGrid = new DistanceGrid(2,2,2,typename DistanceGrid::Coord(0,0,-100),typename DistanceGrid::Coord(0.001f,0.001f,-99.999f));
                for (int i=0; i<emptyGrid->size(); i++)
                    (*emptyGrid)[i] = emptyGrid->maxDist();
                grid->resize(1);
                curDistGrid = emptyGrid->addRef();
                grid->setGrid(curDistGrid,0);
                grid->setActive(false);
            }
        }
        if (computeV.getValue())
        {
        }
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* module)
    {
        //std::cout << "Received FlowVRBeginIteration"<<std::endl;
        flowvr::Message distance;
        double time = getContext()->getTime();

        bool newmotion = false;
        bool newscale = false;
        if (pInMatrix->isConnected())
        {
            flowvr::Message msgmatrix;
            module->get(pInMatrix,msgmatrix);
            int matrixIt = -1;
            msgmatrix.stamps.read(pInMatrix->stamps->it,matrixIt);
            if (matrixIt != matrixLastIt && msgmatrix.data.getSize()>=(int)sizeof(Mat4x4f))
            {
                lastMatrix = matrix;
                matrix = *msgmatrix.data.getRead<Mat4x4f>(0);

                // remove scale component
                float newmscale = 0.0f;
                for(int j=0; j<3; j++)
                    for(int i=0; i<3; i++)
                        newmscale += matrix[j][i]*matrix[j][i];
                newmscale = sqrt(newmscale/3);
                for(int j=0; j<3; j++)
                    for(int i=0; i<3; i++)
                        matrix[j][i] /= newmscale;
                if (matrix[3][3] != 0 && matrix[3][3] != 1)
                {
                    for(int i=0; i<3; i++)
                        matrix[i][3] /= matrix[3][3];
                    newmscale /= matrix[3][3];
                    matrix[3][3] = 1;
                }
                if (newmscale != mscale)
                    newscale = true;
                mscale = newmscale;

                matrixLastIt = matrixIt;
                if (lastMatrix != matrix || newscale)
                    newmotion = true;

                //if(rigid)
                //    (*rigid->getX())[0].fromMatrix(matrix);
            }
        }

        if (pInDistance->isConnected())
        {
            module->get(pInDistance, distance);

            int distanceIt = -1;
            distance.stamps.read(pInDistance->stamps->it,distanceIt);
            //const unsigned int nbv = points.data.getSize()/sizeof(Vec3f);
            if (distanceIt != distanceLastIt || newscale)
            {
                distanceLastIt = distanceIt;
                //const Vec3f* vertices = points.data.getRead<Vec3f>(0);
                const Vec3f trans = mod->f_trans.getValue();
                const float scale = mod->f_scale.getValue()*mscale;

                int nz = 0;
                int ny = 0;
                int nx = 0;
                Vec3f p0, dp;
                int bbox[6] = {-1,-1,-1,-1,-1,-1};
                distance.stamps.read(stampSizes[0], nz);
                distance.stamps.read(stampSizes[1], ny);
                distance.stamps.read(stampSizes[2], nx);
                distance.stamps.read(stampP0[0], p0[0]);
                distance.stamps.read(stampP0[1], p0[1]);
                distance.stamps.read(stampP0[2], p0[2]);
                distance.stamps.read(stampDP[0], dp[0]);
                distance.stamps.read(stampDP[1], dp[1]);
                distance.stamps.read(stampDP[2], dp[2]);
                for (int i=0; i<6; i++)
                    distance.stamps.read(stampBB[i], bbox[i]);

                if (!nx || !ny || !nz || bbox[0] > bbox[3])
                {
                    // empty grid
                    curDistGrid->release();
                    curDistGrid = emptyGrid->addRef();
                    grid->setActive(false);
                }
                else
                {
                    typename DistanceGrid::Coord pmin = trans + p0*scale;
                    typename DistanceGrid::Coord pmax = pmin + Vec3f(dp[0]*(nx-1),dp[1]*(ny-1),dp[2]*(ny-2))*scale;
                    typename DistanceGrid::Coord bbmin = pmin + Vec3f(dp[0]*bbox[0],dp[1]*bbox[1],dp[2]*bbox[2])*scale;
                    typename DistanceGrid::Coord bbmax = pmin + Vec3f(dp[0]*bbox[3],dp[1]*bbox[4],dp[2]*bbox[5])*scale;

                    /*if (scale==1.0f)
                    {
                        curDistGrid->release();
                        curDistGrid = new DistanceGrid(nx,ny,nz, pmin, pmax, new SofaFlowVRAllocator<typename DistanceGrid::Real>(distance.data));
                    }
                    else*/
                    {
                        curDistGrid->release();
                        curDistGrid = new DistanceGrid(nx,ny,nz, pmin, pmax);
                        const float* in = distance.data.getRead<float>();
                        for (int i=0; i<curDistGrid->size(); i++)
                            (*curDistGrid)[i] = in[i]*scale;
                    }
                    curDistGrid->setBBMin(bbmin);
                    curDistGrid->setBBMax(bbmax);
                    grid->setActive(true);
                }
                newmotion = true;
            }
        }

        if (newmotion)
        {
            Mat3x3f rotation;
            Vec3f translation;
            if (matrixLastIt != -20)
            {
                rotation = matrix;
                translation = matrix.col(3);
            }
            else rotation.identity();
            double dt = 0;
            if (computeV.getValue() && grid->getGrid(0) != emptyGrid)
                dt = (motionLastTime == -1000 || motionLastTime >= time) ? 1 : time-motionLastTime;
            grid->setNewState(0,dt,curDistGrid, rotation, translation);
            motionLastTime = time;
        }
    }
};

SOFA_DECL_CLASS(FlowVRInputDistanceGrid)
int FlowVRInputDistanceGridClass = sofa::core::RegisterObject("Import a distance field from a FlowVR InputPort")
        .add< FlowVRInputDistanceGrid<sofa::component::collision::RigidDistanceGridCollisionModel,sofa::component::collision::DistanceGrid> >()
#ifdef SOFA_GPU_CUDA
        .add< FlowVRInputDistanceGrid<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,sofa::gpu::cuda::CudaDistanceGrid> >()
#endif
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
    bool* scratch;
};

std::vector<flowvr::ID> prevP;
std::map<std::string,flowvr::ID> prevVS;
std::map<std::string,flowvr::ID> prevPS;
std::map<std::string,flowvr::ID> prevVB;
std::map<std::string,flowvr::ID> prevIB;
std::map<std::string,flowvr::ID> prevT;
std::map<std::string,flowvr::ID> prevNM;

class FlowVRRenderObject : public FlowVRObject
{
public:
    void clearSharedResources(flowvr::render::ChunkRenderWriter* scene)
    {
        for(std::vector<flowvr::ID>::const_iterator it = prevP.begin(), itend = prevP.end(); it != itend; ++it)
            scene->delPrimitive(*it);
        prevP.clear();
        for(std::map<std::string,flowvr::ID>::const_iterator it = prevVS.begin(), itend = prevVS.end(); it != itend; ++it)
            scene->delVertexShader(it->second);
        prevVS.clear();
        for(std::map<std::string,flowvr::ID>::const_iterator it = prevPS.begin(), itend = prevPS.end(); it != itend; ++it)
            scene->delPixelShader(it->second);
        prevPS.clear();
        for(std::map<std::string,flowvr::ID>::const_iterator it = prevVB.begin(), itend = prevVB.end(); it != itend; ++it)
            scene->delVertexBuffer(it->second);
        prevVB.clear();
        for(std::map<std::string,flowvr::ID>::const_iterator it = prevIB.begin(), itend = prevIB.end(); it != itend; ++it)
            scene->delIndexBuffer(it->second);
        prevIB.clear();
        for(std::map<std::string,flowvr::ID>::const_iterator it = prevT.begin(), itend = prevT.end(); it != itend; ++it)
            scene->delTexture(it->second);
        prevT.clear();
        for(std::map<std::string,flowvr::ID>::const_iterator it = prevNM.begin(), itend = prevNM.end(); it != itend; ++it)
            scene->delTexture(it->second);
        prevNM.clear();
    }

    flowvr::ID loadSharedTexture(flowvr::render::ChunkRenderWriter* scene, const char* filename)
    {
        flowvr::ID& id = prevT[filename];
        if (!id)
        {
            id = mod->module->generateID();
            flowvr::render::ChunkTexture* t = scene->loadTexture(id, filename);
            if (t)
                std::cout << "FlowVR Render: loaded " << t->nx << "x" << t->ny << "x" << ftl::Type::nx(t->pixelType)*(ftl::Type::elemSize(t->pixelType)==0?1:8*ftl::Type::elemSize(t->pixelType)) << " texture " << filename << std::endl;
        }
        return id;
    }

    flowvr::ID loadSharedTextureNormalMap(flowvr::render::ChunkRenderWriter* scene, const char* filename)
    {
        flowvr::ID& id = prevNM[filename];
        if (!id)
        {
            id = mod->module->generateID();
            flowvr::render::ChunkTexture* t = scene->loadTextureNormalMap(id, filename);
            if (t)
            {
                std::cout << "FlowVR Render: loaded " << t->nx << "x" << t->ny << "x" << ftl::Type::nx(t->pixelType)*(ftl::Type::elemSize(t->pixelType)==0?1:8*ftl::Type::elemSize(t->pixelType)) << " normal map " << filename << std::endl;
                std::string outname(filename,strrchr(filename,'.')?strrchr(filename,'.'):filename+strlen(filename));
                outname+="-NM.png";
                scene->saveTexture(t,outname.c_str());
            }
        }
        return id;
    }

    flowvr::ID loadSharedVertexShader(flowvr::render::ChunkRenderWriter* scene, const std::string& filename, const std::string& predefs="")
    {
        flowvr::ID& id = prevVS[filename+"\n"+predefs];
        if (!id)
        {
            id = mod->module->generateID();
            scene->loadVertexShader(id, filename, (predefs.empty()?NULL:predefs.c_str()));
        }
        return id;
    }

    flowvr::ID loadSharedPixelShader(flowvr::render::ChunkRenderWriter* scene, const std::string& filename, const std::string& predefs="")
    {
        flowvr::ID& id = prevPS[filename+"\n"+predefs];
        if (!id)
        {
            id = mod->module->generateID();
            scene->loadPixelShader(id, filename, (predefs.empty()?NULL:predefs.c_str()));
        }
        return id;
    }

    flowvr::ID addVertexBuffer(flowvr::render::ChunkRenderWriter* /*scene*/)
    {
        static int count = 0;
        char buf[16];
        sprintf(buf,"default%d",count);
        ++count;
        flowvr::ID& id = prevVB[buf];
        if (!id)
        {
            id = mod->module->generateID();
        }
        return id;
    }

    flowvr::ID addIndexBuffer(flowvr::render::ChunkRenderWriter* /*scene*/)
    {
        static int count = 0;
        char buf[16];
        sprintf(buf,"default%d",count);
        ++count;
        flowvr::ID& id = prevIB[buf];
        if (!id)
        {
            id = mod->module->generateID();
        }
        return id;
    }

    flowvr::ID addPrimitive(flowvr::render::ChunkRenderWriter* scene, const char* name)
    {
        flowvr::ID id = mod->module->generateID();
        prevP.push_back(id);
        scene->addPrimitive(id, name);
        return id;
    }





    virtual void flowvrRenderInit(flowvr::render::ChunkRenderWriter* /*scene*/, bool* /*scratch*/)
    {
    }

    virtual void flowvrRenderUpdate(flowvr::render::ChunkRenderWriter* /*scene*/, bool* /*scratch*/)
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
                flowvrRenderUpdate(ev->scene, ev->scratch);
        }
    }
};

class FlowVRRenderWriter : public FlowVRRenderObject
{
public:
    flowvr::render::SceneOutputPort* pOutScene;
    flowvr::render::ChunkRenderWriter scene;
    bool init;
    bool scratch;
    FlowVRRenderWriter()
        : pOutScene(createSceneOutputPort()), init(false), scratch(false)
    {
        clearSharedResources(&scene);
    }

    virtual void flowvrPreInit(std::set<flowvr::Port*>* ports)
    {
        std::cout << "Received FlowVRPreInit"<<std::endl;
        ports->insert(pOutScene);
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* /*module*/)
    {
        if (!init)
        {
            FlowVRRenderInitEvent ev;
            ev.from = mod;
            ev.scene = &scene;
            ev.scratch = &scratch;
            getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &ev);
            init = true;
        }
    }

    virtual void flowvrEndIteration(flowvr::ModuleAPI* /*module*/)
    {
        FlowVRRenderUpdateEvent ev;
        ev.from = mod;
        ev.scene = &scene;
        ev.scratch = &scratch;
        getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &ev);
        scene.put(pOutScene, scratch);
        scratch = true;
    }

    virtual void flowvrInit(flowvr::ModuleAPI* module)
    {
        FlowVRRenderObject::flowvrInit(module);
        if (!init)
        {
            //FlowVRRenderInitEvent ev;
            //ev.from = mod;
            //ev.scene = &scene;
            //getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &ev);
            scene.put(pOutScene);
            //init = true;
        }
    }

};

SOFA_DECL_CLASS(FlowVRRenderWriter)
int FlowVRRenderWriterClass = sofa::core::RegisterObject("FlowVRRender scene manager")
        .add<FlowVRRenderWriter>()
        ;

class FlowVRRenderVisualModel : public FlowVRRenderObject//, public sofa::core::VisualModel
{
protected:
    flowvr::render::ChunkRenderWriter* scene;
    bool *scratch;
    flowvr::ModuleAPI* module;
public:
    FlowVRRenderVisualModel()
        : scene(NULL), scratch(NULL), module(NULL)
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
    void flowvrRenderUpdate(flowvr::render::ChunkRenderWriter* scene, bool* scratch)
    {
        if (this->scene == NULL)
        {
            std::cout << "LIVE creation of FlowVRRenderVisualModel detected."<<std::endl;
            this->module = mod->module;
            flowvrRenderInit(scene, scratch);
        }
        renderUpdate();
    }

    /*
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
    */
};

class FlowVRRenderMesh : public FlowVRRenderVisualModel, public sofa::component::visualmodel::OglModel //VisualModelImpl
{
public:
    typedef sofa::component::visualmodel::OglModel Inherit;
    typedef Inherit::VecCoord VecCoord;
    typedef Inherit::Coord Coord;

    Data<std::string> vShader;
    Data<std::string> pShader;
    Data<bool> useTangent;
    std::string texture;

    std::map<std::string,std::string> paramT;
    std::map<std::string,std::string> paramNM;
    std::map<std::string,std::string> paramVS;
    std::map<std::string,std::string> paramPS;

    flowvr::ID idP;
    flowvr::ID idVB;
    flowvr::ID idVBN;
    flowvr::ID idVBT;
    flowvr::ID idVBTangent;
    flowvr::ID idIB;
    flowvr::ID idVS;
    flowvr::ID idPS;
    flowvr::ID idTex;

    bool posModified;
    bool normModified;
    bool meshModified;

    int lastPosRev;
    int lastNormRev;

    FlowVRRenderMesh()
        : vShader(initData(&vShader, std::string(""), "vshader", "vertex shader name"))
        , pShader(initData(&pShader, std::string(""), "pshader", "pixel shader name"))
        , useTangent(initData(&useTangent, false, "useTangent", "enable computation of texture tangent space vectors (for normal mapping)"))
//    , color(initData(&color, Vec4f(1, 1, 1, 0.5f), "color", "RGBA color value"))
//    , topology(NULL)
//    , mmodel(NULL)
        , idP(0)
        , idVB(0)
        , idVBN(0)
        , idVBT(0)
        , idVBTangent(0)
        , idIB(0)
        , idVS(0)
        , idPS(0)
        , idTex(0)
        , posModified(true)
        , meshModified(true)
//    , lastMeshRev(-1)
        , lastPosRev(-1)
    {
    }

    void parse(sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        this->sofa::component::visualmodel::OglModel::parse(arg);
        std::vector<std::string> params;
        arg->getAttributeList(params);
        for (unsigned int i=0; i<params.size(); ++i)
        {
            const char* name = params[i].c_str();
            if (!strncmp(name,"tex_",4))
            {
                std::string filename = arg->getAttribute(name);
                sofa::helper::system::DataRepository.findFile(filename);
                paramT[name+4] = filename;
            }
            if (!strncmp(name,"bump_",5))
            {
                std::string filename = arg->getAttribute(name);
                sofa::helper::system::DataRepository.findFile(filename);
                paramNM[name+5] = filename;
                useTangent.setValue(true);
            }
            else if (!strncmp(name,"ps_",3))
            {
                paramPS[name+3] = arg->getAttribute(name);
            }
            else if (!strncmp(name,"vs_",3))
            {
                paramVS[name+3] = arg->getAttribute(name);
            }
        }
    }

    virtual bool loadTexture(const std::string& filename)
    {
        texture = filename;
        sofa::helper::system::DataRepository.findFile(texture);
        Inherit::loadTexture(filename);
        return true;
    }

    void renderInit()
    {
    }

    void init()
    {
        VisualModelImpl::init();
    }

    void computePositions()
    {
        Inherit::computePositions();
        posModified = true;
    }

    void computeNormals()
    {
        Inherit::computeNormals();
        //normModified = true;
    }

    void computeMesh(sofa::core::topology::BaseMeshTopology* /*topology*/)
    {
        Inherit::computeMesh();
        meshModified = true;
    }

    void renderUpdate()
    {
        if (!idP)
        {
            *scratch = false;
            idP = addPrimitive(scene, getName().c_str());

            std::string predefs;
            bool useSpecular = material.getValue().useSpecular && material.getValue().shininess > 0.0001 && (material.getValue().specular[0] > 0.0001 || material.getValue().specular[1] > 0.0001 || material.getValue().specular[2] > 0.0001);
            if (useSpecular)
                predefs += "#define SPECULAR 1\n";

            const char* vshader;
            if (!vShader.getValue().empty())
                vshader = vShader.getValue().c_str();
            else if (!texture.empty() && useTangent.getValue())
                vshader = "shaders/obj_color_tangent_v.cg";
            else if (!texture.empty())
                vshader = "shaders/obj_color_v.cg";
            else
                vshader = "shaders/obj_v.cg";

            idVS = loadSharedVertexShader(scene, vshader, predefs);
            scene->addParamID(idP, flowvr::render::ChunkPrimParam::VSHADER, "", idVS);

            const char* pshader;
            if (!pShader.getValue().empty())
                pshader = pShader.getValue().c_str();
            else if (!texture.empty() && useTangent.getValue())
                pshader = "shaders/obj_mat_color_tangent_p.cg";
            else if (!texture.empty())
                pshader = "shaders/obj_mat_color_p.cg";
            else
                pshader = "shaders/obj_mat_p.cg";

            idPS = loadSharedPixelShader(scene, pshader, predefs);
            scene->addParamID(idP, flowvr::render::ChunkPrimParam::PSHADER, "", idPS);

            if (!texture.empty())
            {
                idTex = loadSharedTexture(scene,texture.c_str());
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::TEXTURE, "texcolor", idTex);
            }
            ftl::Vec4f ambient  ( 0.3f, 0.3f, 0.3f, 1.0f);
            ftl::Vec3f diffuse  ( 0.6f, 0.6f, 0.6f);
            ftl::Vec4f specular ( 0.9f, 0.9f, 0.9f, 16.0f);
            if (material.getValue().useAmbient) for (int i=0; i<3; i++) ambient[i] = (float)material.getValue().ambient[i] * 0.5f;
            ambient[3] = material.getValue().diffuse[3]; // alpha
            if (material.getValue().useDiffuse) for (int i=0; i<3; i++) diffuse[i] = (float)material.getValue().diffuse[i];
            if (material.getValue().useSpecular) for (int i=0; i<3; i++) specular[i] = (float)material.getValue().specular[i];
            specular[3] = material.getValue().shininess;
            //scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "color", color); //ftl::Vec4f(1, 1, 1, 0.5));
            scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, "mat_ambient" , ambient );
            scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, "mat_diffuse" , diffuse );
            if (useSpecular)
                scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, "mat_specular", specular);

            if (ambient[3] < 1.0)
            {
                // enable alpha blending
                scene->addParam(idP,flowvr::render::ChunkPrimParam::PARAMOPENGL,"Blend",true);
                scene->addParam(idP,flowvr::render::ChunkPrimParam::ORDER,"",1);
                scene->addParam(idP,flowvr::render::ChunkPrimParam::PARAMOPENGL,"DepthWrite",false);
            }

            // add user-defined textures
            for (std::map<std::string,std::string>::const_iterator it = paramT.begin(), itend = paramT.end(); it != itend; ++it)
            {
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::TEXTURE, it->first.c_str(), loadSharedTexture(scene,it->second.c_str()));
            }
            for (std::map<std::string,std::string>::const_iterator it = paramNM.begin(), itend = paramNM.end(); it != itend; ++it)
            {
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::TEXTURE, it->first.c_str(), loadSharedTextureNormalMap(scene,it->second.c_str()));
            }

            // add user-defined params

            for (std::map<std::string,std::string>::const_iterator it = paramVS.begin(), itend = paramVS.end(); it != itend; ++it)
            {
                float vals[4] = {0,0,0,0};
                int n = sscanf(it->second.c_str(),"%f %f %f %f",&vals[0],&vals[1],&vals[2],&vals[3]);
                if (n==4)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, it->first.c_str(), ftl::Vec4f(vals));
                else if (n==3)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, it->first.c_str(), ftl::Vec3f(vals));
                else if (n==2)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, it->first.c_str(), ftl::Vec2f(vals));
                else if (n==1)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, it->first.c_str(), vals[0]);
                else
                {
                    std::cerr << "ERROR: vertex shader parameter "<<it->first<<": cannot parse value "<<it->second<<std::endl;
                }
            }

            for (std::map<std::string,std::string>::const_iterator it = paramPS.begin(), itend = paramPS.end(); it != itend; ++it)
            {
                float vals[4] = {0,0,0,0};
                int n = sscanf(it->second.c_str(),"%f %f %f %f",&vals[0],&vals[1],&vals[2],&vals[3]);
                if (n==4)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, it->first.c_str(), ftl::Vec4f(vals));
                else if (n==3)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, it->first.c_str(), ftl::Vec3f(vals));
                else if (n==2)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, it->first.c_str(), ftl::Vec2f(vals));
                else if (n==1)
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, it->first.c_str(), vals[0]);
                else
                {
                    std::cerr << "ERROR: vertex shader parameter "<<it->first<<": cannot parse value "<<it->second<<std::endl;
                }
            }

            // SOFA = trans + scale * FLOWVR
            // FLOWVR = SOFA * 1/scale - trans * 1/scale

            const Vec3f trans = mod->f_trans.getValue();
            const float scale = mod->f_scale.getValue();
            const float inv_scale = 1/scale;

            scene->addParam(idP, flowvr::render::ChunkPrimParam::TRANSFORM_POSITION, "", ftl::Vec3f(trans.ptr())*(-inv_scale));
            scene->addParam(idP, flowvr::render::ChunkPrimParam::TRANSFORM_SCALE, "", ftl::Vec3f(inv_scale,inv_scale,inv_scale));
        }
        if (posModified)
        {
            if (!idVB)
            {
                *scratch = false;
                idVB = addVertexBuffer(scene);
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::VBUFFER_ID, "position", idVB);
                scene->addParam(idP, flowvr::render::ChunkPrimParam::VBUFFER_NUMDATA, "position", 0);
            }
            const VecCoord& x = vertices;
            int types[1] = { ftl::Type::get(x[0]) };
            flowvr::render::BBox bb(ftl::Vec3f(bbox[0].ptr()),ftl::Vec3f(bbox[1].ptr()));
            flowvr::render::ChunkVertexBuffer* vb = scene->addVertexBuffer(idVB, x.size(), 1, types, bb);
            vb->gen = ++lastPosRev;
            memcpy(vb->data(), &(x[0]), vb->dataSize());
            posModified = false;

            const VecCoord& n = vnormals;
            if (!n.empty())
            {
                if (!idVBN)
                {
                    *scratch = false;
                    //idVBN = module->generateID();
                    idVBN = addVertexBuffer(scene);
                    scene->addParamID(idP, flowvr::render::ChunkPrimParam::VBUFFER_ID, "normal", idVBN);
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::VBUFFER_NUMDATA, "normal", 0);
                }
                int types[1] = { ftl::Type::get(n[0]) };
                flowvr::render::ChunkVertexBuffer* vb = scene->addVertexBuffer(idVBN, n.size(), 1, types, bb);
                vb->gen = lastPosRev;
                memcpy(vb->data(), &(n[0]), vb->dataSize());
            }

            const ResizableExtVector<TexCoord>& t = vtexcoords;
            if (!t.empty() && !idVBT) // only send texcoords once
            {
                if (!idVBT)
                {
                    *scratch = false;
                    //idVBT = module->generateID();
                    idVBT = addVertexBuffer(scene);
                    scene->addParamID(idP, flowvr::render::ChunkPrimParam::VBUFFER_ID, "texcoord0", idVBT);
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::VBUFFER_NUMDATA, "texcoord0", 0);
                }
                int types[1] = { ftl::Type::get(t[0]) };
                flowvr::render::ChunkVertexBuffer* vb = scene->addVertexBuffer(idVBT, n.size(), 1, types, bb);
                //vb->gen = lastPosRev;
                memcpy(vb->data(), &(t[0]), vb->dataSize());
            }

            if (!n.empty() && !t.empty() && useTangent.getValue())
            {
                if (!idVBTangent)
                {
                    *scratch = false;
                    idVBTangent = addVertexBuffer(scene);
                    scene->addParamID(idP, flowvr::render::ChunkPrimParam::VBUFFER_ID, "tangent", idVBTangent);
                    scene->addParam(idP, flowvr::render::ChunkPrimParam::VBUFFER_NUMDATA, "tangent", 0);
                }

                ResizableExtVector<Vec4f> tangent; tangent.resize(t.size());
                ResizableExtVector<Coord> tangent1; tangent1.resize(t.size());
                ResizableExtVector<Coord> tangent2; tangent2.resize(t.size());

                // see http://www.terathon.com/code/tangent.php

                for (unsigned int i=0; i<triangles.size(); i++)
                {
                    int i1 = triangles[i][0];
                    int i2 = triangles[i][1];
                    int i3 = triangles[i][2];

                    const Coord& v1 = x[i1];
                    const Coord& v2 = x[i2];
                    const Coord& v3 = x[i3];

                    const TexCoord& w1 = t[i1];
                    const TexCoord& w2 = t[i2];
                    const TexCoord& w3 = t[i3];

                    float x1 = v2[0] - v1[0];
                    float x2 = v3[0] - v1[0];
                    float y1 = v2[1] - v1[1];
                    float y2 = v3[1] - v1[1];
                    float z1 = v2[2] - v1[2];
                    float z2 = v3[2] - v1[2];

                    float s1 = w2[0] - w1[0];
                    float s2 = w3[0] - w1[0];
                    float t1 = w2[1] - w1[1];
                    float t2 = w3[1] - w1[1];

                    float r = 1.0f / (s1 * t2 - s2 * t1);
                    Coord sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
                    Coord tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

                    tangent1[i1] += sdir;
                    tangent1[i2] += sdir;
                    tangent1[i3] += sdir;

                    tangent2[i1] += tdir;
                    tangent2[i2] += tdir;
                    tangent2[i3] += tdir;
                }

                for (unsigned int i=0; i<quads.size(); i++)
                {
                    int i1 = quads[i][0];
                    int i2 = quads[i][1];
                    int i3 = quads[i][2];
                    int i4 = quads[i][3];

                    const Coord& v1 = x[i1];
                    const Coord& v2 = x[i2];
                    const Coord& v3 = x[i3];
                    const Coord& v4 = x[i4];

                    const TexCoord& w1 = t[i1];
                    const TexCoord& w2 = t[i2];
                    const TexCoord& w3 = t[i3];
                    const TexCoord& w4 = t[i4];

                    // triangle i1 i2 i3
                    {
                        float x1 = v2[0] - v1[0];
                        float x2 = v3[0] - v1[0];
                        float y1 = v2[1] - v1[1];
                        float y2 = v3[1] - v1[1];
                        float z1 = v2[2] - v1[2];
                        float z2 = v3[2] - v1[2];

                        float s1 = w2[0] - w1[0];
                        float s2 = w3[0] - w1[0];
                        float t1 = w2[1] - w1[1];
                        float t2 = w3[1] - w1[1];

                        float r = 1.0f / (s1 * t2 - s2 * t1);
                        Coord sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
                        Coord tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

                        tangent1[i1] += sdir;
                        tangent1[i2] += sdir;
                        tangent1[i3] += sdir;

                        tangent2[i1] += tdir;
                        tangent2[i2] += tdir;
                        tangent2[i3] += tdir;
                    }

                    // triangle i1 i3 i4
                    {
                        float x1 = v3[0] - v1[0];
                        float x2 = v4[0] - v1[0];
                        float y1 = v3[1] - v1[1];
                        float y2 = v4[1] - v1[1];
                        float z1 = v3[2] - v1[2];
                        float z2 = v4[2] - v1[2];

                        float s1 = w3[0] - w1[0];
                        float s2 = w4[0] - w1[0];
                        float t1 = w3[1] - w1[1];
                        float t2 = w4[1] - w1[1];

                        float r = 1.0f / (s1 * t2 - s2 * t1);
                        Coord sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
                        Coord tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

                        tangent1[i1] += sdir;
                        tangent1[i3] += sdir;
                        tangent1[i4] += sdir;

                        tangent2[i1] += tdir;
                        tangent2[i3] += tdir;
                        tangent2[i4] += tdir;
                    }
                }

                for (unsigned int i=0; i<tangent.size(); i++)
                {
                    const Coord& normal = n[i];
                    Coord t1 = tangent1[i];
                    // Gram-Schmidt orthogonalize
                    t1 -= normal * ( normal * t1 );
                    t1.normalize();
                    tangent[i] = t1;
                    // Calculate handedness
                    tangent[i][3] = ((normal.cross(t1) * tangent2[i]) < 0.0f) ? -1.0f : 1.0f;
                }

                int types[1] = { ftl::Type::get(tangent[0]) };
                flowvr::render::ChunkVertexBuffer* vb = scene->addVertexBuffer(idVBTangent, tangent.size(), 1, types, bb);
                vb->gen = lastPosRev;
                memcpy(vb->data(), &(tangent[0]), vb->dataSize());
            }
        }
        if (meshModified)
        {
            if (!idIB)
            {
                *scratch = false;
                //idIB = module->generateID();
                idIB = addIndexBuffer(scene);
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::IBUFFER_ID, "", idIB);
            }
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
            meshModified = false;
        }
        if (xformsModified)
        {
            Vec3f position = xforms[0].getCenter();
            Quatf rotation = xforms[0].getOrientation();

            Mat4x4f matrix;
            matrix.identity();
            rotation.toMatrix(matrix);
            matrix[0][3] = position[0];
            matrix[1][3] = position[1];
            matrix[2][3] = position[2];

            // SOFA = trans + scale * FLOWVR
            // FLOWVR = (SOFA - trans) * 1/scale
            // FLOWVR = SOFA * 1/scale - trans * 1/scale

            Vec3f trans = mod->f_trans.getValue();
            const float scale = mod->f_scale.getValue();
            const float inv_scale = 1/scale;

            matrix[0][3] -= trans[0];
            matrix[1][3] -= trans[1];
            matrix[2][3] -= trans[2];

            matrix *= inv_scale;

            // TODO: use xforms
            //scene->addParam(idP, flowvr::render::ChunkPrimParam::TRANSFORM_POSITION, "", ftl::Vec3f(trans.ptr())*(-inv_scale));
            //scene->addParam(idP, flowvr::render::ChunkPrimParam::TRANSFORM_SCALE, "", ftl::Vec3f(inv_scale,inv_scale,inv_scale));
            scene->addParam(idP, flowvr::render::ChunkPrimParam::TRANSFORM, "", ftl::Mat4x4f(matrix.ptr()));

            xformsModified = false;
        }
    }
};

SOFA_DECL_CLASS(FlowVRRenderMesh)
int FlowVRRenderMeshClass = sofa::core::RegisterObject("FlowVRRender Visual Model")
        .add< FlowVRRenderMesh >()
        ;

} // namespace SofaFlowVR


#ifndef WIN32
#include <dlfcn.h>
bool loadPlugin(const char* filename)
{
    void *handle;
    handle=dlopen(filename, RTLD_LAZY);
    if (!handle)
    {
        std::cerr<<"Error loading plugin "<<filename<<": "<<dlerror()<<std::endl;
        return false;
    }
    std::cerr<<"Plugin "<<filename<<" loaded."<<std::endl;
    return true;
}
#else
bool loadPlugin(const char* filename)
{
    std::cerr << "Plugin loading not supported on this platform.\n";
    return false;
}
#endif


int main(int argc, char** argv)
{
    sofa::simulation::tree::init();
    sofa::helper::BackTrace::autodump();
    sofa::component::init();

    std::cout << "Using " << sofa::helper::system::atomic<int>::getImplName()<<" atomics." << std::endl;

    sofa::gui::SofaGUI::SetProgramName(argv[0]);
    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    bool        temporaryFile = false;
    std::string dimension="800x600";
    bool fullScreen = false;

    std::string gui = sofa::gui::SofaGUI::GetGUIName();
    std::vector<std::string> plugins;
    std::vector<std::string> files;

    std::string gui_help = "choose the UI (";
    gui_help += sofa::gui::SofaGUI::ListSupportedGUI('|');
    gui_help += ")";

    sofa::helper::parse(&files, "This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'s',"start","start the animation loop")
    .option(&printFactory,'p',"factory","print factory logs")
    .option(&gui,'g',"gui",gui_help.c_str())
    .option(&plugins,'l',"load","load given plugins")
    .option(&loadRecent,'r',"recent","load most recently opened file")
    .option(&dimension,'d',"dimension","width and height of the viewer")
    .option(&fullScreen,'f',"fullScreen","start in full screen")
    .option(&temporaryFile,'t',"temporary","the loaded scene won't appear in history of opened files")
    (argc,argv);

    if(gui!="batch") glutInit(&argc,argv);

    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());


    sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;
    sofa::core::ObjectFactory::AddAlias("VisualModel", "FlowVRRenderMesh", true, &classVisualModel);

    if (!files.empty()) fileName = files[0];

    for (unsigned int i=0; i<plugins.size(); i++)
        loadPlugin(plugins[i].c_str());


    if (int err=sofa::gui::SofaGUI::Init(argv[0],gui.c_str()))
        return err;

    if (fileName.empty())
    {
        if (loadRecent) // try to reload the latest scene
        {
            std::string scenes = "config/Sofa.ini";
            scenes = sofa::helper::system::DataRepository.getFile( scenes );
            std::ifstream mrulist(scenes.c_str());
            std::getline(mrulist,fileName);
            mrulist.close();
        }
        else
            fileName = "Demos/liver.scn";

        fileName = sofa::helper::system::DataRepository.getFile(fileName);
    }


    if (int err=sofa::gui::SofaGUI::createGUI(NULL))
        return err;

    sofa::simulation::tree::GNode* groot = dynamic_cast<sofa::simulation::tree::GNode*>( sofa::simulation::tree::getSimulation()->load(fileName.c_str()));
    if (groot==NULL)  groot = new sofa::simulation::tree::GNode;

    sofa::simulation::tree::getSimulation()->init(groot);
    sofa::gui::SofaGUI::CurrentGUI()->setScene(groot,fileName.c_str(), temporaryFile);


    //=======================================
    //Apply Options

    if (startAnim)  groot->setAnimate(true);

    //Dimension Option
    std::string::size_type separator=dimension.find_first_of('x');
    if (separator != std::string::npos)
    {
        std::string stringWidth=dimension.substr(0,separator);
        std::string stringHeight=dimension.substr(separator+1);
        sofa::gui::SofaGUI::CurrentGUI()->setDimension(atoi(stringWidth.c_str()), atoi(stringHeight.c_str()));
    }

    if (printFactory)
    {
        std::cout << "////////// FACTORY //////////" << std::endl;
        sofa::helper::printFactoryLog();
        std::cout << "//////// END FACTORY ////////" << std::endl;
    }

    if (fullScreen) sofa::gui::SofaGUI::CurrentGUI()->setFullScreen();

    //=======================================
    // Run the main loop
    if (int err=sofa::gui::SofaGUI::MainLoop(groot,fileName.c_str()))
        return err;
    groot = dynamic_cast<sofa::simulation::tree::GNode*>( sofa::gui::SofaGUI::CurrentSimulation() );


    if (groot!=NULL) sofa::simulation::tree::getSimulation()->unload(groot);

    sofa::simulation::tree::cleanup();
    return 0;
}
