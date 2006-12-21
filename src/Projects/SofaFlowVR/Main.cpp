#include <iostream>
#include <fstream>
#include <stack>

#include <flowvr/module.h>
#include <flowvr/render/chunkwriter.h>
//#include <flowvr/interact/chunkwriter.h>

#include "Sofa/Components/Graph/Simulation.h"
#include "Sofa/Components/Graph/Action.h"
#include "Sofa/Components/Graph/ParallelActionScheduler.h"
#include "Sofa/Components/Graph/CactusStackStorage.h"
#include <Sofa/Components/Common/ObjectFactory.h>
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/BackTrace.h"
#include "Sofa/Components/Thread/CTime.h"
#include "Sofa/Abstract/Event.h"
#include "Sofa/Components/AnimateBeginEvent.h"
#include "Sofa/Components/AnimateEndEvent.h"
#include "Sofa/Components/MeshTopology.h"
#if defined(SOFA_GUI_QT)
#include "Sofa/GUI/QT/Main.h"
#elif defined(SOFA_GUI_FLTK)
#include "Sofa/GUI/FLTK/Main.h"
#endif
using Sofa::Components::Thread::CTime;
using Sofa::Components::Thread::ctime_t;
using namespace Sofa::Components::Graph;


//#define VERBOSE

// ---------------------------------------------------------------------
// --- SOFA+FLOWVR integration
// ---------------------------------------------------------------------
namespace SofaFlowVR
{

flowvr::ModuleAPI* module = NULL;

//using namespace flowvr::render;
//using namespace flowvr::interact;

using namespace Sofa::Abstract;
using namespace Sofa::Components;
using namespace Sofa::Components::Common;

class FlowVREvent : public Event
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

class FlowVRObject : public virtual Sofa::Abstract::BaseObject
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

    virtual void handleEvent(Sofa::Abstract::Event* event)
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
        module = flowvr::initModule(ports);
        if (module == NULL)
        {
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


void create(FlowVRModule*& obj, ObjectDescription* arg)
{
    obj = new FlowVRModule;
    obj->parseFields( arg->getAttributeMap() );
}
SOFA_DECL_CLASS(FlowVRModule)
Creator<ObjectFactory, FlowVRModule> FlowVRModuleClass("FlowVRModule");

//flowvr::interact::ObjectsOutputPort pObjectsOut("objects");
//flowvr::interact::ChunkInteractWriter objects;

class FlowVRInputMesh : public FlowVRObject
{
public:
    flowvr::InputPort pInFacets;
    flowvr::InputPort pInPoints;
    flowvr::InputPort pInMatrix;

    Mat4x4f matrix;
    int facetsLastIt;
    int pointsLastIt;
    int matrixLastIt;

    FlowVRInputMesh()
        : pInFacets("facets"), pInPoints("points"), pInMatrix("matrix")
        , facetsLastIt(-20), pointsLastIt(-20), matrixLastIt(-20)
    {
    }

    virtual void flowvrPreInit(std::vector<flowvr::Port*>* ports)
    {
        std::cout << "Received FlowVRPreInit"<<std::endl;
        ports->push_back(&pInFacets);
        ports->push_back(&pInPoints);
        ports->push_back(&pInMatrix);
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* module)
    {
        std::cout << "Received FlowVRBeginIteration"<<std::endl;
        flowvr::Message points, facets;

        module->get(&pInPoints, points);
        module->get(&pInFacets, facets);

        bool newmatrix = false;
        if (pInMatrix.isConnected())
        {
            flowvr::Message msgmatrix;
            module->get(&pInMatrix,msgmatrix);
            int matrixIt = -1;
            msgmatrix.stamps.read(pInMatrix.stamps->it,matrixIt);
            if (matrixIt != matrixLastIt && msgmatrix.data.getSize()>=(int)sizeof(Mat4x4f))
            {
                matrix = *msgmatrix.data.getRead<Mat4x4f>(0);
                matrixLastIt = matrixIt;
                newmatrix = true;
            }
        }

        int pointsIt = -1;
        points.stamps.read(pInPoints.stamps->it,pointsIt);
        const unsigned int nbv = points.data.getSize()/sizeof(Vec3f);
        if (pointsIt != pointsLastIt || newmatrix)
        {
            pointsLastIt = pointsIt;
            const Vec3f* vertices = points.data.getRead<Vec3f>(0);

            BaseObject* mmodel = getContext()->getMechanicalModel();
            MechanicalModel<Vec3fTypes>* mmodel3f;
            MechanicalModel<Vec3dTypes>* mmodel3d;
            if ((mmodel3f = dynamic_cast<MechanicalModel<Vec3fTypes>*>(mmodel))!=NULL)
            {
                std::cout << "Copying "<<nbv<<" vertices to mmodel3f"<<std::endl;
                mmodel3f->resize(nbv);
                Vec3fTypes::VecCoord& x = *mmodel3f->getX();
                if (matrixLastIt==-20)
                {
                    for (unsigned int i=0; i<nbv; i++)
                        x[i] = vertices[i];
                }
                else
                {
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        const Vec3f& v = vertices[i];
                        Vec4f tv = matrix * Vec4f(v[0],v[1],v[2],1.0f);
                        x[i] = Vec3f(tv[0],tv[1],tv[2]);
                    }
                }
            }
            else if ((mmodel3d = dynamic_cast<MechanicalModel<Vec3dTypes>*>(mmodel))!=NULL)
            {
                std::cout << "Copying "<<nbv<<" vertices to mmodel3d"<<std::endl;
                mmodel3d->resize(nbv);
                Vec3dTypes::VecCoord& x = *mmodel3d->getX();
                if (matrixLastIt==-20)
                {
                    for (unsigned int i=0; i<nbv; i++)
                        x[i] = vertices[i];
                }
                else
                {
                    for (unsigned int i=0; i<nbv; i++)
                    {
                        const Vec3f& v = vertices[i];
                        Vec4f tv = matrix * Vec4f(v[0],v[1],v[2],1.0f);
                        x[i] = Vec3d(tv[0],tv[1],tv[2]);
                    }
                }
            }
        }

        int facetsIt = -1;
        facets.stamps.read(pInFacets.stamps->it,facetsIt);
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
            MeshTopology* mesh;
            if ((mesh = dynamic_cast<MeshTopology*>(topology))!=NULL)
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

void create(FlowVRInputMesh*& obj, ObjectDescription* arg)
{
    obj = new FlowVRInputMesh;
    obj->parseFields( arg->getAttributeMap() );
}
SOFA_DECL_CLASS(FlowVRInputMesh)
Creator<ObjectFactory, FlowVRInputMesh> FlowVRInputMeshClass("FlowVRInputMesh");

bool init()
{
    //std::vector<flowvr::Port*> ports;
    //ports.push_back(&pObjectsOut);

    //objects.put(&pObjectsOut);

    return true;
}

bool step()
{
    return true;
}

void quit()
{
    //if (module==NULL) return;
    //module->close();
}

} // namespace SofaFlowVR

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    Sofa::Components::Common::BackTrace::autodump();
    SofaFlowVR::init();

    std::string fileName = "/home/allardj/work/sig07et/data/test1.scn";
    //int nbIter = 500;

    if (argc>1)
        fileName = argv[1];

    GNode* groot = NULL;

    if (!fileName.empty())
    {
        groot = Simulation::load(fileName.c_str());
    }

    if (groot==NULL)
    {
        groot = new GNode;
    }

#if defined(SOFA_GUI_QT)
    Sofa::GUI::QT::MainLoop(argv[0],groot,fileName.c_str());
#elif defined(SOFA_GUI_FLTK)
    Sofa::GUI::FLTK::MainLoop(argv[0],groot);
#endif

    SofaFlowVR::quit();

    return 0;
}
