#include <iostream>
#include <fstream>
#include <stack>

#include <flowvr/module.h>
#include <flowvr/render/chunkwriter.h>
//#include <flowvr/interact/chunkwriter.h>

#include "Sofa-old/Components/Graph/Simulation.h"
#include "Sofa-old/Components/Graph/Action.h"
#include "Sofa-old/Components/Graph/ParallelActionScheduler.h"
#include "Sofa-old/Components/Graph/CactusStackStorage.h"
#include <Sofa-old/Components/Common/ObjectFactory.h>
#include "Sofa-old/Components/Common/Vec3Types.h"
#include "Sofa-old/Components/Common/BackTrace.h"
#include "Sofa-old/Components/Thread/CTime.h"
#include "Sofa-old/Abstract/Event.h"
#include "Sofa-old/Components/AnimateBeginEvent.h"
#include "Sofa-old/Components/AnimateEndEvent.h"
#include "Sofa-old/Components/MeshTopology.h"
#if defined(SOFA_GUI_QT)
#include "Sofa-old/GUI/QT/Main.h"
#elif defined(SOFA_GUI_FLTK)
#include "Sofa-old/GUI/FLTK/Main.h"
#endif
using Sofa::Components::Thread::CTime;
using Sofa::Components::Thread::ctime_t;
using namespace Sofa::Components::Graph;


//#define VERBOSE

// ---------------------------------------------------------------------
// --- SOFA+FLOWVR integration
// ---------------------------------------------------------------------

namespace ftl
{

namespace Type
{

template<int N, typename real>
Type get(const Sofa::Components::Common::Vec<N,real>&)
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

std::map<std::string, flowvr::ModuleAPI*> moduleMap;
std::map<std::string, flowvr::InputPort*> inputPortMap;
std::map<std::string, flowvr::OutputPort*> outputPortMap;
std::map<std::string, flowvr::render::SceneOutputPort*> sceneOutputPortMap;

flowvr::ModuleAPI* createModule(const std::vector<flowvr::Port*>& ports, const char* name="")
{
    flowvr::ModuleAPI*& module = moduleMap[name];
    if (module==NULL)
        module = flowvr::initModule(ports);
    return module;
}

flowvr::InputPort* createInputPort(const char* name)
{
    flowvr::InputPort*& port = inputPortMap[name];
    if (port==NULL)
        port = new flowvr::InputPort(name);
    return port;
}

flowvr::OutputPort* createOutputPort(const char* name)
{
    flowvr::OutputPort*& port = outputPortMap[name];
    if (port==NULL)
        port = new flowvr::OutputPort(name);
    return port;
}

flowvr::render::SceneOutputPort* createSceneOutputPort(const char* name="scene")
{
    flowvr::render::SceneOutputPort*& port = sceneOutputPortMap[name];
    if (port==NULL)
        port = new flowvr::render::SceneOutputPort(name);
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
    flowvr::InputPort* pInFacets;
    flowvr::InputPort* pInPoints;
    flowvr::InputPort* pInMatrix;

    Mat4x4f matrix;
    int facetsLastIt;
    int pointsLastIt;
    int matrixLastIt;

    FlowVRInputMesh()
        : pInFacets(createInputPort("facets")), pInPoints(createInputPort("points")), pInMatrix(createInputPort("matrix"))
        , facetsLastIt(-20), pointsLastIt(-20), matrixLastIt(-20)
    {
    }

    virtual void flowvrPreInit(std::vector<flowvr::Port*>* ports)
    {
        std::cout << "Received FlowVRPreInit"<<std::endl;
        ports->push_back(pInFacets);
        ports->push_back(pInPoints);
        ports->push_back(pInMatrix);
    }

    virtual void flowvrBeginIteration(flowvr::ModuleAPI* module)
    {
        std::cout << "Received FlowVRBeginIteration"<<std::endl;
        flowvr::Message points, facets;

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
};

class FlowVRRenderUpdateEvent : public FlowVRRenderEvent
{
public:
    virtual ~FlowVRRenderUpdateEvent() {}
    flowvr::render::ChunkRenderWriter* scene;
};

class FlowVRRenderObject : public FlowVRObject
{
    virtual void flowvrRenderInit(flowvr::render::ChunkRenderWriter* /*scene*/)
    {
    }

    virtual void flowvrRenderUpdate(flowvr::render::ChunkRenderWriter* /*scene*/)
    {
    }

    virtual void handleEvent(Sofa::Abstract::Event* event)
    {
        FlowVRObject::handleEvent(event);
        if (dynamic_cast<FlowVRRenderEvent*>(event))
        {
            if (FlowVRRenderInitEvent* ev = dynamic_cast<FlowVRRenderInitEvent*>(event))
                flowvrRenderInit(ev->scene);
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
    bool init;
    FlowVRRenderWriter()
        : pOutScene(createSceneOutputPort()), init(false)
    {
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
            getContext()->propagateEvent(&ev);
            init = true;
        }
    }

    virtual void flowvrEndIteration(flowvr::ModuleAPI* /*module*/)
    {
        FlowVRRenderUpdateEvent ev;
        ev.scene = &scene;
        getContext()->propagateEvent(&ev);
        scene.put(pOutScene);
    }
};

void create(FlowVRRenderWriter*& obj, ObjectDescription* arg)
{
    obj = new FlowVRRenderWriter;
    obj->parseFields( arg->getAttributeMap() );
}

SOFA_DECL_CLASS(FlowVRRenderWriter)
Creator<ObjectFactory, FlowVRRenderWriter> FlowVRRenderWriterClass("FlowVRRenderWriter");

class FlowVRRenderVisualModel : public FlowVRRenderObject, public Sofa::Abstract::VisualModel
{
protected:
    flowvr::render::ChunkRenderWriter* scene;
    flowvr::ModuleAPI* module;
public:
    FlowVRRenderVisualModel()
        : scene(NULL), module(NULL)
    {
    }

    virtual void renderInit() = 0;
    virtual void renderUpdate()= 0;

    void flowvrInit(flowvr::ModuleAPI* module)
    {
        this->module = module;
    }
    void flowvrRenderInit(flowvr::render::ChunkRenderWriter* scene)
    {
        this->scene = scene;
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

    Sofa::Components::MeshTopology* topology;
    MechanicalModel<DataTypes>* mmodel;

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

    FlowVRRenderMesh(MechanicalModel<DataTypes>* = NULL)
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
        const MeshTopology::SeqTriangles& triangles = topology->getTriangles();
        const MeshTopology::SeqQuads& quads = topology->getQuads();
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
        mmodel = dynamic_cast<MechanicalModel<DataTypes>*>(getContext()->getMechanicalModel());
        topology = dynamic_cast<MeshTopology*>(getContext()->getTopology());
        if (!module || !scene || !mmodel) return;

        renderUpdate();
    }

    void renderUpdate()
    {
        if (!idP)
        {
            idP = module->generateID();
            scene->addPrimitive(idP, getName().c_str());
            if (!vShader.getValue().empty())
            {
                idVS = module->generateID();
                scene->loadVertexShader(idVS, vShader.getValue().c_str());
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::VSHADER, "", idVS);
            }
            if (!pShader.getValue().empty())
            {
                idPS = module->generateID();
                scene->loadPixelShader(idPS, pShader.getValue().c_str());
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::PSHADER, "", idPS);
            }
            scene->addParamEnum(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "Proj", flowvr::render::ChunkPrimParam::Projection);
            scene->addParamEnum(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "ModelViewProj", flowvr::render::ChunkPrimParam::ModelViewProjection);
            scene->addParamEnum(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "ModelViewIT", flowvr::render::ChunkPrimParam::ModelView_InvTrans);
            ftl::Vec3f light(1,1,2); light.normalize();
            scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMPSHADER, "lightdir", light);
            scene->addParam(idP, flowvr::render::ChunkPrimParam::PARAMVSHADER, "color", ftl::Vec4f(color.getValue().ptr())); //ftl::Vec4f(1, 1, 1, 0.5));
        }
        if (mmodel)
        {
            computeBBox();
            if (!idVB)
            {
                idVB = module->generateID();
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
                idVBN = module->generateID();
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
                idIB = module->generateID();
                scene->addParamID(idP, flowvr::render::ChunkPrimParam::IBUFFER_ID, "", idIB);
            }
            if (topology->getRevision() != lastMeshRev)
            {
                lastMeshRev = topology->getRevision();
                const MeshTopology::SeqTriangles& triangles = topology->getTriangles();
                const MeshTopology::SeqQuads& quads = topology->getQuads();
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
};

template<class DataTypes>
void create(FlowVRRenderMesh<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< FlowVRRenderMesh<DataTypes>, MechanicalModel<DataTypes> >(obj, arg);
    if (obj != NULL)
    {
        obj->parseFields( arg->getAttributeMap() );
    }
}

SOFA_DECL_CLASS(FlowVRRenderMesh)
Creator<ObjectFactory, FlowVRRenderMesh<Vec3fTypes> > FlowVRRenderMesh3fClass("FlowVRRenderMesh", true);
Creator<ObjectFactory, FlowVRRenderMesh<Vec3dTypes> > FlowVRRenderMesh3dClass("FlowVRRenderMesh", true);

} // namespace SofaFlowVR

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    Sofa::Components::Common::BackTrace::autodump();

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

    return 0;
}
