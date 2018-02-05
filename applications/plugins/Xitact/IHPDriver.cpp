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
#define XITACT_VISU
#include "IHPDriver.h"

#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/objectmodel/XitactEvent.h>
//
////force feedback
#include <SofaHaptics/ForceFeedback.h>
#include <SofaHaptics/NullForceFeedbackT.h>
//
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
//
#include <sofa/simulation/Node.h>
#include <cstring>

#include <SofaOpenglVisual/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
//sensable namespace
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>


#include <sofa/helper/system/thread/CTime.h>




namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;
using namespace core::behavior;
using namespace sofa::defaulttype;



SOFA_XITACTPLUGIN_API void UpdateForceFeedBack(void* toolData)
{
    allXiToolDataIHP* myData = static_cast<allXiToolDataIHP*>(toolData);

    //en comm parce que ça plante l'autocomplete
    RigidTypes::VecCoord positionDevs;
    RigidTypes::VecDeriv forceDevs;
    forceDevs.clear();
    positionDevs.resize(myData->xiToolData.size());
    forceDevs.resize(myData->xiToolData.size());


    if(myData->xiToolData[0]->lcp_forceFeedback)
    {
        //get tool state for each xitact
        double pi = 3.1415926535;
        for(unsigned int i=0; i<myData->xiToolData.size(); i++)
        {
            xiTrocarAcquire();
            XiToolState state;
            xiTrocarQueryStates();
            xiTrocarGetState(myData->xiToolData[i]->indexTool, &state);

            Vector3 dir;

            dir[0] = -(double)state.trocarDir[0];
            dir[1] = (double)state.trocarDir[2];
            dir[2] = -(double)state.trocarDir[1];

            double thetaY;
            double thetaX;

            thetaY = (atan2(dir[0],-sqrt(1-dir[0]*dir[0])));
            thetaX = (pi-acos(dir[2]*sqrt(1-dir[0]*dir[0])/(dir[0]*dir[0]-1)));

            //look if thetaX and thetaY are NaN
            if(!(thetaX == thetaX))
            {
                cout<<"ratrapage X"<<endl;
                thetaX=pi;
            }
            if(!(thetaY == thetaY))
            {
                cout<<"ratrapage Y"<<endl;
                thetaY=pi;
            }

            if(dir[1]>=0)
                thetaX*=-1;

            while(thetaY<=0)
                thetaY+=2*pi;
            while(thetaX<=0)
                thetaX+=2*pi;
            while(thetaY>2*pi)
                thetaY-=2*pi;
            while(thetaX>2*pi)
                thetaX-=2*pi;

            //mettre le posBaseglobal dans data
            SolidTypes<double>::Transform sofaWorld_H_base(myData->xiToolData[i]->posBase,myData->xiToolData[i]->quatBase);	  //sofaWorld_H_base

            SolidTypes<double>::Transform tampon = sofaWorld_H_base;

            sofa::helper::Quater<double> qy(Vec3d(0,1,0),thetaY);
            sofa::helper::Quater<double> qx(Vec3d(1,0,0),thetaX);
            SolidTypes<double>::Transform transform2(Vec3d(0.0,0.0,0.0),qx*qy);
            tampon*=transform2;		 //*base_H_trocard

            sofa::helper::Quater<float> quarter3(Vec3d(0.0,0.0,1.0),-state.toolRoll);
            SolidTypes<double>::Transform transform3(Vec3d(0.0,0.0,-state.toolDepth*myData->xiToolData[i]->scale),quarter3);
            tampon*=transform3;		   //*trocard_H_tool

            positionDevs[i].getCenter()=tampon.getOrigin();
            positionDevs[i].getOrientation()=tampon.getOrientation();
        }

        if(myData->xiToolData[0]->lcp_forceFeedback != NULL)
            myData->xiToolData[0]->lcp_forceFeedback->computeForce(positionDevs, forceDevs);

        for(unsigned int i=0; i<myData->xiToolData.size(); i++)
        {
            //cout<<i<<" "<<forceDevs[i]<<endl;
            SolidTypes<double>::SpatialVector Wrench_tool_inWorld(forceDevs[i].getVCenter(), forceDevs[i].getVOrientation());

            SolidTypes<double>::SpatialVector Wrench_tool_inXiatctBase(myData->xiToolData[i]->quatBase.inverseRotate(Wrench_tool_inWorld.getForce()),  myData->xiToolData[i]->quatBase.inverseRotate(Wrench_tool_inWorld.getTorque())  );

            XiToolForce_ ff;
            ff.tipForce[0] = (float)(Wrench_tool_inXiatctBase.getForce()[0] * myData->xiToolData[i]->forceScale);  //OK
            ff.tipForce[1] = (float)-(Wrench_tool_inXiatctBase.getForce()[2] * myData->xiToolData[i]->forceScale);	 //OK
            ff.tipForce[2] = (float)(Wrench_tool_inXiatctBase.getForce()[1] * myData->xiToolData[i]->forceScale);  // OK

            //if(Wrench_tool_inXiatctBase.getForce()[0]>0.0000001)
            //	cout<<"Wrench_tool_inXiatctBase.getForce()"<<Wrench_tool_inXiatctBase.getForce()<<endl;

            ff.rollForce = 0.0f;

            xiTrocarSetForce(myData->xiToolData[i]->indexTool, &ff);
            xiTrocarFlushForces();
        }
    }

    //// Compute actual tool state:
    //xiTrocarAcquire();
    //XiToolState state;
    //xiTrocarQueryStates();
    //xiTrocarGetState(myData->indexTool, &state);

    //Vector3 dir;
    //
    //dir[0] = -(double)state.trocarDir[0];
    //dir[1] = (double)state.trocarDir[2];
    //dir[2] = -(double)state.trocarDir[1];

    //double pi = 3.1415926535;

    //double thetaY;
    //double thetaX;

    //thetaY = (atan2(dir[0],-sqrt(1-dir[0]*dir[0])));
    //thetaX = (pi-acos(dir[2]*sqrt(1-dir[0]*dir[0])/(dir[0]*dir[0]-1)));

    ////look if thetaX and thetaY are NaN
    //if(!(thetaX == thetaX))
    //{
    //	cout<<"ratrapage X"<<endl;
    //	thetaX=pi;
    //}
    //if(!(thetaY == thetaY))
    //{
    //	cout<<"ratrapage Y"<<endl;
    //	thetaY=pi;
    //}

    //if(dir[1]>=0)
    //	thetaX*=-1;

    //while(thetaY<=0)
    //	thetaY+=2*pi;
    //while(thetaX<=0)
    //	thetaX+=2*pi;
    //while(thetaY>2*pi)
    //	thetaY-=2*pi;
    //while(thetaX>2*pi)
    //	thetaX-=2*pi;

    //double toolDpth = state.toolDepth;

    //SolidTypes<double>::Transform sofaWorld_H_base(positionBaseGlobal[0].getCenter(),positionBaseGlobal[0].getOrientation());	  //sofaWorld_H_base
    //SolidTypes<double>::Transform tampon = sofaWorld_H_base;

    //sofa::helper::Quater<double> qy(Vec3d(0,1,0),thetaY);
    //sofa::helper::Quater<double> qx(Vec3d(1,0,0),thetaX);
    //SolidTypes<double>::Transform transform2(Vec3d(0.0,0.0,0.0),qx*qy);
    //tampon*=transform2;		 //*base_H_trocard

    //sofa::helper::Quater<float> quarter3(Vec3d(0.0,0.0,1.0),-state.toolRoll);
    //SolidTypes<double>::Transform transform3(Vec3d(0.0,0.0,-state.toolDepth*myData->scale),quarter3);
    //tampon*=transform3;		   //*trocard_H_tool

    //Vec3d world_pos_tool = sofaWorld_H_base.getOrigin();
    //Quat world_quat_tool = sofaWorld_H_base.getOrientation();

    //SolidTypes<double>::SpatialVector Twist_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0)); // Todo: compute a velocity !!
    //SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0));

    //if (myData->lcp_forceFeedback != NULL)
    //{
    //	(myData->lcp_forceFeedback)->computeWrench(tampon,Twist_tool_inWorld,Wrench_tool_inWorld );
    //}

    // we compute its value in the current Tool frame:
    //	SolidTypes<double>::SpatialVector Wrench_tool_inXiatctBase(world_quat_tool.inverseRotate(Wrench_tool_inWorld.getForce()),  world_quat_tool.inverseRotate(Wrench_tool_inWorld.getTorque())  );

    //	XiToolForce_ ff;
    //	ff.tipForce[0] = (Wrench_tool_inXiatctBase.getForce()[0] * myData->forceScale);  //OK
    //	ff.tipForce[1] = -(Wrench_tool_inXiatctBase.getForce()[2] * myData->forceScale);	 //OK
    //	ff.tipForce[2] = (Wrench_tool_inXiatctBase.getForce()[1] * myData->forceScale);  // OK

    //	if(Wrench_tool_inXiatctBase.getForce()[0]>0.0000001)
    //		cout<<"Wrench_tool_inXiatctBase.getForce()"<<Wrench_tool_inXiatctBase.getForce()<<endl;

    //	ff.rollForce = 0.0f;

    //	xiTrocarSetForce(0, &ff);
    //	xiTrocarFlushForces();
    //}
}


SOFA_XITACTPLUGIN_API bool isInitialized = false;

SOFA_XITACTPLUGIN_API int initDevice(XiToolDataIHP& /*data*/)
{
    std::cout<<"initDevice called:"<<std::endl;////////////////////////////////////////////////////////////////////

    if (isInitialized) return 0;
    isInitialized = true;

    const char* vendor = getenv("XITACT_VENDOR");
    if (!vendor || !*vendor)
        vendor = "INRIA_Lille";
    xiSoftwareVendor(vendor);

    return 0;
}


SOFA_DECL_CLASS(IHPDriver)

int IHPDriverClass = core::RegisterObject("Driver and Controller of IHP Xitact Device")
        .add< IHPDriver >();


IHPDriver::IHPDriver()
    : Scale(initData(&Scale, 1.0, "Scale","Default scale applied to the Phantom Coordinates. "))
    , forceScale(initData(&forceScale, 0.0001, "forceScale","Default scale applied to the force feedback. "))
    , permanent(initData(&permanent, true, "permanent" , "Apply the force feedback permanently"))
    , indexTool(initData(&indexTool, (int)0,"toolIndex", "index of the tool to simulate (if more than 1). Index 0 correspond to first tool."))
    , graspThreshold(initData(&graspThreshold, 0.2, "graspThreshold","Threshold value under which grasping will launch an event."))
    , showToolStates(initData(&showToolStates, false, "showToolStates" , "Display states and forces from the tool."))
    , testFF(initData(&testFF, false, "testFF" , "If true will add force when closing handle. As if tool was entering an elastic body."))
    , RefreshFrequency(initData(&RefreshFrequency, (int)500,"RefreshFrequency", "Frequency of the haptic loop."))
    , xitactVisu(initData(&xitactVisu, false, "xitactVisu", "Visualize the position of the interface in the virtual scene"))
    , positionBase(initData(&positionBase, "positionBase", "position of the base of the device"))
    , locPosBati(initData(&locPosBati,std::string("nodeBati/posBati"),"locPosBati","localisation of the restPosition of the bati"))
    , deviceIndex(initData(&deviceIndex,1,"deviceIndex","index of the device"))
    , openTool(initData(&openTool,"openTool","opening of the tool"))
    , maxTool(initData(&maxTool,1.0,"maxTool","maxTool value"))
    , minTool(initData(&minTool,0.0,"minTool","minTool value"))
{
    std::cout<<"IHPDriver::IHPDriver() called:"<<std::endl;/////////////////////////////////////////////////////////

    myPaceMaker = NULL;
    posTool = NULL;
    nodeTool = NULL;
    this->f_listening.setValue(true);

    noDevice = false;
    graspElasticMode = false;
    findForceFeedback= false;

#ifdef SOFA_DEV
    data.vm_forceFeedback=NULL;
#endif
    //data.lcp_forceFeedback=new NullForceFeedbackT<Rigid3dTypes>();
    data.lcp_forceFeedback=NULL;

    firstDevice = true;

}

IHPDriver::~IHPDriver()
{
    std::cerr<<"IHPDriver::~IHPDriver() called:"<<std::endl;/////////////////////////////////////////////////////////
    xiTrocarRelease();
    this->deleteCallBack();
}

void IHPDriver::cleanup()
{
    /*
    	std::cout<<"IHPDriver::cleanup() called:"<<std::endl;/////////////////////////////////////////////////////////

        isInitialized = false;
    	if (permanent.getValue())
    		this->deleteCallBack();
    */

}

void IHPDriver::setLCPForceFeedback(LCPForceFeedback<Rigid3dTypes>* ff)
{
    std::cout<<"IHPDriver::setLCPForceFeedback() called:"<<std::endl;/////////////////////////////////////////////////////////
    if(firstDevice)
    {
        if(data.lcp_forceFeedback == ff)
        {
            return;
        }

//		if(data.lcp_forceFeedback)
//			delete data.lcp_forceFeedback;
        data.lcp_forceFeedback=NULL;
        data.lcp_forceFeedback =ff;
        data.lcp_true_vs_vm_false = true;
    }
};

#ifdef SOFA_DEV
void IHPDriver::setVMForceFeedback(VMechanismsForceFeedback<defaulttype::Vec1dTypes>* ff)
{
    std::cout<<"IHPDriver::setVMForceFeedback() called:"<<std::endl;/////////////////////////////////////////////////////////
    if(data.vm_forceFeedback == ff)
    {
        return;
    }

//	if(data.vm_forceFeedback)
//		delete data.vm_forceFeedback;
    data.vm_forceFeedback =ff;
    data.lcp_true_vs_vm_false=false;

    std::cout<<"IHPDriver::setVMForceFeedback() ok:"<<std::endl;/////////////////////////////////////////////////////////
};
#endif

void IHPDriver::init()
{

    //dev multi device
    std::cout<<"IHPDDriver::init() called:"<<std::endl;


    if(firstDevice)
    {
        std::cout<<"initilisation of the first device"<<std::endl;
        simulation::Node *context = dynamic_cast<simulation::Node*>(this->getContext());
        context->getTreeObjects<IHPDriver>(&otherXitact);
        cout<<"Xitact found :"<<endl;
        for(unsigned int i=0; i<otherXitact.size(); i++)
        {
            cout<<i<<" - Xitact	: "<<otherXitact[i]->getName()<<endl;
            otherXitact[i]->deviceIndex.setValue(i);
            allData.xiToolData.push_back(&(otherXitact[i]->data));
            otherXitact[i]->firstDevice=false;
        }
        firstDevice=true;
    }

    //mis en commentaire parceque ça fait planter l'autocomplete
    VecCoord& posB = (*positionBase.beginEdit());
    posB.resize(1);
    posB[0].getOrientation().normalize();
    positionBase.endEdit();

    //visual variable
    visualXitactDOF = NULL;
    visualAxesDOF = NULL;
    nodeXitactVisual = NULL;
    nodeAxesVisual = NULL;

    if(visualXitactDOF == NULL && visualAxesDOF == NULL)
    {
        cout<<"init visual for xitact "<<deviceIndex.getValue()<<endl;

        //Xitact Visual Node
        nodeXitactVisual = sofa::simulation::getSimulation()->createNewGraph("nodeXitactVisual "+this->name.getValue());
        visuActif=false;
        if(xitactVisu.getValue())
        {
            simulation::Node *parent = dynamic_cast<simulation::Node *>(this->getContext());
            parent->addChild(nodeXitactVisual);
            nodeXitactVisual->updateContext();
            visuActif = true;
        }

        visualXitactDOF = sofa::core::objectmodel::New< sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> >();
        nodeXitactVisual->addObject(visualXitactDOF);
        visualXitactDOF->name.setValue("rigidDOF");

        //mis en commentaire parce que ça fait planter l'auto complete
        VecCoord& posH =*(visualXitactDOF->x.beginEdit());
        posH.resize(4);
        posH[0]=positionBase.getValue()[0];
        visualXitactDOF->x.endEdit();

        visualXitactDOF->init();
        nodeXitactVisual->updateContext();

        //Axes node
        nodeAxesVisual = sofa::simulation::getSimulation()->createNewGraph("nodeAxesVisual "+this->name.getValue());

        visualAxesDOF = sofa::core::objectmodel::New< sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> >();
        nodeAxesVisual->addObject(visualAxesDOF);
        visualAxesDOF->name.setValue("rigidDOF");

        //mis en commentaire parce que ça fait planter l'auto complete
        VecCoord& posA =*(visualAxesDOF->x.beginEdit());
        posA.resize(3);
        posA[0]=positionBase.getValue()[0];
        posA[1]=positionBase.getValue()[0];
        posA[2]=positionBase.getValue()[0];
        visualAxesDOF->x.endEdit();

        visualAxesDOF->init();
        nodeAxesVisual->updateContext();

        visualNode[0].node = sofa::simulation::getSimulation()->createNewGraph("base");
        visualNode[1].node = sofa::simulation::getSimulation()->createNewGraph("trocar");
        visualNode[2].node = sofa::simulation::getSimulation()->createNewGraph("stylet");
        visualNode[3].node = sofa::simulation::getSimulation()->createNewGraph("stylet up");
        visualNode[4].node = sofa::simulation::getSimulation()->createNewGraph("axe X");
        visualNode[5].node = sofa::simulation::getSimulation()->createNewGraph("axe Y");
        visualNode[6].node = sofa::simulation::getSimulation()->createNewGraph("axe Z");

        for(int i=0; i<7; i++)
        {
            visualNode[i].visu = NULL;
            visualNode[i].mapping = NULL;
            if(visualNode[i].visu == NULL && visualNode[i].mapping == NULL)
            {
                visualNode[i].visu = sofa::core::objectmodel::New< sofa::component::visualmodel::OglModel >();
                visualNode[i].node->addObject(visualNode[i].visu);
                visualNode[i].visu->name.setValue("VisualParticles");
                if(i==0)
                    visualNode[i].visu->fileMesh.setValue("mesh/baseXitact.obj");
                if(i==1)
                    visualNode[i].visu->fileMesh.setValue("mesh/trocar.obj");
                if(i==2)
                    visualNode[i].visu->fileMesh.setValue("mesh/stylusXitact.obj");
                if(i==3)
                    visualNode[i].visu->fileMesh.setValue("mesh/stylusUpXitact.obj");
                if(i==4)
                {
                    visualNode[i].visu->fileMesh.setValue("mesh/axeY.obj");
                    visualNode[i].visu->setScale(5.0,5.0,5.0);
                }
                if(i==5)
                {
                    visualNode[i].visu->fileMesh.setValue("mesh/axeZ.obj");
                    visualNode[i].visu->setScale(5.0,5.0,5.0);
                }
                if(i==6)
                {
                    visualNode[i].visu->fileMesh.setValue("mesh/axeX.obj");
                    visualNode[i].visu->setScale(5.0,5.0,5.0);
                }
                visualNode[i].visu->init();
                visualNode[i].visu->initVisual();
                visualNode[i].visu->updateVisual();
                if(i<4)
                {
                    visualNode[i].mapping = sofa::core::objectmodel::New< sofa::component::mapping::RigidMapping< Rigid3dTypes, ExtVec3fTypes > >();
                    visualNode[i].mapping->setModels(visualXitactDOF.get(),visualNode[i].visu.get());
                }
                else
                {
                    visualNode[i].mapping = sofa::core::objectmodel::New< sofa::component::mapping::RigidMapping< Rigid3dTypes, ExtVec3fTypes > >();
                    visualNode[i].mapping->setModels(visualAxesDOF.get(),visualNode[i].visu.get());
                }
                visualNode[i].node->addObject(visualNode[i].mapping);
                visualNode[i].mapping->name.setValue("RigidMapping");
                visualNode[i].mapping->f_mapConstraints.setValue(false);
                visualNode[i].mapping->f_mapForces.setValue(false);
                visualNode[i].mapping->f_mapMasses.setValue(false);
                visualNode[i].mapping->setPathInputObject("@../RigidDOF");
                visualNode[i].mapping->setPathOutputObject("@VisualParticles");
//				visualNode[i].mapping->m_inputObject.setValue("@../RigidDOF");
//				visualNode[i].mapping->m_outputObject.setValue("@VisualParticles");
                if(i<4)
                    visualNode[i].mapping->index.setValue(i);
                else
                    visualNode[i].mapping->index.setValue(i-4);
                //visualNode[i].mapping->init();
                if(i<4)
                    nodeXitactVisual->addChild(visualNode[i].node);
                else
                    nodeAxesVisual->addChild(visualNode[i].node);
                visualNode[i].mapping->init();
            }
        }
        visualNode[4].visu->setColor(1.0,0.0,0.0,1.0);
        visualNode[5].visu->setColor(0.0,1.0,0.0,1.0);
        visualNode[6].visu->setColor(0.0,0.0,1.0,1.0);

        for(int i=0; i<7; i++)
        {
            visualNode[i].node->updateContext();
        }

        for(int j=0; j<4; j++)
        {
            sofa::defaulttype::ResizableExtVector<sofa::defaulttype::Vec<3,float>> &scaleMapping = *(visualNode[j].mapping->points.beginEdit());
            for(unsigned int i=0; i<scaleMapping.size(); i++)
                for(int p=0; p<3; p++)
                    scaleMapping[i].at(p)*=(float)(Scale.getValue()/100.0);
            visualNode[j].mapping->points.endEdit();
        }

        oldScale=(float)Scale.getValue();
        changeScale=false;
    }

    visuAxes = false;
    modX=false;
    modY=false;
    modZ=false;
    modS=false;
    initVisu=true;

    std::cout<<"IHPDDriver::init() ended:"<<std::endl;
}


void IHPDriver::bwdInit()
{

    std::cout<<"IHPDriver::bwdInit() called:"<<std::endl;
    //find ff & mechanical state only for first device

    if(firstDevice)
    {
        simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node

        posTool = context->get<sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> > ();

        if(posTool==NULL)
        {
            cout<<"no meca object found"<<endl;
        }
        else
        {
            VecCoord& posT = *(posTool->x0.beginEdit());
            posT.resize(otherXitact.size());
            posTool->x0.endEdit();
            for(unsigned int i=1; i<otherXitact.size(); i++)
                otherXitact[i]->posTool=posTool;
        }

        //MechanicalStateForceFeedback<Rigid3dTypes>* ff = context->getTreeObject<MechanicalStateForceFeedback<Rigid3dTypes>>();
        LCPForceFeedback<Rigid3dTypes>* ff = context->getTreeObject<LCPForceFeedback<Rigid3dTypes>>();

        findForceFeedback = false;
        if(ff)
        {
            this->setLCPForceFeedback(ff);
            findForceFeedback = true;
            sout << "setLCPForceFeedback(ff) ok" << sendl;
        }
        else
        {
#ifdef SOFA_DEV

            VMechanismsForceFeedback<defaulttype::Vec1dTypes> *ff = context->get<VMechanismsForceFeedback<defaulttype::Vec1dTypes>>();
            if(ff)
            {
                this->setVMForceFeedback(ff);
                findForceFeedback = true;
                sout << "setVMForceFeedback(ff) ok" << sendl;
            }
            else
#endif
                std::cout << " Error: no FF found" << std::endl;
        }

        if(initDevice(data)==-1)
        {
            noDevice=true;
            std::cout<<"WARNING NO LICENCE"<<std::endl;
        }
    }

    setDataValue();

    if(!noDevice)
    {
        xiTrocarAcquire();
        char name[1024];
        char serial[16];
        int nbr = this->indexTool.getValue();
        xiTrocarGetDeviceDescription(nbr, name);
        xiTrocarGetSerialNumber(nbr,serial );
        std::cout << "Index: " << deviceIndex.getValue() << std::endl;
        std::cout << "Tool: " << nbr << std::endl;
        std::cout << "name: " << name << std::endl;
        std::cout << "serial: " << serial << std::endl;
        //xiTrocarQueryStates();
        xiTrocarGetState(nbr, &data.restState);
        xiTrocarRelease();

        data.indexTool = nbr;
    }

    std::cout<<" CREATE CALLBACK CALL"<<std::endl;

    if(firstDevice)
    {
        if (this->permanent.getValue() && findForceFeedback)
        {
            this->createCallBack();
            std::cout<<" CREATE CALLBACK OK"<<std::endl;
        }
        else
        {
            std::cout<<"no FF found or not permanent so no callback created"<<std::endl;
            this->deleteCallBack();
        }
    }


}


void IHPDriver::setDataValue()
{
    std::cout<<"IHPDriver::setDataValue() called:"<<std::endl;/////////////////////////////////////////////////////////
    data.scale = Scale.getValue();
    data.forceScale = forceScale.getValue();
    data.permanent_feedback = permanent.getValue();
}

void IHPDriver::reset()
{

}

void IHPDriver::reinitVisual()
{
    std::cout<<"IHPDriver::reinitVisual() called:"<<std::endl;/////////////////////////////////////////////////////////

}

void IHPDriver::reinit()
{
    std::cout<<"IHPDriver::reinit() called:"<<std::endl;/////////////////////////////////////////////////////////
    this->cleanup();
    this->bwdInit();
    this->reinitVisual();
}


//inutilisé
void IHPDriver::updateForce()
{
    std::cout<<"IHPDriver::updateForce() called:"<<std::endl;/////////////////////////////////////////////////////////
    // Quick FF test. Add force when using handle. Like in documentation.
    int tool = indexTool.getValue();
    float graspReferencePoint[3] = { 0.0f, 0.0f, 0.0f };
    float kForceScale = 100.0;
    XiToolForce manualForce = { 0 };

    // Checking either handle is open or not:
    if ( (data.simuState.opening <= 0.1) && (!graspElasticMode)) //Activate
    {
        graspElasticMode = true;
        for (unsigned int i = 0; i < 3; ++i)
            graspReferencePoint[i] = data.simuState.trocarDir[i] * data.simuState.toolDepth;
    }

    if ( (data.simuState.opening > 0.1) && (graspElasticMode)) //Desactivate
    {
        graspElasticMode = false;
        xiTrocarSetForce(tool, &manualForce);
        xiTrocarFlushForces();
    }

    if (graspElasticMode)
    {
        for (unsigned int i = 0; i<3; ++i)
            manualForce.tipForce[i] = (graspReferencePoint[i] - (data.simuState.trocarDir[i] * data.simuState.toolDepth)) * kForceScale;

        if (showToolStates.getValue())
        {
            char toolID[16];
            xiTrocarGetSerialNumber(tool,toolID);
            std::cout << toolID << " => Forces = " << manualForce.tipForce[0] << " | " << manualForce.tipForce[1] << " | " << manualForce.tipForce[2] << std::endl;
        }

        manualForce.rollForce = 1.0f;
        xiTrocarSetForce(tool, &manualForce);
        xiTrocarFlushForces();
    }

    std::cout<<"IHPDriver::updateForce() ended:"<<std::endl;
}


void IHPDriver::displayState()
{
    std::cout<<"IHPDriver::displayState() called:"<<std::endl;/////////////////////////////////////////////////////////
    // simple function print the current device state to the screen.
    char toolID[16];
    xiTrocarGetSerialNumber(indexTool.getValue(),toolID);
    XiToolState state = data.simuState;
    std::cout << toolID
            << " => X = " << state.trocarDir[0]
            << ", Y = "   << state.trocarDir[1]
            << ", Z = "   << state.trocarDir[2]
            << ", Ins = " << state.toolDepth
            << ", Roll(rad) = " << state.toolRoll
            << ", Open = " << state.opening << std::endl;
}

void IHPDriver::handleEvent(core::objectmodel::Event *event)
{
    //std::cout<<"IHPDriver::handleEvent() called:"<<std::endl;//////////////////////////////////////////////////////
    static double time_prev;

    if (firstDevice && dynamic_cast<sofa::simulation::AnimateEndEvent *> (event))
    {
        // force the simulation to be "real-time"
        CTime *timer;
        timer = new CTime();
        double time = 0.001*timer->getRefTime()* PaceMaker::time_scale; // in sec

        // if the computation time is shorter than the Dt set in the simulation... it waits !
        if ((time- time_prev) < getContext()->getDt() )
        {
            double wait_time = getContext()->getDt() - time + time_prev;
            timer->sleep(wait_time);
        }

        time_prev=time;
    }

    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        //turnOn xitactVisu
        if(!visuActif && xitactVisu.getValue() && initVisu)
        {
            simulation::Node *parent = dynamic_cast<simulation::Node *>(this->getContext());
            parent->addChild(nodeXitactVisual);
            nodeXitactVisual->updateContext();
            visuActif = true;
        }
        //turnOff xitactVisu
        else if(initVisu && visuActif && !xitactVisu.getValue())
        {
            simulation::Node *parent = dynamic_cast<simulation::Node *>(this->getContext());
            parent->removeChild(nodeXitactVisual);
            nodeXitactVisual->updateContext();
            visuActif=false;
        }


        // calcul des angles à partir de la direction proposée par l'interface...
        // cos(ThetaX) = cx   sin(ThetaX) = sx  cos(ThetaZ) = cz   sin(ThetaZ) = sz .
        // au repos (si cx=1 et cz=1) on a  Axe y
        // on commence par tourner autour de x   puis autour de z
        //   [cz  -sz   0] [1   0   0 ] [0]   [ -sz*cx]
        //   [sz   cz   0]*[0   cx -sx]*[1] = [ cx*cz ]
        //   [0    0    1] [0   sx  cx] [0]   [ sx    ]

        xiTrocarAcquire();
        XiToolState state;

        xiTrocarQueryStates();
        xiTrocarGetState(indexTool.getValue(), &state);

        // saving informations in class structure.
        data.simuState = state;

        Vector3 dir;

        dir[0] = -(double)state.trocarDir[0];
        dir[1] = (double)state.trocarDir[2];
        dir[2] = -(double)state.trocarDir[1];

        double pi = 3.1415926535;

        double thetaY;
        double thetaX;

        thetaY = (atan2(dir[0],-sqrt(1-dir[0]*dir[0])));
        thetaX = (pi-acos(dir[2]*sqrt(1-dir[0]*dir[0])/(dir[0]*dir[0]-1)));

        //look if thetaX and thetaY are NaN
        if(!(thetaX == thetaX))
        {
            cout<<"ratrapage X"<<endl;
            thetaX=pi;
        }
        if(!(thetaY == thetaY))
        {
            cout<<"ratrapage Y"<<endl;
            thetaY=pi;
        }

        if(dir[1]>=0)
            thetaX*=-1;

        while(thetaY<=0)
            thetaY+=2*pi;
        while(thetaX<=0)
            thetaX+=2*pi;
        while(thetaY>2*pi)
            thetaY-=2*pi;
        while(thetaX>2*pi)
            thetaX-=2*pi;



        if (showToolStates.getValue()) // print tool state
            this->displayState();

        if (testFF.getValue()) // try FF when closing handle
            this->updateForce();

        // Button and grasp handling event
        XiStateFlags stateFlag;
        stateFlag = state.flags - data.restState.flags;
        if (stateFlag == XI_ToolButtonLeft)
            this->leftButtonPushed();
        else if (stateFlag == XI_ToolButtonRight)
            this->rightButtonPushed();

        if (state.opening < graspThreshold.getValue())
        {
            this->graspClosed();
        }

        //XitactVisu
        VecCoord& posD =(*visualXitactDOF->x.beginEdit());
        posD.resize(4);
        VecCoord& posA =(*visualAxesDOF->x.beginEdit());
        posA.resize(3);

        VecCoord& posB =(*positionBase.beginEdit());
        //data.positionBaseGlobal[0]=posB[0];
        data.posBase=posB[0].getCenter();
        data.quatBase=posB[0].getOrientation();
        SolidTypes<double>::Transform tampon(posB[0].getCenter(),posB[0].getOrientation());
        positionBase.endEdit();
        posD[0].getCenter() =  tampon.getOrigin();
        posD[0].getOrientation() =  tampon.getOrientation();

        sofa::helper::Quater<double> qRotX(Vec3d(1,0,0),pi/2);
        sofa::helper::Quater<double> qRotY(Vec3d(0,0,-1),pi/2);
        SolidTypes<double>::Transform transformRotX(Vec3d(0.0,0.0,0.0),qRotX);
        SolidTypes<double>::Transform transformRotY(Vec3d(0.0,0.0,0.0),qRotY);
        SolidTypes<double>::Transform tamponAxes=tampon;
        posA[0].getCenter() =  tamponAxes.getOrigin();
        posA[0].getOrientation() =  tamponAxes.getOrientation();
        tamponAxes*=transformRotX;
        posA[1].getCenter() =  tamponAxes.getOrigin();
        posA[1].getOrientation() =  tamponAxes.getOrientation();
        tamponAxes*=transformRotY;
        posA[2].getCenter() =  tamponAxes.getOrigin();
        posA[2].getOrientation() =  tamponAxes.getOrientation();

        sofa::helper::Quater<double> qy(Vec3d(0,1,0),thetaY);
        sofa::helper::Quater<double> qx(Vec3d(1,0,0),thetaX);
        SolidTypes<double>::Transform transform2(Vec3d(0.0,0.0,0.0),qx*qy);
        tampon*=transform2;
        posD[1].getCenter() =  tampon.getOrigin();
        posD[1].getOrientation() =  tampon.getOrientation();

        sofa::helper::Quater<float> quarter3(Vec3d(0.0,0.0,1.0),-state.toolRoll);
        SolidTypes<double>::Transform transform3(Vec3d(0.0,0.0,-state.toolDepth*Scale.getValue()),quarter3);
        tampon*=transform3;
        posD[2].getCenter() =  tampon.getOrigin();
        posD[2].getOrientation() =  tampon.getOrientation();

        if(posTool)
        {
            VecCoord& posT = *(posTool->x0.beginEdit());
            //cout<<"xitact "<<deviceIndex.getValue()<<" "<<posD[2]<<endl;
            posT[deviceIndex.getValue()]=posD[2];
            posTool->x0.endEdit();
        }

        sofa::helper::Quater<float> quarter4(Vec3d(0.0,1.0,0.0),-data.simuState.opening/(float)2.0);
        SolidTypes<double>::Transform transform4(Vec3d(0.0,0.0,0.44*Scale.getValue()),quarter4);
        tampon*=transform4;
        posD[3].getCenter() =  tampon.getOrigin();
        posD[3].getOrientation() =  tampon.getOrientation();
        visualXitactDOF->x.endEdit();

        Vec1d& openT = (*openTool.beginEdit());
        openT[0]=(data.simuState.opening)*(maxTool.getValue()-minTool.getValue())+minTool.getValue();
        openTool.endEdit();

        if(changeScale)
        {
            float rapport=((float)Scale.getValue())/oldScale;
            for(int j = 0; j<4 ; j++)
            {
                sofa::defaulttype::ResizableExtVector<sofa::defaulttype::Vec<3,float>> &scaleMapping = *(visualNode[j].mapping->points.beginEdit());
                for(unsigned int i=0; i<scaleMapping.size(); i++)
                {
                    for(int p=0; p<3; p++)
                        scaleMapping[i].at(p)*=rapport;
                }
                visualNode[j].mapping->points.endEdit();
            }
            oldScale=(float)Scale.getValue();
            changeScale=false;
        }


    }
    if (dynamic_cast<core::objectmodel::KeypressedEvent *>(event))
    {
        core::objectmodel::KeypressedEvent *kpe = dynamic_cast<core::objectmodel::KeypressedEvent *>(event);
        onKeyPressedEvent(kpe);
    }
    else if (dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event))
    {
        core::objectmodel::KeyreleasedEvent *kre = dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event);
        onKeyReleasedEvent(kre);
    }


    //std::cout<<"IHPDriver::handleEvent() ended:"<<std::endl;
}

void IHPDriver::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kpe)
{
    cout<<"kpe"<<endl;
    cout<<initVisu<<endl;
    cout<<int(kpe->getKey())<<" "<<kpe->getKey()<<endl;
    if(!visuAxes && kpe->getKey()==49+deviceIndex.getValue() && initVisu)
    {
        cout<<"axes on"<<endl;
        simulation::Node *parent = dynamic_cast<simulation::Node *>(this->getContext());
        parent->addChild(nodeAxesVisual);
        nodeAxesVisual->updateContext();
        visuAxes=true;
    }
    else if(visuAxes && kpe->getKey()==49+deviceIndex.getValue() && initVisu)
    {
        cout<<"axes off"<<endl;
        simulation::Node *parent = dynamic_cast<simulation::Node *>(this->getContext());
        parent->removeChild(nodeAxesVisual);
        nodeAxesVisual->updateContext();
        visuAxes=false;
    }

    if(visuAxes  && xitactVisu.getValue())
    {
        double pi = 3.1415926535;
        if ((kpe->getKey()=='X' || kpe->getKey()=='x') && !modX )
        {
            modX=true;
        }
        if ((kpe->getKey()=='Y' || kpe->getKey()=='y') && !modY )
        {
            modY=true;
        }
        if ((kpe->getKey()=='Z' || kpe->getKey()=='z') && !modZ )
        {
            modZ=true;
        }
        if ((kpe->getKey()=='Q' || kpe->getKey()=='q') && !modS )
        {
            modS=true;
        }
        if (kpe->getKey()==18) //left
        {
            if(modX || modY || modZ)
            {
                VecCoord& posB =(*positionBase.beginEdit());
                posB[0].getCenter()+=posB[0].getOrientation().rotate(Vec3d(-5*(int)modX,-5*(int)modY,-5*(int)modZ));
                positionBase.endEdit();
            }
            else if(modS)
            {
                Scale.setValue(Scale.getValue()-5);
                changeScale = true;
            }
        }
        else if (kpe->getKey()==20) //right
        {

            if(modX || modY || modZ)
            {
                VecCoord& posB =(*positionBase.beginEdit());
                posB[0].getCenter()+=posB[0].getOrientation().rotate(Vec3d(5*(int)modX,5*(int)modY,5*(int)modZ));
                positionBase.endEdit();
            }
            else if(modS)
            {
                Scale.setValue(Scale.getValue()+5);
                changeScale = true;
            }
        }
        else if ((kpe->getKey()==21) && (modX || modY || modZ)) //down
        {
            VecCoord& posB =(*positionBase.beginEdit());
            sofa::helper::Quater<double> quarter_transform(Vec3d((int)modX,(int)modY,(int)modZ),-pi/50);
            posB[0].getOrientation()*=quarter_transform;
            positionBase.endEdit();
        }
        else if ((kpe->getKey()==19) && (modX || modY || modZ)) //up
        {
            VecCoord& posB =(*positionBase.beginEdit());
            sofa::helper::Quater<double> quarter_transform(Vec3d((int)modX,(int)modY,(int)modZ),+pi/50);
            posB[0].getOrientation()*=quarter_transform;
            positionBase.endEdit();
        }
        if ((kpe->getKey()=='E' || kpe->getKey()=='e'))
        {
            VecCoord& posB =(*positionBase.beginEdit());
            posB[0].clear();
            positionBase.endEdit();
        }

    }

}

void IHPDriver::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *kre)
{
//std::cout<<"IHPDriver::onKeyReleasedEvent() called:"<<std::endl;/////////////////////////////////////////////////////////
    if (kre->getKey()=='X' || kre->getKey()=='x' )
    {
        modX=false;
    }
    if (kre->getKey()=='Y' || kre->getKey()=='y' )
    {
        modY=false;
    }
    if (kre->getKey()=='Z' || kre->getKey()=='z' )
    {
        modZ=false;
    }
    if (kre->getKey()=='Q' || kre->getKey()=='q' )
    {
        modS=false;
    }
}






//inutilisé
//Quat IHPDriver::fromGivenDirection( Vector3& dir,  Vector3& local_dir, Quat old_quat)
//{
//std::cout<<"IHPDriver::fromGivenDirection() called:"<<std::endl;/////////////////////////////////////////////////////////
//      local_dir.normalize();
//      Vector3 old_dir = old_quat.rotate(local_dir);
//      dir.normalize();
//
//      if (dot(dir, old_dir)<1.0)
//      {
//            Vector3 z = cross(old_dir, dir);
//            z.normalize();
//            double alpha = acos(dot(old_dir, dir));
//
//            Quat dq, Quater_result;
//
//			dq.axisToQuat(z, alpha);
//
//            Quater_result =  old_quat+dq;
//
//            //std::cout<<"debug - verify fromGivenDirection  dir = "<<dir<<"  Quater_result.rotate(local_dir) = "<<Quater_result.rotate(local_dir)<<std::endl;
//
//            return Quater_result;
//      }
//
//      return old_quat;
//}


void IHPDriver::createCallBack()
{

    cout<<"IHPDriver::createCallBack() called:"<<endl;/////////////////////////////////////////////////////////
    if (myPaceMaker)
        delete myPaceMaker;
    myPaceMaker=NULL;

    myPaceMaker = new sofa::component::controller::PaceMaker(RefreshFrequency.getValue());
    myPaceMaker->pToFunc =  &UpdateForceFeedBack;
    myPaceMaker->Pdata = &allData;
    myPaceMaker->createPace();


    //This function create a thread calling stateCallBack() at a given frequence
    std::cout<<"IHPDriver::createCallBack() ok:"<<std::endl;/////////////////////////////////////////////////////////
}


void IHPDriver::deleteCallBack()
{
    std::cerr<<"IHPDriver::deleteCallBack() called:"<<std::endl;/////////////////////////////////////////////////////////
    if (myPaceMaker)
    {
        delete myPaceMaker;
        myPaceMaker = NULL;
    }
}


void IHPDriver::stateCallBack()
{
    std::cout<<"IHPDriver::stateCallBack() called:"<<std::endl;/////////////////////////////////////////////////////////
    // this function delete thread
}

void IHPDriver::rightButtonPushed()
{
    std::cout<<"IHPDriver::rightButtonPushed() called:"<<std::endl;/////////////////////////////////////////////////////////
    this->operation = true;
}

void IHPDriver::leftButtonPushed()
{
    std::cout<<"IHPDriver::leftButtonPushed() called:"<<std::endl;/////////////////////////////////////////////////////////
    this->operation = false;
}

void IHPDriver::graspClosed()
{
//std::cout<<"IHPDriver::graspClosed() called:"<<std::endl;/////////////////////////////////////////////////////////
    if (operation)//Right pedal operation
    {
        return;
    }
    else //Left pedal operation
        return;
}


} // namespace controller

} // namespace component

} // namespace sofa
