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

#include "HaptionDriver.h"


namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;
using namespace std;

void HaptionDriver::haptic_callback(VirtContext, void *param)
{
    HaptionData* data = static_cast<HaptionData*>(param);
    float position[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    virtGetAvatarPosition(data->m_virtContext, position);

    SolidTypes<double>::Transform sofaWorld_H_Tool(
        Vec3d(data->scale*position[0],data->scale*position[1],data->scale*position[2]),
        Quat(position[3],position[4],position[5],position[6]));

    SolidTypes<double>::SpatialVector Twist_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0));
    SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0));

    if(data->forceFeedback != NULL)
    {
        (data->forceFeedback)->computeWrench(sofaWorld_H_Tool, Twist_tool_inWorld, Wrench_tool_inWorld);

        float force[6] = {(float) Wrench_tool_inWorld.getForce()[0]*data->forceScale,
                (float) Wrench_tool_inWorld.getForce()[1]*data->forceScale,
                (float) Wrench_tool_inWorld.getForce()[2]*data->forceScale,
                (float) Wrench_tool_inWorld.getTorque()[0]*data->torqueScale,
                (float) Wrench_tool_inWorld.getTorque()[1]*data->torqueScale,
                (float) Wrench_tool_inWorld.getTorque()[2]*data->torqueScale
                         };

        for(int i=0; i<3; i++)
        {
            if(force[i]>15.0f)
            {
                cout<<"saturation F+ "<<i<<" "<<force[i]<<endl;
                force[i]=15.0f;
            }
            if(force[i]<-15.0f)
            {
                cout<<"saturation F- "<<i<<" "<<force[i]<<endl;
                force[i]=-15.0f;
            }
            if(force[i+3]>1.0f)
            {
                cout<<"saturation C+ "<<i<<" "<<force[i+3]<<endl;
                force[i+3]=1.0f;
            }
            if(force[i+3]<-1.0f)
            {
                cout<<"saturation C- "<<i<<" "<<force[i+3]<<endl;
                force[i+3]=-1.0f;
            }
        }

        virtSetForce(data->m_virtContext,force);


        //if( force[0]>0.01 || force[1]>0.01 || force[2]>0.01 || force[3]>0.01 || force[4]>0.01 || force[5]>0.01)
        //	cout<<Wrench_tool_inWorld.getForce()[0]<<" "<<
        //	  Wrench_tool_inWorld.getForce()[1]<<" "<<
        //	  Wrench_tool_inWorld.getForce()[2]<<" "<<
        //	  Wrench_tool_inWorld.getTorque()[0]<<" "<<
        //	  Wrench_tool_inWorld.getTorque()[1]<<" "<<
        //	  Wrench_tool_inWorld.getTorque()[2]<<endl;
    }
}


//constructeur
HaptionDriver::HaptionDriver()
    :scale(initData(&scale, 100.0, "Scale","Default scale applied to the Haption Coordinates. ")),
     state_button(initData(&state_button, false, "state_button","state of the first button")),
     haptionVisu(initData(&haptionVisu, false, "haptionVisu","Visualize the position of the interface in the virtual scene")),
     posBase(initData(&posBase, "positionBase","Position of the interface base in the scene world coordinates")),
     torqueScale(initData(&torqueScale, 0.5, "torqueScale","Default scale applied to the Haption torque. ")),
     forceScale(initData(&forceScale, 1.0, "forceScale","Default scale applied to the Haption force. ")),
     ip_haption(initData(&ip_haption,std::string("localhost"),"ip_haption","ip of the device")),
     m_speedFactor(1.0),
     m_forceFactor(1.0),
     haptic_time_step(0.003f),
     connection_device(0),
     initCallback(false),
     nodeHaptionVisual(NULL),
     visualHaptionDOF(NULL),
     nodeAxesVisual(NULL),
     visualAxesDOF(NULL),
     oldScale(0),
     changeScale(0),
     visuAxes(false),
     modX(false),
     modY(false),
     modZ(false),
     modS(false),
     visuActif(false)
{
    rigidDOF=NULL;
}

//destructeur
HaptionDriver::~HaptionDriver()
{
    closeDevice();
}


void HaptionDriver::init()
{
    cout << "HaptionDriver::init()" << endl;

    VecCoord& posB = (*posBase.beginEdit());
    posB.resize(1);
    posB[0].getOrientation().normalize();
    posBase.endEdit();

    char *ip_char = new char[ip_haption.getValue().size() + 1];
    std::copy(ip_haption.getValue().begin(), ip_haption.getValue().end(), ip_char);
    ip_char[ip_haption.getValue().size()] = '\0';
    connection_device = initDevice(ip_char);
    delete[] ip_char;
    myData.forceFeedback = NULL;


    if(visualHaptionDOF == NULL && visualAxesDOF == NULL)
    {
        cout<<"init Visual"<<endl;
        simulation::Node *context;
        context = dynamic_cast<simulation::Node*>(this->getContext());

        //Haption node
        nodeHaptionVisual = sofa::simulation::getSimulation()->createNewGraph("nodeHaptionVisual");
        if(haptionVisu.getValue())
        {
            sofa::simulation::tree::GNode *parent = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
            parent->getParent()->addChild(nodeHaptionVisual);
            nodeHaptionVisual->updateContext();
            visuActif=true;
        }

        visualHaptionDOF = new sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>();
        nodeHaptionVisual->addObject(visualHaptionDOF);
        visualHaptionDOF->name.setValue("rigidDOF");

        VecCoord& posH =*(visualHaptionDOF->x.beginEdit());
        posH.resize(2);
        posH[0]=posBase.getValue()[0];
        visualHaptionDOF->x.endEdit();

        visualHaptionDOF->init();
        nodeHaptionVisual->updateContext();

        //Axes node
        nodeAxesVisual = sofa::simulation::getSimulation()->createNewGraph("nodeAxesVisual");
        //context->addChild(nodeAxesVisual);
        //nodeAxesVisual->updateContext();

        visualAxesDOF = new sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes>();
        nodeAxesVisual->addObject(visualAxesDOF);
        visualAxesDOF->name.setValue("rigidDOF");

        VecCoord& posA =*(visualAxesDOF->x.beginEdit());
        posA.resize(3);
        posA[0]=posBase.getValue()[0];
        posA[1]=posBase.getValue()[0];
        posA[2]=posBase.getValue()[0];
        visualAxesDOF->x.endEdit();

        visualAxesDOF->init();
        nodeAxesVisual->updateContext();

        visualNode[0].node = sofa::simulation::getSimulation()->createNewGraph("base");
        visualNode[1].node = sofa::simulation::getSimulation()->createNewGraph("avatar");
        visualNode[2].node = sofa::simulation::getSimulation()->createNewGraph("axe X");
        visualNode[3].node = sofa::simulation::getSimulation()->createNewGraph("axe Y");
        visualNode[4].node = sofa::simulation::getSimulation()->createNewGraph("axe Z");

        for(int i=0; i<5; i++)
        {
            visualNode[i].visu = NULL;
            visualNode[i].mapping = NULL;
            if(visualNode[i].visu == NULL && visualNode[i].mapping == NULL)
            {
                visualNode[i].visu = new sofa::component::visualmodel::OglModel();
                visualNode[i].node->addObject(visualNode[i].visu);
                visualNode[i].visu->name.setValue("VisualParticles");
                if(i==0)
                    visualNode[i].visu->fileMesh.setValue("mesh/virtuose_base.obj");
                if(i==1)
                    visualNode[i].visu->fileMesh.setValue("mesh/virtuose_avatar.obj");
                if(i==2)
                {
                    visualNode[i].visu->fileMesh.setValue("mesh/axeXH.obj");
                    visualNode[i].visu->setScale(5.0,5.0,5.0);
                }
                if(i==3)
                {
                    visualNode[i].visu->fileMesh.setValue("mesh/axeYH.obj");
                    visualNode[i].visu->setScale(5.0,5.0,5.0);
                }
                if(i==4)
                {
                    visualNode[i].visu->fileMesh.setValue("mesh/axeZH.obj");
                    visualNode[i].visu->setScale(5.0,5.0,5.0);
                }
                visualNode[i].visu->init();
                visualNode[i].visu->initVisual();
                visualNode[i].visu->updateVisual();
                if(i<2)
                    visualNode[i].mapping = new sofa::component::mapping::RigidMapping< Rigid3dTypes, ExtVec3fTypes >(visualHaptionDOF,visualNode[i].visu);
                else
                    visualNode[i].mapping = new sofa::component::mapping::RigidMapping< Rigid3dTypes, ExtVec3fTypes >(visualAxesDOF,visualNode[i].visu);
                visualNode[i].node->addObject(visualNode[i].mapping);
                visualNode[i].mapping->name.setValue("RigidMapping");
                visualNode[i].mapping->f_mapConstraints.setValue(false);
                visualNode[i].mapping->f_mapForces.setValue(false);
                visualNode[i].mapping->f_mapMasses.setValue(false);
                visualNode[i].mapping->m_inputObject.setValue("@../rigidDOF");
                visualNode[i].mapping->m_outputObject.setValue("@VisualParticles");
                if(i<2)
                    visualNode[i].mapping->index.setValue(i);
                else
                    visualNode[i].mapping->index.setValue(i-2);
                visualNode[i].mapping->init();
                if(i<2)
                    nodeHaptionVisual->addChild(visualNode[i].node);
                else
                    nodeAxesVisual->addChild(visualNode[i].node);
            }
        }

        visualNode[2].visu->setColor(1.0,0.0,0.0,1.0);
        visualNode[3].visu->setColor(0.0,1.0,0.0,1.0);
        visualNode[4].visu->setColor(0.0,0.0,1.0,1.0);

        for(int j=0; j<5; j++)
        {
            visualNode[j].node->updateContext();
        }

        for(int j=0; j<2; j++)
        {
            sofa::defaulttype::ResizableExtVector<sofa::defaulttype::Vec<3,float>> &scaleMapping = *(visualNode[j].mapping->points.beginEdit());
            for(unsigned int i=0; i<scaleMapping.size(); i++)
                for(int p=0; p<3; p++)
                    scaleMapping[i].at(p)*=(float)(scale.getValue());
            visualNode[j].mapping->points.endEdit();
        }

        oldScale=(float)scale.getValue();
        changeScale=false;
    }




}


void HaptionDriver::bwdInit()
{
    std::cout<<"HaptionDriver::bwdInit() is called"<<std::endl;

    simulation::Node *context = dynamic_cast<simulation::Node*>(this->getContext());

    rigidDOF = context->get<sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> > ();


    if (rigidDOF==NULL)
    {
        std::cout<<" no Meca Object found"<<std::endl;
    }

    //search force feedback
    MechanicalStateForceFeedback<Rigid3dTypes>* ff = context->getTreeObject<MechanicalStateForceFeedback<Rigid3dTypes>>();

    if(ff)
    {
        setForceFeedback(ff);
        cout<<"force feedback found"<<endl;
    }
}

void HaptionDriver::setForceFeedback(MechanicalStateForceFeedback<Rigid3dTypes>* ff)
{
    if(myData.forceFeedback == ff)
    {
        return;
    }
    if(myData.forceFeedback)
        delete myData.forceFeedback;
    myData.forceFeedback=NULL;
    myData.forceFeedback=ff;
}

void HaptionDriver::reset()
{
    std::cout<<"HaptionDriver::reset() is called" <<std::endl;
    this->reinit();


}

void HaptionDriver::reinit()
{
    std::cout<<"HaptionDriver::reinit() is called" <<std::endl;
    myData.scale = (float) scale.getValue();
    myData.torqueScale = (float) torqueScale.getValue();
    myData.forceScale = (float) forceScale.getValue();
}

int HaptionDriver::initDevice(char* ip)
{
    cout<<"HaptionDriver::initDevice() called"<<endl;

    connection_device = 0;
    /*m_indexingMode = INDEXING_ALL_FORCE_FEEDBACK_INHIBITION;*/
    m_indexingMode = INDEXING_ALL;
    m_speedFactor = 1.0;
    haptic_time_step = 0.003f;
    //haptic_time_step = 0.5f;
    myData.m_virtContext = NULL;
    cout<<"tentative de connection sur: "<<ip<<endl;
    myData.m_virtContext = virtOpen (ip);
    if (myData.m_virtContext == NULL)
    {
        cout<<"erreur connection"<<endl;
        return 0;
    }
    else
        cout<<"connection OK"<<endl;

    virtSetIndexingMode(myData.m_virtContext, m_indexingMode);
    cout<<"virtSetSpeedFactor return "<<virtSetSpeedFactor(myData.m_virtContext, m_speedFactor)<<endl;
    float speddFactor[1];
    virtGetSpeedFactor(myData.m_virtContext, speddFactor);
    cout<<"virtGetSpeedFactor return "<<speddFactor[0]<<endl;
    virtSetTimeStep(myData.m_virtContext,haptic_time_step);

    cout<<"set base frame ok"<<endl;

    m_typeCommand = COMMAND_TYPE_IMPEDANCE;
    m_forceFactor = 1.0f;

    virtSetCommandType(myData.m_virtContext, m_typeCommand);
    virtSetForceFactor(myData.m_virtContext, m_forceFactor);

    virtSetPowerOn(myData.m_virtContext, 1);
    cout<<"init callback"<<endl;
    virtSetPeriodicFunction(myData.m_virtContext, haptic_callback, &haptic_time_step, &myData);
    cout<<"callback initialise"<<endl;

    virtSaturateTorque(myData.m_virtContext, 15.0f,0.7f);

    cout<<posBase.getValue()[0].getCenter()<<" "<<posBase.getValue()[0].getOrientation()<<endl;
    float baseFrame[7] = { (float) posBase.getValue()[0].getCenter().x()/(float) scale.getValue(),
            (float) posBase.getValue()[0].getCenter().y()/(float) scale.getValue(),
            (float) posBase.getValue()[0].getCenter().z()/(float) scale.getValue(),
            (float) posBase.getValue()[0].getOrientation()[0],
            (float) posBase.getValue()[0].getOrientation()[1],
            (float) posBase.getValue()[0].getOrientation()[2],
            (float) posBase.getValue()[0].getOrientation()[3]
                         };


    cout<<"virtSetBaseFrame return "<<virtSetBaseFrame(myData.m_virtContext, baseFrame)<<endl;
    cout<<"virtGetErrorCode return "<<virtGetErrorCode(myData.m_virtContext)<<endl;

    return 1;
}

void HaptionDriver::closeDevice()
{
    if (myData.m_virtContext)
    {
        virtSetPowerOn(myData.m_virtContext, 0);
        //virtDetachVO(myData.m_virtContext);
        virtClose(myData.m_virtContext);
        myData.m_virtContext = 0;
    }
}

void HaptionDriver::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kpe)
{
    if(!visuAxes && kpe->getKey()==49)
    {
        sofa::simulation::tree::GNode *parent = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
        parent->getParent()->addChild(nodeAxesVisual);
        nodeAxesVisual->updateContext();
        visuAxes=true;
    }
    else if(visuAxes && kpe->getKey()==49)
    {
        sofa::simulation::tree::GNode *parent = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
        parent->getParent()->removeChild(nodeAxesVisual);
        nodeAxesVisual->updateContext();
        visuAxes=false;
    }

    if(visuAxes  && haptionVisu.getValue())
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
                VecCoord& posB =(*posBase.beginEdit());
                posB[0].getCenter()+=posB[0].getOrientation().rotate(Vec3d(-(int)modX,-(int)modY,-(int)modZ));
                posBase.endEdit();
            }
            else
            {
                scale.setValue(scale.getValue()-5);
                changeScale = true;
            }
        }
        else if (kpe->getKey()==20) //right
        {

            if(modX || modY || modZ)
            {
                VecCoord& posB =(*posBase.beginEdit());
                posB[0].getCenter()+=posB[0].getOrientation().rotate(Vec3d((int)modX,(int)modY,(int)modZ));
                posBase.endEdit();
            }
            else
            {
                scale.setValue(scale.getValue()+5);
                changeScale = true;
            }
        }
        else if ((kpe->getKey()==21) && (modX || modY || modZ)) //down
        {
            VecCoord& posB =(*posBase.beginEdit());
            sofa::helper::Quater<double> quarter_transform(Vec3d((int)modX,(int)modY,(int)modZ),-pi/50);
            posB[0].getOrientation()*=quarter_transform;
            posBase.endEdit();
        }
        else if ((kpe->getKey()==19) && (modX || modY || modZ)) //up
        {
            VecCoord& posB =(*posBase.beginEdit());
            sofa::helper::Quater<double> quarter_transform(Vec3d((int)modX,(int)modY,(int)modZ),+pi/50);
            posB[0].getOrientation()*=quarter_transform;
            posBase.endEdit();
        }
        if ((kpe->getKey()=='E' || kpe->getKey()=='e'))
        {
            VecCoord& posB =(*posBase.beginEdit());
            posB[0].clear();
            posBase.endEdit();
        }

        if(modX || modY || modZ)
        {
            float baseFrame[7] = { (float) posBase.getValue()[0].getCenter()[0]/(float) scale.getValue(),
                    (float) posBase.getValue()[0].getCenter()[1]/(float) scale.getValue(),
                    (float) posBase.getValue()[0].getCenter()[2]/(float) scale.getValue(),
                    (float) posBase.getValue()[0].getOrientation()[0],
                    (float) posBase.getValue()[0].getOrientation()[1],
                    (float) posBase.getValue()[0].getOrientation()[2],
                    (float) posBase.getValue()[0].getOrientation()[3]
                                 };


            cout<<"virtSetBaseFrame return "<<virtSetBaseFrame(myData.m_virtContext, baseFrame)<<endl;
            cout<<"virtGetErrorCode return "<<virtGetErrorCode(myData.m_virtContext)<<endl;

            VecCoord& posH =*(visualHaptionDOF->x.beginEdit());
            posH[0]=posBase.getValue()[0];
            visualHaptionDOF->x.endEdit();

            VecCoord& posA =*(visualAxesDOF->x.beginEdit());
            posA[0]=posBase.getValue()[0];
            posA[1]=posBase.getValue()[0];
            posA[2]=posBase.getValue()[0];
            visualAxesDOF->x.endEdit();
        }
    }

}

void HaptionDriver::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *kre)
{
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

void HaptionDriver::onAnimateBeginEvent()
{
    if(connection_device)
    {
        if(!initCallback && myData.forceFeedback)
        {
            initCallback = true;
            virtStartLoop(myData.m_virtContext);
        }
        float	position[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
        virtGetAvatarPosition(myData.m_virtContext, position);

        VecCoord& posR = (*rigidDOF->x0.beginEdit());
        posR[0].getCenter() = Vec3d(scale.getValue()*position[0],scale.getValue()*position[1],scale.getValue()*position[2]);
        posR[0].getOrientation() = Quat(position[3],position[4],position[5],position[6]);
        rigidDOF->x0.endEdit();

        //button_state
        int buttonState[1];
        virtGetButton(myData.m_virtContext,1,buttonState);
        state_button.setValue(buttonState[0]);

        //visu
        if(visuActif)
        {
            VecCoord& posH = (*visualHaptionDOF->x.beginEdit());
            posH[1].getCenter() = Vec3d(scale.getValue()*position[0],scale.getValue()*position[1],scale.getValue()*position[2]);
            posH[1].getOrientation() = Quat(position[3],position[4],position[5],position[6]);
            visualHaptionDOF->x.endEdit();
        }

        if(haptionVisu.getValue() && !visuActif)
        {
            cout<<"add visu called"<<endl;
            sofa::simulation::tree::GNode *parent = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
            parent->getParent()->addChild(nodeHaptionVisual);
            nodeHaptionVisual->updateContext();
            visuActif=true;
            cout<<"add visu ok"<<endl;
        }

        if(!haptionVisu.getValue() && visuActif)
        {
            cout<<"remove visu called"<<endl;
            sofa::simulation::tree::GNode *parent = dynamic_cast<sofa::simulation::tree::GNode*>(this->getContext());
            parent->getParent()->removeChild(nodeHaptionVisual);
            nodeHaptionVisual->updateContext();
            visuActif=false;
            cout<<"remove visu ok"<<endl;
        }
    }
}

void HaptionDriver::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        onAnimateBeginEvent();
    }
    else if (dynamic_cast<core::objectmodel::KeypressedEvent *>(event))
    {
        core::objectmodel::KeypressedEvent *kpe = dynamic_cast<core::objectmodel::KeypressedEvent *>(event);
        onKeyPressedEvent(kpe);
    }
    else if (dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event))
    {
        core::objectmodel::KeyreleasedEvent *kre = dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event);
        onKeyReleasedEvent(kre);
    }
}

int HaptionDriverClass = core::RegisterObject("Solver to test compliance computation for new articulated system objects")
        .add< HaptionDriver >();

SOFA_DECL_CLASS(HaptionDriver)

} // namespace controller

} // namespace component

} // namespace sofa
