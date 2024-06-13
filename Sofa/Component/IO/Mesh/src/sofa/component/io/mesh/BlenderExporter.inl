/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/io/mesh/BlenderExporter.h>
#include <iomanip>
#include <iostream>
#include <sofa/helper/system/FileSystem.h>


namespace sofa::component::_blenderexporter_
{

using namespace std;

template<class T>
BlenderExporter<T>::BlenderExporter()
    : d_path(initData(&d_path, "path", "output path")),
      d_baseName(initData(&d_baseName, "baseName", "Base name for the output files")),
      d_simulationType(initData(&d_simulationType, 0, "simulationType", "simulation type (0: soft body, 1: particles, 2:cloth, 3:hair)")),
      d_simulationStep(initData(&d_simulationStep, 2, "step", "save the  simulation result every step frames")),
      d_nbPtsByHair(initData(&d_nbPtsByHair, 20, "nbPtsByHair", "number of element by hair strand")),
      frameCounter(0)
{
    Inherit::f_listening.setValue(true);

    path.setParent(&d_path);
    baseName.setParent(&d_baseName);
    simulationType.setParent(&d_simulationType);
    simulationStep.setParent(&d_simulationStep);
    nbPtsByHair.setParent(&d_nbPtsByHair);

}

template<class T>
void BlenderExporter<T>::init()
{
    mmodel = getContext()->template get<DataType>(sofa::core::objectmodel::BaseContext::SearchDirection::Local);
    if(mmodel == nullptr)
        msg_error()<<"Initialization failed!";
    Inherit::init();
    // if hair type simulation, create an additional information frame
    if(d_simulationType.getValue() == Hair)
    {
        ostringstream iss;
        iss << helper::system::FileSystem::append(d_path.getValue(), d_baseName.getValue())
            << "_000000_00.bphys";
        string fileName = iss.str();
        // create the file
        ofstream file(fileName.c_str(), ios::out|ios::binary);
        if(file)
        {
            msg_info()<<"writing in "<<fileName;
            const string bphysics = "BPHYSICS";
            file.write(bphysics.c_str(),8);
            unsigned tmp[3]={1,1,64};
            file.write((char*)&tmp,12);
            float tmp2[3]={0,100,100};
            file.write((char*)&tmp2,12);
            file.close();
        }
    }
}

template<class T>
void BlenderExporter<T>::reset()
{
    frameCounter=0;
}

template<class T>
void BlenderExporter<T>::handleEvent(sofa::core::objectmodel::Event* event)
{

    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        if(!(frameCounter % d_simulationStep.getValue())) // save a new frame!
        {
            int frameNumber = frameCounter / d_simulationStep.getValue();
            ostringstream iss;
            iss << helper::system::FileSystem::append(d_path.getValue(), d_baseName.getValue()) << "_";
            iss<<std::setfill('0') << std::setw(6) << frameNumber+1<<"_00.bphys";
            string fileName = iss.str();

            // create the file
            ofstream file(fileName.c_str(), ios::out|ios::binary);

            if(file)
            {
                msg_info()<<"writing in "<<fileName;

                // ############# write header

                const string bphysics = "BPHYSICS";
                file.write(bphysics.c_str(),8);

                // types
                unsigned type;
                if(d_simulationType.getValue() == Hair)
                    type = Cloth; // blender hair exception
                else
                    type = d_simulationType.getValue();
                file.write((char*)&type,4);

                // number of data
                auto size = mmodel->getSize();
                if(d_simulationType.getValue() == Hair)
                {
                    unsigned sizeHair = size+size/(d_nbPtsByHair.getValue());
                    file.write((char*)&sizeHair,4);
                }
                else
                    file.write((char*)&size,4);

                // dataType
                unsigned dataType;
                switch (d_simulationType.getValue())
                {
                case SoftBody: dataType = 6;
                    break;
                case Hair: dataType = 22;
                    break;
                default:  dataType = 6;
                    break;
                }
                file.write((char*)&dataType,4);

                // ############# write data

                float pos[3]={0,0,0};
                float vel[3]={0,0,0};
                float rest[3]={0,0,0};
                float pos0[3]={0,0,0};
                float vel0[3]={0,0,0};


                ReadVecCoord posData = mmodel->readPositions();


                for(int i= (int)size-1; i>=0; i--)
                {
                    //create an additional point for root tangent
                    if((d_simulationType.getValue() == Hair && (i % d_nbPtsByHair.getValue() == 0)))
                    {

                       auto  x0 = T::getCPos(posData[i]);
                       auto  x1 = T::getCPos(posData[i+1]);

                        x1 = x1-x0;
                        x1.normalize();

                        x0 = x0+x1;

                        pos0[0] = (float)x0[0];
                        pos0[1] = (float)x0[1];
                        pos0[2] = (float)x0[2];

                        file.write((char*)pos0,12);
                        file.write((char*)vel0,12);
                        file.write((char*)pos0,12);
                    }

                    //Coord x0=restData[i];
                    Coord x=posData[i];
                    pos[0] = (float)x[0];
                    pos[1] = (float)x[1];
                    pos[2] = (float)x[2];

                    Deriv v;
                    if((mmodel->read(core::ConstVecDerivId::velocity())) && ((int) mmodel->readVelocities().size()>i))
                    {
                        v =mmodel->readVelocities()[i];
                        vel[0] = (float)v[0];
                        vel[1] = (float)v[1];
                        vel[2] = (float)v[2];
                    }

                    Coord x0;
                    if((mmodel->read(core::ConstVecCoordId::restPosition())) && ( (int)mmodel->readRestPositions().size()>i))
                    {
                        x0 =mmodel->readRestPositions()[i];
                        rest[0] = (float)x0[0];
                        rest[1] = (float)x0[1];
                        rest[2] = (float)x0[2];
                    }



                    switch (d_simulationType.getValue())
                    {
                    case SoftBody:
                        file.write((char*)pos,12);
                        file.write((char*)vel,12);
                        break;
                    case Hair:
                        file.write((char*)pos,12);
                        file.write((char*)vel,12);
                        file.write((char*)rest,12);
                        break;
                    default:
                        file.write((char*)pos,12);
                        file.write((char*)vel,12);
                        break;
                    }
                }
                file.close();
            }
            else
                msg_error() << "Unable to create the following file: "<<fileName;

        }
        frameCounter++;
    }

}


} // namespace sofa::component::_blenderexporter_
