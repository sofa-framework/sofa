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
#include <SofaExporter/BlenderExporter.h>
#include <iomanip>
#include <iostream>

namespace sofa
{

    namespace component
    {

        namespace misc
        {

            using namespace std;

            template<class T>
            BlenderExporter<T>::BlenderExporter()
                : path(initData(&path,"path","output path")),
                baseName(initData(&baseName, "baseName", "Base name for the output files")),
                simulationType(initData(&simulationType,0, "simulationType", "simulation type (0: soft body, 1: particles, 2:cloth, 3:hair)")),
                simulationStep(initData(&simulationStep,2, "step", "save the  simulation result every step frames")),
                nbPtsByHair(initData(&nbPtsByHair,20, "nbPtsByHair", "number of element by hair strand")),
                frameCounter(0)
            {
                Inherit::f_listening.setValue(true);
            }

            template<class T>
            void BlenderExporter<T>::init()
            {
                mmodel = Inherit::searchLocal<DataType>();
                if(mmodel == NULL)
                    serr<<"Initialization failed!"<<sendl;
                Inherit::init();
                // if hair type simulation, create an additional information frame 
                if(simulationType.getValue()==Hair)
                {
                    ostringstream iss;
                    iss<<path.getValue()<<"/"<<baseName.getValue()<<"_000000_00.bphys";
                    string fileName = iss.str();
                    // create the file
                    ofstream file(fileName.c_str(), ios::out|ios::binary);
                    if(file)
                    {
                        sout<<"writing in "<<fileName<<sendl;
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
                    if(!(frameCounter%simulationStep.getValue())) // save a new frame!
                    {
                        int frameNumber = frameCounter/simulationStep.getValue();
                        ostringstream iss;
                        iss<<path.getValue()<<"/"<<baseName.getValue()<<"_";
                        iss<<std::setfill('0') << std::setw(6) << frameNumber+1<<"_00.bphys";
                        string fileName = iss.str();

                        // create the file
                        ofstream file(fileName.c_str(), ios::out|ios::binary);

                        if(file)
                        {
                            sout<<"writing in "<<fileName<<sendl;

                            // ############# write header

                            const string bphysics = "BPHYSICS";
                            file.write(bphysics.c_str(),8);

                            // types
                            unsigned type;
                            if(simulationType.getValue()==Hair)
                                type = Cloth; // blender hair exception
                            else
                                type = simulationType.getValue();
                            file.write((char*)&type,4);

                            // number of data
                            unsigned size = mmodel->getSize();
                            if(simulationType.getValue()==Hair)
                            {
                                unsigned sizeHair = size+size/(nbPtsByHair.getValue());
                                file.write((char*)&sizeHair,4);
                            }
                            else
                                file.write((char*)&size,4);

                            // dataType
                            unsigned dataType;
                            switch (simulationType.getValue())
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


                            for(int i=(int)size-1; i>=0; i--)
                            {
                                //create an additional point for root tangent
                                if((simulationType.getValue() == Hair && (i%nbPtsByHair.getValue()==0)))
                                {

                                   defaulttype::Vector3  x0 = T::getCPos(posData[i]);
                                   defaulttype::Vector3  x1 = T::getCPos(posData[i+1]);

                                    x1 = x1-x0;
                                   // sout<<"tangeant direction: "<<x1<<sendl;
                                    x1.normalize();

                                   // sout<<"tangeant direction normalized: "<<x1<<sendl;

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
                                if((mmodel->read(core::ConstVecDerivId::velocity())) && ( (defaulttype::BaseVector::Index)mmodel->readVelocities().size()>i))
                                {
                                    v =mmodel->readVelocities()[i];
                                    vel[0] = (float)v[0]; 
                                    vel[1] = (float)v[1]; 
                                    vel[2] = (float)v[2];
                                }

                                Coord x0;
                                if((mmodel->read(core::ConstVecCoordId::restPosition())) && ( (defaulttype::BaseVector::Index)mmodel->readRestPositions().size()>i))
                                {
                                    x0 =mmodel->readRestPositions()[i];
                                    rest[0] = (float)x0[0]; 
                                    rest[1] = (float)x0[1]; 
                                    rest[2] = (float)x0[2];
                                }



                                switch (simulationType.getValue())
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
                            serr<<"Unable to create the following file: "<<fileName<<sendl;

                    }
                    frameCounter++;
                }

            }


        } // namespace misc

    } // namespace component

} // namespace sofa
