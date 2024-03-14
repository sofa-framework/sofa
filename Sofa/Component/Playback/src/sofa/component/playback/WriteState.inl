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
#pragma once

#include <sofa/component/playback/WriteState.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/BaseMass.h>
#include <fstream>
#include <sstream>

namespace sofa::component::playback
{

WriteState::WriteState()
    : d_filename( initData(&d_filename, "filename", "output file name"))
    , d_writeX( initData(&d_writeX, true, "writeX", "flag enabling output of X vector"))
    , d_writeX0( initData(&d_writeX0, false, "writeX0", "flag enabling output of X0 vector"))
    , d_writeV( initData(&d_writeV, false, "writeV", "flag enabling output of V vector"))
    , d_writeF( initData(&d_writeF, false, "writeF", "flag enabling output of F vector"))
    , d_time( initData(&d_time, type::vector<double>(0), "time", "set time to write outputs (by default export at t=0)"))
    , d_period( initData(&d_period, 0.0, "period", "period between outputs"))
    , d_DOFsX( initData(&d_DOFsX, type::vector<unsigned int>(0), "DOFsX", "set the position DOFs to write"))
    , d_DOFsV( initData(&d_DOFsV, type::vector<unsigned int>(0), "DOFsV", "set the velocity DOFs to write"))
    , d_stopAt( initData(&d_stopAt, 0.0, "stopAt", "stop the simulation when the given threshold is reached"))
    , d_keperiod( initData(&d_keperiod, 0.0, "keperiod", "set the period to measure the kinetic energy increase"))
    , mmodel(nullptr)
    , outfile(nullptr)
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , gzfile(nullptr)
#endif
    , nextIteration(0)
    , lastTime(0)
    , kineticEnergyThresholdReached(false)
    , timeToTestEnergyIncrease(0)
    , savedKineticEnergy(0)
{
    this->f_listening.setValue(true);
}


WriteState::~WriteState()
{
    if (outfile)
        delete outfile;
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    if (gzfile)
        gzclose(gzfile);
#endif
}


void WriteState::init()
{
    validInit = true;
    periodicExport = false;
    mmodel = this->getContext()->getMechanicalState();

    // test the size and range of the DOFs to write in the file output
    if (mmodel)
    {
        timeToTestEnergyIncrease = d_keperiod.getValue();
    }
    ///////////// end of the tests.

    const std::string& filename = d_filename.getFullPath();
    if (!filename.empty())
    {
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
        if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
        {
            gzfile = gzopen(filename.c_str(),"wb");
            if( !gzfile )
            {
                msg_error() << "Error creating compressed file " << filename
                            << ". Reason: " << std::strerror(errno);
            }
        }
        else
#endif
        {
            outfile = new std::ofstream(filename.c_str());
            if( !outfile->is_open() )
            {
                msg_error() << "Error creating file "<<filename;
                delete outfile;
                outfile = nullptr;
            }
        }
    }

    ///Check all input data
    const double dt = this->getContext()->getDt();
    //check filename is set
    if(!d_filename.isSet())
    {
        msg_warning() << "a filename must be specified for export"
                      << "default: defaultExportFile";
        d_filename.setValue(" defaultExportFile");
    }


    //check period
    if(d_period.isSet())
    {
        periodicExport = true;

        if(d_time.getValue().size() == 0)
        {
            msg_warning() << "starting time should be specified to know when to start the periodic export"
                          << "by default: start time=0";

            type::vector<double>& timeVector = *d_time.beginEdit();
            timeVector.clear();
            timeVector.resize(1);
            timeVector[0] = 0.0;
            d_time.endEdit();
        }
        if(d_time.getValue().size() > 1)
        {
            type::vector<double>& timeVector = *d_time.beginEdit();
            timeVector.resize(1);
            d_time.endEdit();
            msg_warning() << "using the period assume to have only one starting time for export: "<<d_time.getValue()[0];
        }

        if(d_period.getValue() < dt)
        {
            msg_warning() << "period ("<< d_period.getValue() <<") input value is too low regarding the time step ("<< dt <<")";
        }

        if(d_time.getValue()[0]!=0.0 && d_time.getValue()[0]<dt)
        {
            msg_warning() << "starting export time ("<< d_time.getValue()[0] <<") is too low regarding the time step ("<< dt <<")";
        }
    }
    else
    {
        if(!d_time.isSet())
        {
            d_period.setValue(this->getContext()->getDt());
            periodicExport = true;
        }
    }


    //check time
    if(!d_time.isSet())
    {
        msg_warning() << "an export time should be specified"
                      << "by default, export at t=0.0";
        type::vector<double>& timeVector = *d_time.beginEdit();
        timeVector.clear();
        timeVector.resize(1);
        timeVector[0] = 0.0;
        d_time.endEdit();
    }
    else
    {
        for(unsigned int i=0; i<d_time.getValue().size(); i++)
        {
            if(d_time.getValue()[i] <= 0)
            {
                if(i==0)
                {
                    if(d_time.getValue()[i] != 0)
                    {
                        msg_error() << "time of export should always be zero or positive, no export will be done";
                        validInit = false;
                        return;
                    }
                }
                else
                {
                    msg_error() << "time of export should always be positive, no export will be done";
                    validInit = false;
                    return;
                }
            }

            //check that the desired export times will be met with the chosen time step
            const double nbDtInTime = d_time.getValue()[i]/dt;
            const int intnbDtInTime = (int) nbDtInTime;
            const double rest = nbDtInTime - intnbDtInTime;
            if(rest > std::numeric_limits<double>::epsilon())
            {
                msg_warning() << "desired export time ("<< d_time.getValue()[i] <<") can not be met with the chosen time step("<< dt <<")";
            }
        }

    }

    //check stopAt
    if(d_stopAt.getValue()<0)
    {
        msg_warning() << "stopping time should be strictly positive"
                      << "default value stopAt=0.0";
        d_stopAt.setValue(0.0);
    }
}

void WriteState::reinit(){
if (outfile)
    delete outfile;
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
if (gzfile)
    gzclose(gzfile);
#endif
init();
}
void WriteState::reset()
{
    nextIteration = 0;
    lastTime = 0;
    kineticEnergyThresholdReached = false;
    timeToTestEnergyIncrease = d_keperiod.getValue();
    savedKineticEnergy = 0;
}


void WriteState::handleEvent(sofa::core::objectmodel::Event* event)
{
    if(!validInit)
        return;

    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        if (!mmodel) return;
        if (!outfile
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
            && !gzfile
#endif
           )
            return;

        if (kineticEnergyThresholdReached)
            return;

        const double time = getContext()->getTime();
        // the time to measure the increase of energy is reached
        if (d_stopAt.getValue())
        {
            if (time > timeToTestEnergyIncrease)
            {
                const simulation::Node *gnode = dynamic_cast<simulation::Node *>(this->getContext());
                if (!gnode->mass)
                {
                    // Error: the mechanical model has no mass
                    msg_error() << "Error: Kinetic energy can not be computed. The mass for " << mmodel->getName() << " has no been defined";
                    exit(EXIT_FAILURE);
                }
                else
                {
                    // computes the energy increase
                    if (fabs(gnode->mass->getKineticEnergy() - savedKineticEnergy) < d_stopAt.getValue())
                    {
                        msg_info() << "WriteState has been stopped. Kinetic energy threshold has been reached";
                        kineticEnergyThresholdReached = true;
                    }
                    else
                    {
                        // save the last energy measured
                        savedKineticEnergy = gnode->mass->getKineticEnergy();
                        // increase the period to measure the energy
                        timeToTestEnergyIncrease+=d_keperiod.getValue();
                    }
                }
            }
        }

        //check if the state has to be written or not
        bool writeCurrent = false;
        const SReal epsilonStep = 0.1*this->getContext()->getDt();
        if (nextIteration<d_time.getValue().size())
        {
            // store the actual time instant
            lastTime = d_time.getValue()[nextIteration];
            // if the time simulation is >= that the actual time instant
            if ( (time > lastTime) || (fabs(time - lastTime)< epsilonStep) )
            {
                writeCurrent = true;
                firstExport = true;
                nextIteration++;
            }
        }
        else
        {
            if(firstExport && periodicExport)
            {
                const double nextTime = lastTime + d_period.getValue();
                // write the state using a period
                if ( (time > nextTime) || (fabs(time - nextTime)< epsilonStep) )
                {
                    writeCurrent = true;
                    lastTime += d_period.getValue();
                }
            }
        }
        if (writeCurrent)
        {
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
            if (gzfile)
            {
                // write the X state
                std::ostringstream str;
                str << "T= "<< time << "\n";
                if (d_writeX.getValue())
                {
                    str << "  X= ";
                    mmodel->writeVec(core::VecId::position(), str);
                    str << "\n";
                }
                if (d_writeX0.getValue())
                {
                    str << "  X0= ";
                    mmodel->writeVec(core::VecId::restPosition(), str);
                    str << "\n";
                }
                //write the V state
                if (d_writeV.getValue())
                {
                    str << "  V= ";
                    mmodel->writeVec(core::VecId::velocity(), str);
                    str << "\n";
                }
                gzputs(gzfile, str.str().c_str());
                gzflush(gzfile, Z_SYNC_FLUSH);
            }
            else
#endif
                if (outfile)
                {
                    // write the X state
                    (*outfile) << "T= "<< time << "\n";
                    if (d_writeX.getValue())
                    {
                        (*outfile) << "  X= ";
                        mmodel->writeVec(core::VecId::position(), *outfile);
                        (*outfile) << "\n";
                    }
                    if (d_writeX0.getValue())
                    {
                        (*outfile) << "  X0= ";
                        mmodel->writeVec(core::VecId::restPosition(), *outfile);
                        (*outfile) << "\n";
                    }
                    //write the V state
                    if (d_writeV.getValue())
                    {
                        (*outfile) << "  V= ";
                        mmodel->writeVec(core::VecId::velocity(), *outfile);
                        (*outfile) << "\n";
                    }
                    //write the F state
                    if (d_writeF.getValue())
                    {
                        (*outfile) << "  F= ";
                        mmodel->writeVec(core::VecId::force(), *outfile);
                        (*outfile) << "\n";
                    }
                    outfile->flush();
                }
            msg_info() <<"Export done (time = "<< time <<")";
        }
    }
}

} // namespace sofa::component::playback
