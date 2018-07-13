/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_WRITESTATE_INL
#define SOFA_COMPONENT_MISC_WRITESTATE_INL

#include <SofaExporter/WriteState.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <fstream>
#include <sstream>

namespace sofa
{

namespace component
{

namespace misc
{


WriteState::WriteState()
    : f_filename( initData(&f_filename, "filename", "output file name"))
    , f_writeX( initData(&f_writeX, true, "writeX", "flag enabling output of X vector"))
    , f_writeX0( initData(&f_writeX0, false, "writeX0", "flag enabling output of X0 vector"))
    , f_writeV( initData(&f_writeV, false, "writeV", "flag enabling output of V vector"))
    , f_writeF( initData(&f_writeF, false, "writeF", "flag enabling output of F vector"))
    , f_time( initData(&f_time, helper::vector<double>(0), "time", "set time to write outputs (by default export at t=0)"))
    , f_period( initData(&f_period, 0.0, "period", "period between outputs"))
    , f_DOFsX( initData(&f_DOFsX, helper::vector<unsigned int>(0), "DOFsX", "set the position DOFs to write"))
    , f_DOFsV( initData(&f_DOFsV, helper::vector<unsigned int>(0), "DOFsV", "set the velocity DOFs to write"))
    , f_stopAt( initData(&f_stopAt, 0.0, "stopAt", "stop the simulation when the given threshold is reached"))
    , f_keperiod( initData(&f_keperiod, 0.0, "keperiod", "set the period to measure the kinetic energy increase"))
    , mmodel(NULL)
    , outfile(NULL)
#ifdef SOFA_HAVE_ZLIB
    , gzfile(NULL)
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
#ifdef SOFA_HAVE_ZLIB
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
        timeToTestEnergyIncrease = f_keperiod.getValue();
    }
    ///////////// end of the tests.

    const std::string& filename = f_filename.getFullPath();
    if (!filename.empty())
    {
#ifdef SOFA_HAVE_ZLIB
        if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
        {
            gzfile = gzopen(filename.c_str(),"wb");
            if( !gzfile )
            {
                msg_error() << "Error creating compressed file "<<filename;
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
                outfile = NULL;
            }
        }
    }

    ///Check all input data
    double dt = this->getContext()->getDt();
    //check filename is set
    if(!f_filename.isSet())
    {
        msg_warning() << "a filename must be specified for export"
                      << "default: defaultExportFile";
        f_filename.setValue(" defaultExportFile");
    }

    //check period
    if(f_period.isSet())
    {
        if(f_time.getValue().size() == 0)
        {
            msg_warning() << "starting time should be specified to know when to start the periodic export"
                          << "by default: start time=0";

            helper::vector<double>& timeVector = *f_time.beginEdit();
            timeVector.clear();
            timeVector.resize(1);
            timeVector[0] = 0.0;
            f_time.endEdit();
        }
        if(f_time.getValue().size() > 1)
        {
            helper::vector<double>& timeVector = *f_time.beginEdit();
            timeVector.resize(1);
            f_time.endEdit();
            msg_warning() << "using the period assume to have only one starting time for export: "<<f_time.getValue()[0];
        }

        if(f_period.getValue() < dt)
        {
            msg_warning() << "period ("<< f_period.getValue() <<") input value is too low regarding the time step ("<< dt <<")";
        }

        if(f_time.getValue()[0]!=0.0 && f_time.getValue()[0]<dt)
        {
            msg_warning() << "starting export time ("<< f_time.getValue()[0] <<") is too low regarding the time step ("<< dt <<")";
        }
        periodicExport = true;
    }

    //check time
    if(!f_time.isSet())
    {
        msg_warning() << "an export time should be specified"
                      << "by default, export at t=0";
        helper::vector<double>& timeVector = *f_time.beginEdit();
        timeVector.clear();
        timeVector.resize(1);
        timeVector[0] = 0.0;
        f_time.endEdit();
    }
    else
    {
        for(unsigned int i=0; i<f_time.getValue().size(); i++)
        {
            if(f_time.getValue()[i] <= 0)
            {
                if(i==0)
                {
                    if(f_time.getValue()[i] != 0)
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
            double mutiple = fmod(f_time.getValue()[i],dt);
            int integerM = (int) mutiple;
            mutiple -= (double)integerM;
            if(mutiple > std::numeric_limits<double>::epsilon())
            {
                msg_warning() << "desired export time ("<< f_time.getValue()[i] <<") can not be met with the chosen time step("<< dt <<")";
            }
        }
    }

    //check stopAt
    if(f_stopAt.getValue()<0)
    {
        msg_warning() << "stopping time should be strictly positive"
                      << "default value stopAt=0.0";
        f_stopAt.setValue(0.0);
    }
}

void WriteState::reinit(){
if (outfile)
    delete outfile;
#ifdef SOFA_HAVE_ZLIB
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
    timeToTestEnergyIncrease = f_keperiod.getValue();
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
#ifdef SOFA_HAVE_ZLIB
            && !gzfile
#endif
           )
            return;

        if (kineticEnergyThresholdReached)
            return;

        double time = getContext()->getTime();
        // the time to measure the increase of energy is reached
        if (f_stopAt.getValue())
        {
            if (time > timeToTestEnergyIncrease)
            {
                simulation::Node *gnode = dynamic_cast<simulation::Node *>(this->getContext());
                if (!gnode->mass)
                {
                    // Error: the mechanical model has no mass
                    msg_error() << "Error: Kinetic energy can not be computed. The mass for " << mmodel->getName() << " has no been defined";
                    exit(EXIT_FAILURE);
                }
                else
                {
                    // computes the energy increase
                    if (fabs(gnode->mass->getKineticEnergy() - savedKineticEnergy) < f_stopAt.getValue())
                    {
                        sout << "WriteState has been stopped. Kinetic energy threshold has been reached" << sendl;
                        kineticEnergyThresholdReached = true;
                    }
                    else
                    {
                        // save the last energy measured
                        savedKineticEnergy = gnode->mass->getKineticEnergy();
                        // increase the period to measure the energy
                        timeToTestEnergyIncrease+=f_keperiod.getValue();
                    }
                }
            }
        }

        //check if the state has to be written or not
        bool writeCurrent = false;
        if (nextIteration<f_time.getValue().size())
        {
            // store the actual time instant
            lastTime = f_time.getValue()[nextIteration];
            // if the time simulation is >= that the actual time instant
            if ( (time > lastTime) || (fabs(time - lastTime)< std::numeric_limits<double>::epsilon()) )
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
                double nextTime = lastTime + f_period.getValue();
                // write the state using a period
                if ( (time > nextTime) || (fabs(time - nextTime)< std::numeric_limits<double>::epsilon()) )
                {
                    writeCurrent = true;
                    lastTime += f_period.getValue();
                }
            }
        }
        if (writeCurrent)
        {
#ifdef SOFA_HAVE_ZLIB
            if (gzfile)
            {
                // write the X state
                std::ostringstream str;
                str << "T= "<< time << "\n";
                if (f_writeX.getValue())
                {
                    str << "  X= ";
                    mmodel->writeVec(core::VecId::position(), str);
                    str << "\n";
                }
                if (f_writeX0.getValue())
                {
                    str << "  X0= ";
                    mmodel->writeVec(core::VecId::restPosition(), str);
                    str << "\n";
                }
                //write the V state
                if (f_writeV.getValue())
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
                    if (f_writeX.getValue())
                    {
                        (*outfile) << "  X= ";
                        mmodel->writeVec(core::VecId::position(), *outfile);
                        (*outfile) << "\n";
                    }
                    if (f_writeX0.getValue())
                    {
                        (*outfile) << "  X0= ";
                        mmodel->writeVec(core::VecId::restPosition(), *outfile);
                        (*outfile) << "\n";
                    }
                    //write the V state
                    if (f_writeV.getValue())
                    {
                        (*outfile) << "  V= ";
                        mmodel->writeVec(core::VecId::velocity(), *outfile);
                        (*outfile) << "\n";
                    }
                    //write the F state
                    if (f_writeF.getValue())
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

} // namespace misc

} // namespace component

} // namespace sofa

#endif
