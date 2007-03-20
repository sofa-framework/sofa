#ifndef SOFA_COMPONENT_MISC_WRITESTATE_INL
#define SOFA_COMPONENT_MISC_WRITESTATE_INL

#include "WriteState.h"

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
WriteState<DataTypes>::WriteState()
    : f_filename( dataField(&f_filename, "filename", "output file name"))
    , f_writeX( dataField(&f_writeX, true, "writeX", "flag enabling output of X vector"))
    , f_writeV( dataField(&f_writeV, false, "writeV", "flag enabling output of V vector"))
    , f_interval( dataField(&f_interval, 0.0, "interval", "time duration between outputs"))
    , mmodel(NULL)
    , outfile(NULL)
    , nextTime(0)
{
    this->f_listening.setValue(true);
}

template<class DataTypes>
WriteState<DataTypes>::~WriteState()
{
    if (outfile)
        delete outfile;
}

template<class DataTypes>
void WriteState<DataTypes>::init()
{
    mmodel = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    const std::string& filename = f_filename.getValue();
    if (!filename.empty())
    {
        std::ifstream infile(filename.c_str());
        if( infile.is_open() )
        {
            std::cerr << "ERROR: file "<<filename<<" already exists. Remove it to record new motion."<<std::endl;
        }
        else
        {
            outfile = new std::ofstream(filename.c_str());
            if( !outfile->is_open() )
            {
                std::cerr << "Error creating file "<<filename<<std::endl;
                delete outfile;
                outfile = NULL;
            }
        }
    }
}

template<class DataTypes>
void WriteState<DataTypes>::reset()
{
    nextTime = 0;
}

template<class DataTypes>
void WriteState<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::tree::AnimateBeginEvent* ev = */ dynamic_cast<simulation::tree::AnimateBeginEvent*>(event))
    {
        if (outfile && mmodel)
        {
            double time = getContext()->getTime();
            if (time >= nextTime)
            {
                (*outfile) << "T= "<< time << "\n";
                if (f_writeX.getValue())
                    (*outfile) << "  X= "<< (*mmodel->getX()) << "\n";
                if (f_writeV.getValue())
                    (*outfile) << "  V= "<< (*mmodel->getV()) << "\n";
                nextTime += f_interval.getValue();
                if (nextTime < time) nextTime = time;
            }
        }
    }
    if (/* simulation::tree::AnimateEndEvent* ev = */ dynamic_cast<simulation::tree::AnimateEndEvent*>(event))
    {
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
