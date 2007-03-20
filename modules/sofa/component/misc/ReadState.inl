#ifndef SOFA_COMPONENT_MISC_READSTATE_INL
#define SOFA_COMPONENT_MISC_READSTATE_INL

#include "ReadState.h"
#include <sofa/simulation/tree/MechanicalAction.h>
#include <sofa/simulation/tree/UpdateMappingAction.h>

#include <sstream>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
ReadState<DataTypes>::ReadState()
    : f_filename( dataField(&f_filename, "filename", "output file name"))
    , f_interval( dataField(&f_interval, 0.0, "interval", "time duration between outputs"))
    , mmodel(NULL)
    , infile(NULL)
    , nextTime(0)
{
    this->f_listening.setValue(true);
}

template<class DataTypes>
ReadState<DataTypes>::~ReadState()
{
    if (infile)
        delete infile;
}

template<class DataTypes>
void ReadState<DataTypes>::init()
{
    mmodel = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    reset();
}

template<class DataTypes>
void ReadState<DataTypes>::reset()
{
    if (infile)
        delete infile;
    const std::string& filename = f_filename.getValue();
    if (!filename.empty())
    {
        infile = new std::ifstream(filename.c_str());
        if( !infile->is_open() )
        {
            std::cerr << "Error opening file "<<filename<<std::endl;
            delete infile;
            infile = NULL;
        }
    }
    nextTime = 0;
}

template<class DataTypes>
void ReadState<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::tree::AnimateBeginEvent* ev = */ dynamic_cast<simulation::tree::AnimateBeginEvent*>(event))
    {
        bool updated = false;
        if (infile && mmodel)
        {
            double time = getContext()->getTime();
            while (nextTime <= time && !infile->eof())
            {
                std::string line, cmd;
                getline(*infile, line);
                //std::cout << "line= "<<line<<std::endl;
                std::istringstream str(line);
                str >> cmd;
                //std::cout << "cmd= "<<cmd<<std::endl;
                if (cmd == "T=")
                    str >> nextTime;
                else if (cmd == "X=")
                {
                    str >> (*mmodel->getX());
                    updated = true;
                }
                else if (cmd == "V=")
                {
                    str >> (*mmodel->getV());
                    updated = true;
                }
                else
                {
                    std::cerr << "ERROR: Unknown command " << cmd << " in file "<<f_filename.getValue()<<std::endl;
                }
            }
        }
        if (updated)
        {
            //std::cout<<"update from file"<<std::endl;
            static_cast<sofa::simulation::tree::GNode*>(this->getContext())->execute<sofa::simulation::tree::MechanicalPropagatePositionAndVelocityAction>();
            static_cast<sofa::simulation::tree::GNode*>(this->getContext())->execute<sofa::simulation::tree::UpdateMappingAction>();
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
