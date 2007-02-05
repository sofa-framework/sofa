#ifndef SOFA_COMPONENTS_READSTATE_INL
#define SOFA_COMPONENTS_READSTATE_INL

#include "ReadState.h"
#include "Sofa/Components/Graph/MechanicalAction.h"
#include "Sofa/Components/Graph/UpdateMappingAction.h"

#include <sstream>

namespace Sofa
{

namespace Components
{

using namespace Common;

template<class DataTypes>
ReadState<DataTypes>::ReadState(Core::MechanicalModel<DataTypes>*)
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
    mmodel = dynamic_cast<Core::MechanicalModel<DataTypes>*>(this->getContext()->getMechanicalModel());
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
void ReadState<DataTypes>::handleEvent(Sofa::Abstract::Event* event)
{
    if (AnimateBeginEvent* ev = dynamic_cast<AnimateBeginEvent*>(event))
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
            static_cast<Sofa::Components::Graph::GNode*>(this->getContext())->execute<Sofa::Components::Graph::MechanicalPropagatePositionAndVelocityAction>();
            static_cast<Sofa::Components::Graph::GNode*>(this->getContext())->execute<Sofa::Components::Graph::UpdateMappingAction>();
        }
    }
    if (AnimateEndEvent* ev = dynamic_cast<AnimateEndEvent*>(event))
    {
    }
}

} // namespace Components

} // namespace Sofa

#endif
