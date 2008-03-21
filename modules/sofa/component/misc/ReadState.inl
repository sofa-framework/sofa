#ifndef SOFA_COMPONENT_MISC_READSTATE_INL
#define SOFA_COMPONENT_MISC_READSTATE_INL

#include <sofa/component/misc/ReadState.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/simulation/tree/UpdateMappingVisitor.h>

#include <sstream>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
ReadState<DataTypes>::ReadState()
    : f_filename( initData(&f_filename, "filename", "output file name"))
    , f_interval( initData(&f_interval, 0.0, "interval", "time duration between inputs"))
    , f_shift( initData(&f_shift, 0.0, "shift", "shift between times in the file and times when they will be read"))
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
//     mmodel = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    reset();
}

template<class DataTypes>
void ReadState<DataTypes>::reset()
{
    mmodel = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
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
        processReadState();
    }
    if (/* simulation::tree::AnimateEndEvent* ev = */ dynamic_cast<simulation::tree::AnimateEndEvent*>(event))
    {
    }
}



template<class DataTypes>
void ReadState<DataTypes>::setTime(double time)
{
    if (time < nextTime) {reset(); nextTime=0.0;}
}

template<class DataTypes>
void ReadState<DataTypes>::processReadState(double time)
{
    if (time == lastTime) return;
    setTime(time);
    processReadState();
}

template<class DataTypes>
void ReadState<DataTypes>::processReadState()
{
    bool updated = false;

    if (infile && mmodel)
    {
        double time = getContext()->getTime() + f_shift.getValue();
        lastTime = time;
        std::vector<std::string> validLines;
        std::string line, cmd;
        while (nextTime <= time && !infile->eof())
        {
            getline(*infile, line);
            //std::cout << "line= "<<line<<std::endl;
            std::istringstream str(line);
            str >> cmd;
            //std::cout << "cmd= "<<cmd<<std::endl;
            if (cmd == "T=")
            {
                str >> nextTime;
                validLines.clear();
            }

            if (nextTime <= time) validLines.push_back(line);
        }
        for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end(); ++it)
        {
            std::istringstream str(*it);
            cmd.clear();
            str >> cmd;
            //std::cout << "cmd= "<<cmd<<std::endl;
            if (cmd == "X=")
            {
                int prevSize = mmodel->getSize();
                //str >> (*mmodel->getX());
                VecCoord v;
                str >> v;
                int newSize = v.size();
                if (newSize == 0)
                {
                    std::cerr << "ReadState("<<f_filename.getValue()<<"): Empty vector read. Ignoring..."<<std::endl;
                }
                else
                {
                    if (newSize != prevSize)
                    {
                        std::cout << "ReadState("<<f_filename.getValue()<<"): size changed from "<<prevSize<<" to "<<newSize<<std::endl;
                        mmodel->resize(newSize);
                    }
                    (*mmodel->getX()) = v;
                }
                updated = true;
            }
            else if (cmd == "V=")
            {
                int prevSize = mmodel->getSize();
                //str >> (*mmodel->getV());
                VecDeriv v;
                str >> v;
                int newSize = v.size();
                if (newSize == 0)
                {
                    std::cerr << "ReadState("<<f_filename.getValue()<<"): Empty vector read. Ignoring..."<<std::endl;
                }
                else
                {
                    if (newSize != prevSize)
                    {
                        std::cout << "ReadState("<<f_filename.getValue()<<"): size changed from "<<prevSize<<" to "<<newSize<<std::endl;
                        mmodel->resize(newSize);
                    }
                    (*mmodel->getV()) = v;
                }
                updated = true;
            }
        }
    }
    if (updated)
    {
        //std::cout<<"update from file"<<std::endl;
        sofa::simulation::tree::MechanicalPropagatePositionAndVelocityVisitor action1;
        this->getContext()->executeVisitor(&action1);
        sofa::simulation::tree::UpdateMappingVisitor action2;
        this->getContext()->executeVisitor(&action2);
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
