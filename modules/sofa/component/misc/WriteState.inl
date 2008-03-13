#ifndef SOFA_COMPONENT_MISC_WRITESTATE_INL
#define SOFA_COMPONENT_MISC_WRITESTATE_INL

#include "WriteState.h"
#include <sofa/defaulttype/DataTypeInfo.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
WriteState<DataTypes>::WriteState()
    : f_filename( initData(&f_filename, "filename", "output file name"))
    , f_writeX( initData(&f_writeX, true, "writeX", "flag enabling output of X vector"))
    , f_writeV( initData(&f_writeV, false, "writeV", "flag enabling output of V vector"))
    , f_interval( initData(&f_interval, 0.0, "interval", "time duration between outputs"))
    , f_time( initData(&f_time, helper::vector<double>(0), "time", "sets time to write outputs"))
    , f_period( initData(&f_period, 0.0, "period", "period between outputs"))
    , f_DOFsX( initData(&f_DOFsX, helper::vector<unsigned int>(0), "DOFsX", "sets the position DOFs to write"))
    , f_DOFsV( initData(&f_DOFsV, helper::vector<unsigned int>(0), "DOFsV", "sets the velocity DOFs to write"))
    , mmodel(NULL)
    , outfile(NULL)
    , nextTime(0)
    , lastTime(0)
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

    // test the size and range of the DOFs to write in the file output
    if (mmodel)
    {
        // test the position DOFs
        defaulttype::DataTypeInfo<typename DataTypes::Coord> dataInfoX;
        if (dataInfoX.size() < f_DOFsX.getValue().size())
        {
            std::cerr << "ERROR: the size of DOFsX must be equal or smaller than the size of the mechanical data type."<<std::endl;
            exit(-1);
        }
        else
        {
            for (unsigned int i=0; i<f_DOFsX.getValue().size(); i++)
            {
                if (dataInfoX.size() < f_DOFsX.getValue()[i])
                {
                    std::cerr << "ERROR: DOFX index " << f_DOFsX.getValue()[i] << " must contain a value between 0 and " << dataInfoX.size()-1 << std::endl;
                    exit(-1);
                }
            }
        }
        // test the velocity DOFs
        defaulttype::DataTypeInfo<typename DataTypes::Deriv> dataInfoV;
        if (dataInfoV.size() < f_DOFsV.getValue().size())
        {
            std::cerr << "ERROR: the size of DOFsV must be equal or smaller than the size of the mechanical data type."<<std::endl;
            exit(-1);
        }
        else
        {
            for (unsigned int i=0; i<f_DOFsV.getValue().size(); i++)
            {
                if (dataInfoV.size() < f_DOFsV.getValue()[i])
                {
                    std::cerr << "ERROR: DOFV index " << f_DOFsV.getValue()[i] << " must contain a value between 0 and " << dataInfoV.size()-1 << std::endl;
                    exit(-1);
                }
            }
        }
    }
    ///////////// end of the tests.

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
    lastTime = 0;
}

template<class DataTypes>
void WriteState<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::tree::AnimateBeginEvent* ev = */ dynamic_cast<simulation::tree::AnimateBeginEvent*>(event))
    {
        if (outfile && mmodel)
        {
            double time = getContext()->getTime();
            if (nextTime<f_time.getValue().size())
            {
                // store the actual time instant
                lastTime = f_time.getValue()[nextTime];
                if (time >= lastTime) // if the time simulation is >= that the actual time instant
                {
                    // write the X state
                    (*outfile) << "T= "<< time << "\n";
                    if (f_writeX.getValue())
                    {
                        (*outfile) << "  X=";
                        for (int i=0; i<mmodel->getSize(); i++)
                        {
                            for (unsigned int j=0; j<f_DOFsX.getValue().size(); j++)
                                (*outfile) << " " << (*mmodel->getX())[i][f_DOFsX.getValue()[j]];
                        }
                        (*outfile) << "\n";
                    }
                    // write the V state
                    if (f_writeV.getValue())
                    {
                        (*outfile) << "  V=";
                        for (int i=0; i<mmodel->getSize(); i++)
                        {
                            for (unsigned int j=0; j<f_DOFsV.getValue().size(); j++)
                                (*outfile) << " " << (*mmodel->getV())[i][f_DOFsV.getValue()[j]];
                        }
                        (*outfile) << "\n";
                    }
                    outfile->flush();
                    nextTime++;
                }
            }
            else
            {
                // write the state using a period
                if (time >= (lastTime + f_period.getValue()))
                {
                    (*outfile) << "T= "<< time << "\n";
                    if (f_writeX.getValue())
                    {
                        (*outfile) << "  X=";
                        for (int i=0; i<mmodel->getSize(); i++)
                        {
                            for (unsigned int j=0; j<f_DOFsX.getValue().size(); j++)
                                (*outfile) << " " << (*mmodel->getX())[i][f_DOFsX.getValue()[j]];
                        }
                        (*outfile) << "\n";
                    }
                    if (f_writeV.getValue())
                    {
                        (*outfile) << "  V=";
                        for (int i=0; i<mmodel->getSize(); i++)
                        {
                            for (unsigned int j=0; j<f_DOFsV.getValue().size(); j++)
                                (*outfile) << " " << (*mmodel->getV())[i][f_DOFsV.getValue()[j]];
                        }
                        (*outfile) << "\n";
                    }
                    outfile->flush();
                    lastTime += f_period.getValue();
                }
            }
        }
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
