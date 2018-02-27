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
#ifndef SOFA_COMPONENT_MISC_EVALPOINTSDISTANCE_INL
#define SOFA_COMPONENT_MISC_EVALPOINTSDISTANCE_INL

#include "EvalPointsDistance.h"
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/helper/gl/template.h>

#include <iomanip>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
EvalPointsDistance<DataTypes>::EvalPointsDistance()
    : f_draw( initData(&f_draw, true, "draw", "activate rendering of lines between associated points"))
    , isToPrint( initData(&isToPrint, false, "isToPrint", "suppress somes data before using save as function"))
    , f_filename( initData(&f_filename, "filename", "output file name"))
    , f_period( initData(&f_period, 0.0, "period", "period between outputs"))
    , dist( initData(&dist, "distance", "distances (OUTPUT)"))
    , distMean( initData(&distMean, 1.0, "distMean", "mean distance (OUTPUT)"))
    , distMin( initData(&distMin, 1.0, "distMin", "min distance (OUTPUT)"))
    , distMax( initData(&distMax, 1.0, "distMax", "max distance (OUTPUT)"))
    , distDev( initData(&distDev, 1.0, "distDev", "distance standard deviation (OUTPUT)"))
    , rdistMean( initData(&rdistMean, 1.0, "rdistMean", "mean relative distance (OUTPUT)"))
    , rdistMin( initData(&rdistMin, 1.0, "rdistMin", "min relative distance (OUTPUT)"))
    , rdistMax( initData(&rdistMax, 1.0, "rdistMax", "max relative distance (OUTPUT)"))
    , rdistDev( initData(&rdistDev, 1.0, "rdistDev", "relative distance standard deviation (OUTPUT)"))
    , mstate1(initLink("object1", "Mechanical state 1"))
    , mstate2(initLink("object2", "Mechanical state 2"))
    , outfile(NULL)
    , lastTime(0)
{


    mstate1.setPath("@./"); // default path: state in the same node
    mstate2.setPath("@./"); // default path: state in the same node
    box1 = sofa::defaulttype::BoundingBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    box2 = sofa::defaulttype::BoundingBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
}

template<class DataTypes>
EvalPointsDistance<DataTypes>::~EvalPointsDistance()
{
    if (outfile)
        delete outfile;
}


//-------------------------------- init------------------------------------
template<class DataTypes>
void EvalPointsDistance<DataTypes>::init()
{
    if(isToPrint.getValue()==true) dist.setPersistent(false);
    if (!mstate1 )
    {
        mstate1 = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
        box1 = mstate1->f_bbox.getValue();
        serr << " Mechanical State object1 not found, this will be taken in the same context " << sendl;
    }
    if (!mstate2)
    {
        mstate2 = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
        this->box2 = mstate1->f_bbox.getValue();
        serr << " Mechanical State object2 not found, this will be taken in the same context " << sendl;
    }


    if (!mstate1 || !mstate2)
    {
        serr << " ERROR Mechanical State object1 and object2 expected  " << sendl;
        return;
    }

    reinit();
}

//-------------------------------- reinit ----------------------------------
template<class DataTypes>
void EvalPointsDistance<DataTypes>::reinit()
{
    if (outfile)
        delete outfile;
    const std::string& filename = f_filename.getFullPath();
    if (!filename.empty())
    {
        outfile = new std::ofstream(filename.c_str());
        if( !outfile->is_open() )
        {
            serr << "Error creating file "<<filename<<sendl;
            delete outfile;
            outfile = NULL;
        }
        else
        {
            (*outfile) << "# name\t\t\ttime\t\tmean\t\tmin\t\tmax\t\tdev\t\tmean(%)\t\tmin(%)\t\tmax(%)\t\tdev(%)" << std::endl;
            msg_info() << "OutputFile " << filename << " created.";
        }
    }
    else
    {
        outfile = NULL;
    }

    if(f_period.getValue() == 0.0)
    {
        serr << " ERROR period must be different of zero  " << sendl;
        return;
    }

    lastTime = -f_period.getValue();
    eval();
}


//-------------------------------- eval ------------------------------------
template<class DataTypes>
SReal EvalPointsDistance<DataTypes>::eval()
{
    if (!mstate1 || !mstate2)
        return 0.0;
    const VecCoord& x0 = mstate1->read(core::ConstVecCoordId::restPosition())->getValue();
    const VecCoord& x1 = mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x2 = mstate2->read(core::ConstVecCoordId::position())->getValue();

    return this->doEval(x1, x2, x0);
}

//-------------------------------- doEval------------------------------------
template<class DataTypes>
SReal EvalPointsDistance<DataTypes>::doEval(const VecCoord& x1, const VecCoord& x2, const VecCoord& /*x0*/)
{
    const int n = (x1.size()<x2.size())?x1.size():x2.size();
    int s1 = x1.size()-n;
    int s2 = x2.size()-n;
    Real dsum = 0.0;
    Real dmin = 0.0;
    Real dmax = 0.0;
    Real d2 = 0.0;
    Real rdsum = 0.0;
    Real rdmin = 0.0;
    Real rdmax = 0.0;
    Real rd2 = 0.0;
    int rn=0;
//    Coord dx0 = x2[s2]-x0[s1];

    const Vec3 minBox = box1.minBBox();
    const Vec3 maxBox = box1.maxBBox();
    Real meanRefSize = (Real)((maxBox[0]-minBox[0])+(maxBox[1]-minBox[1])+(maxBox[2]-minBox[2]))/3.0f;
    helper::vector<Real> &distances = *dist.beginEdit();
    distances.resize(n);
    for (int i=0; i<n; ++i)
    {
        Real d = (Real)(x1[s1+i]-x2[s2+i]).norm();
        distances[i] = d;
        dsum += d;
        d2 += d*d;
        if (i==0 || d < dmin) dmin = d;
        if (i==0 || d > dmax) dmax = d;
        //Real d0 = (Real)(x1[s1+i]-x0[s1+i]).norm();
        //Real d0 = (Real)(x2[s2+i]-x0[s1+i]-dx0).norm();
        Real d0 = meanRefSize;
        if (d0 > 1.0e-6)
        {
            Real rd = d/d0;
            rdsum += rd;
            rd2 += rd*rd;
            if (rn==0 || rd < rdmin) rdmin = rd;
            if (rn==0 || rd > rdmax) rdmax = rd;
            ++rn;
        }
    }
    dist.endEdit();

    Real dmean = (n>0)?dsum/n : (Real)0.0;
    Real ddev = (Real)((n>1)?sqrtf((float)(d2/n - (dsum/n)*(dsum/n))) : 0.0);
    distMean.setValue(dmean);
    distMin.setValue(dmin);
    distMax.setValue(dmax);
    distDev.setValue(ddev);

    Real rdmean = (rn>0)?rdsum/rn : (Real)0.0;
    Real rddev = (Real)((rn>1)?sqrtf((float)(rd2/rn - (rdsum/rn)*(rdsum/rn))) : 0.0);
    rdistMean.setValue(rdmean);
    rdistMin.setValue(rdmin);
    rdistMax.setValue(rdmax);
    rdistDev.setValue(rddev);

    return dmean;
}

//-------------------------------- draw ------------------------------------
template<class DataTypes>
void EvalPointsDistance<DataTypes>::draw(const core::visual::VisualParams* )
{
    if (!f_draw.getValue())
        return;
    if (!mstate1 || !mstate2)
        return;
    const VecCoord& x1 = mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x2 = mstate2->read(core::ConstVecCoordId::position())->getValue();
    this->doDraw(x1,x2);
}

//-------------------------------- doDraw------------------------------------
template<class DataTypes>
void EvalPointsDistance<DataTypes>::doDraw(const VecCoord& x1, const VecCoord& x2)
{
#ifndef SOFA_NO_OPENGL
    const int n = (x1.size()<x2.size())?x1.size():x2.size();
    int s1 = x1.size()-n;
    int s2 = x2.size()-n;
    glDisable(GL_LIGHTING);
    glColor3f(1.0f,0.5f,0.5f);
    glBegin(GL_LINES);
    for (int i=0; i<n; ++i)
    {
        helper::gl::glVertexT(x1[s1+i]);
        helper::gl::glVertexT(x2[s2+i]);
    }
    glEnd();
#endif /* SOFA_NO_OPENGL */
}

//-------------------------------- handleEvent ------------------------------------
template<class DataTypes>
void EvalPointsDistance<DataTypes>::handleEvent(sofa::core::objectmodel::Event* event)
{

    if (!mstate1 || !mstate2)
            return;
    // std::ostream *out = (outfile==NULL)? (std::ostream *)(&sout) : outfile;
    if (simulation::AnimateEndEvent::checkEventType(event))
    {
        double time = getContext()->getTime();
        // write the state using a period
        if (time+getContext()->getDt()/2 >= (lastTime + f_period.getValue()))
        {

            eval();
            if (outfile==NULL)
            {
                sout << "# name\ttime\tmean\tmin\tmax\tdev\tmean(%)\tmin(%)\tmax(%)\tdev(%)" << sendl;
                sout << this->getName() << "\t" << time
                     << "\t" << distMean.getValue() << "\t" << distMin.getValue() << "\t" << distMax.getValue() << "\t" << distDev.getValue()
                     << "\t" << 100*rdistMean.getValue() << "\t" << 100*rdistMin.getValue() << "\t" << 100*rdistMax.getValue() << "\t" << 100*rdistDev.getValue()
                     <<  sendl;
            }
            else
            {
                (*outfile) << std::setfill(' ') << std::setw(10) << this->getName() << "\t" << std::setw(10) << time
                           << "\t" << std::setw(10) << distMean.getValue() << "\t" << std::setw(10) << distMin.getValue() << "\t" << std::setw(10) << distMax.getValue()
                           << "\t" << std::setw(10) << distDev.getValue()  << "\t" << std::setw(10) << 100*rdistMean.getValue() << "\t" << std::setw(10) << 100*rdistMin.getValue()
                           << "\t" << std::setw(10) << 100*rdistMax.getValue() << "\t" << std::setw(10) << 100*rdistDev.getValue()
                           << std::endl;
            }
            lastTime = time;
        }
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
