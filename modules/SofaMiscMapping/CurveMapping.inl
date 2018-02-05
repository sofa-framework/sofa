/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
//
// C++ Interface: CurveMapping
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_MAPPING_CURVEMAPPING_INL
#define SOFA_COMPONENT_MAPPING_CURVEMAPPING_INL

#include <SofaMiscMapping/CurveMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/VecId.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>

#include <sofa/defaulttype/Mat.h>

#include <sofa/helper/rmath.h>
#include <sofa/helper/gl/template.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/Simulation.h>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
typename CurveMapping<TIn, TOut>::Real CurveMapping<TIn, TOut>::advanceAbscissa(Real ab, Real dist)
{
    int integer = helper::rfloor(ab);
    if (integer < 0) integer = 0;
    else if (integer > (int)this->lengthElements.size()-1) integer = this->lengthElements.size()-1;
    double fraction = ab - integer;
    while (integer < (int)this->lengthElements.size()-1 && lengthElements[integer] * (1-fraction) < dist)
    {
        dist -= (Real) (lengthElements[integer] * (1-fraction));
        ++integer;
        fraction = 0;
    }
    fraction += dist / lengthElements[integer];
    return (Real) (integer + fraction);
}

inline sofa::defaulttype::Quat computeOrientation(const sofa::defaulttype::Vec3d& AB, const sofa::defaulttype::Quat& Q)
{
    sofa::defaulttype::Vec3d PQ = AB;
    sofa::defaulttype::Quat quat = Q;

    sofa::defaulttype::Vec3d x = quat.rotate(sofa::defaulttype::Vec3d(1,0,0));
    PQ.normalize();

    if (dot(x, PQ) > 0.99)
        return Q;

    sofa::defaulttype::Vec3d y;
    double alpha;

    if (dot(x, PQ) < -0.99)
    {
        y = quat.rotate(sofa::defaulttype::Vec3d(0,0,1));
        alpha = M_PI;
    }
    else
    {
        y = cross(x, PQ);
        y.normalize();
        alpha = acos(dot(x, PQ));
    }

    sofa::defaulttype::Quat qaux = sofa::defaulttype::Quat(y, alpha);

    return (qaux * quat);
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::init()
{
    int nin = this->fromModel->getSize();
    int nout = numNodes.getValue();
    this->toModel->resize(nout);

    lengthElements.resize(nin-1);
    //	const InVecCoord& x0 = this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue();
    const InVecCoord& x0 = this->fromModel->read(core::ConstVecCoordId::position())->getValue();

    for (int i=0; i<nin-1; i++)
    {
        lengthElements[i] = (Real)(x0[i+1]-x0[i]).norm();
        //sout << "l["<<i<<"] = "<<lengthElements[i]<<sendl;
    }

    helper::vector<Real> &a = *angle.beginEdit();
    Real a0 = (a.empty()?(Real)0.0:a[a.size()-1]);
    int ai = a.size();
    a.resize(nout);
    while (ai < nout)
        a[ai++] = a0;
    angle.endEdit();
    quatElements.resize(nin-1);
    sofa::defaulttype::Quat q(0,0,0,1);
    for (int i=0; i<nin-1; i++)
    {
        quatElements[i] = computeOrientation(x0[i+1]-x0[i], q);
        q = quatElements[i];
    }
    rotateElements();

    helper::vector<Real> ab;

    ab = abscissa.getValue();
    ab.resize(nout);

    if (stepNode.getValue() != 0)
    {
        for (unsigned int i=1; i<ab.size(); i++)
            ab[i] = ab[i-1] + stepNode.getValue();
    }
    else if (distNode.getValue() != 0)
    {
        for (unsigned int i=1; i<ab.size(); i++)
            ab[i] = advanceAbscissa( ab[i-1], distNode.getValue());
    }

    abscissa.setValue(ab);

    old_integer.resize(nout);
    fill(old_integer.begin(), old_integer.end(), -1);

    old_angle.resize(nout);
    fill(old_angle.begin(), old_angle.end(), 0.0);

    //apply(*this->toModel->read(sofa::core::ConstVecCoordId::position())->getValue(),this->fromModel->read(core::ConstVecCoordId::position())->getValue());
    //apply(*this->toModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue(), *this->fromModel->read(sofa::core::ConstVecCoordId::freePosition())->getValue());

    Inherit::apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::position(),     core::ConstVecCoordId::position());
    Inherit::apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());

    /*

        VecCoord& xto =this->toModel->read(core::ConstVecCoordId::position())->getValue();
        InVecCoord& xfrom =this->fromModel->read(core::ConstVecCoordId::position())->getValue();

    	std::ofstream catheterDataFile("/media/data/sofa-dev/trunk/Sofa/applications/projects/runSofa/rest_position.dat", std::fstream::out | std::fstream::binary);

    	apply(xto, xfrom);

       catheterDataFile << "BeamModel_Name modelName"
    			//write the name
    			 << endl
    			 << "Number_of_Nodes "
    			 << xto.size()
    			 << endl
    			 << "Number_of_Segments "
    			 << xto.size() - 1
    			 << endl
    			 << "List_of_Nodes" << endl;

    	for (unsigned int i=0; i<xto.size(); i++)
    	{
    		catheterDataFile << i + 1
    				 << "	"
    				 << xto[i].getCenter()
    				 << "	"
    				 << xto[i].getOrientation().rotate(Vec3d(0, 1, 0))
    				 << "	"
    				 << xto[i].getOrientation().rotate(Vec3d(0, 0, 1))
    				 << endl;
    	}

    	catheterDataFile << "Segments" << endl;

    	for (unsigned int i=0; i<xto.size()-1; i++)
    	{
    		catheterDataFile << i + 1
    				 << " "
    				 << i + 1
    				 << " "
    				 << i + 2
    				 << "	"
    				 << "2560000.000000" << endl;
    	}

    	this->toModel->resize(1);
    */
    old_integer.resize(this->toModel->getSize());
    fill(old_integer.begin(), old_integer.end(), -1);
    this->Inherit::init();
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::reinit()
{
    rotateElements();
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::rotateElements()
{
    int nin = this->fromModel->getSize();
    rotatedQuatElements.resize(nin-1);
    //const InVecCoord& x0 = this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue();
    Real a = angle.getValue()[0];
    for (int i=0; i<nin-1; i++)
    {
        rotatedQuatElements[i] = quatElements[i]*sofa::defaulttype::Quat(sofa::defaulttype::Vec3d(1.0,0.0,0.0), a);
        rotatedQuatElements[i].normalize();
    }
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::storeResetState()
{
    reset_abscissa = abscissa.getValue();
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::reset()
{
    abscissa.setValue(reset_abscissa );
    fill(old_integer.begin(), old_integer.end(), -1);
    fill(old_angle.begin(), old_angle.end(), 0.0);
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::apply(const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn)
{
    helper::WriteAccessor< OutDataVecCoord > out = dOut;
    helper::ReadAccessor< InDataVecCoord > in = dIn;

    out.resize(abscissa.getValue().size());
    for (unsigned int i=0; i<out.size(); i++)
    {
        int integer = helper::rfloor(abscissa.getValue()[i]);
        if (integer < 0) integer = 0;
        else if (integer > (int)in.size()-2) integer = in.size()-2;
        double fraction = abscissa.getValue()[i] - integer;
        //if (fraction > 1.0) fraction = 1.0;
        {
            InCoord A, B, AB;
            A = in[integer];
            B = in[integer+1];
            AB = B - A;
            out[i].getCenter() = A + (AB * fraction);
            //if (integer != old_integer[i]) // rigid position has changed
            {
                out[i].getOrientation() = rotatedQuatElements[integer]; //computeOrientation(AB, out[i].getOrientation());

            }
// 			Real Dtheta = angle.getValue()[i] - old_angle[i];
// 			if (fabs(Dtheta) > 0.00001)
// 			{
// 			 Quat angularRotation = Quat(Vec3d(0.0,0.0,1.0), Dtheta);
// 			out[i].getOrientation() = angularRotation * out[i].getOrientation();
// 			out[i].getOrientation().normalize();
// 			old_angle[i] = angle.getValue()[i];
// 			}
            old_integer[i] = integer;
        }
    }
}


template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn)
{
    VecDeriv &out = *dOut.beginEdit();
    const InVecDeriv &in = dIn.getValue();

    out.resize(abscissa.getValue().size());
    bool isV = false;
    const typename In::VecCoord x = mparams->readX(this->fromModel)->getValue();

    if (&in == &(mparams->readV(this->fromModel)->getValue()))
    {
        isV = true;
    }

    for (unsigned int i=0; i<out.size(); i++)
    {
        out[i] = typename Out::Deriv();
        int integer = helper::rfloor(abscissa.getValue()[i]);
        if (integer < 0)
            integer = 0;
        else if (integer > (int)in.size()-2)
            integer = in.size()-2;

        double fraction = abscissa.getValue()[i] - integer;
        //if (fraction > 1.0) fraction = 1.0;
        {
            typename In::Deriv A, B, AB;
            A = in[integer];
            B = in[integer+1];
            AB = B - A;
            getVCenter(out[i]) = A + (AB * fraction);
            if (isV)
                getVCenter(out[i]) += (x[integer+1] - x[integer]) * (velocity.getValue()/lengthElements[integer]);
            getVOrientation(out[i]).clear();
            //out[i].getOrientation() = computeOrientation(AB, out[i].getOrientation());
        }
    }

    dOut.endEdit();
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::applyJT(const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn)
{
    helper::WriteAccessor< InDataVecDeriv > out = dOut;
    helper::ReadAccessor< OutDataVecDeriv > in = dIn;

    const unsigned int pathsize = this->fromModel->getSize();
    out.resize(pathsize);
    for (unsigned int i=0; i<in.size(); i++)
    {
        int integer = helper::rfloor(abscissa.getValue()[i]);
        if (integer < 0) integer = 0;
        else if (integer > (int)pathsize-2) integer = pathsize-2;
        double fraction = abscissa.getValue()[i] - integer;
        //if (fraction > 1.0) fraction = 1.0;
        {
            out[integer] += getVCenter(in[i]) * (1-fraction);
            out[integer+1] += getVCenter(in[i]) * (fraction);
        }
    }
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data< typename In::MatrixDeriv >& /*out*/, const Data< typename Out::MatrixDeriv >& /*in*/)
{
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        if (fabs(velocity.getValue()) > 0.00001)
        {
            helper::vector<Real> ab;
            ab = abscissa.getValue();
            Real s = velocity.getValue() * (Real)this->getContext()->getDt();
//            sout << "abscissa += "<<s<<sendl;
            for(unsigned int i=0; i<abscissa.getValue().size(); i++)
            {
                //ab[i] += s;
                ab[i] = advanceAbscissa(ab[i], s);
                //if (ab[i] > this->fromModel->getSize())
                //    ab[i] = this->fromModel->getSize();
//                sout << "abscissa["<<i<<"] = "<<ab[i]<<sendl;
            }
            if (distNode.getValue() != 0)
            {
                for (unsigned int i=1; i<ab.size(); i++)
                    ab[i] = advanceAbscissa( ab[i-1], distNode.getValue());
            }
            abscissa.setValue(ab);
        }
    }
    else if (sofa::core::objectmodel::MouseEvent::checkEventType(event))
    {
        sofa::core::objectmodel::MouseEvent *ev = static_cast<sofa::core::objectmodel::MouseEvent *>(event);

        switch (ev->getState())
        {
        case sofa::core::objectmodel::MouseEvent::Wheel :
        {
            helper::vector< Real > ab;
            ab = abscissa.getValue();
            for(unsigned int i=0; i<abscissa.getValue().size(); i++)
            {
                ab[i] = (ev->getWheelDelta() > 0) ? ab[i] + step.getValue() : ab[i] - step.getValue();

                if (ab[i] > this->fromModel->getSize())
                    ab[i] = (Real)this->fromModel->getSize();

                if (ab[i] < 0.0)
                    ab[i] = 0.0;
            }
            abscissa.setValue(ab);
        }
        break;

        default:
            break;
        }
    }
    else if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);

        helper::vector<Real> ab;
        helper::vector<Real> ang;
        switch(ev->getKey())
        {
        case 'A':
            velocity.setValue(velocity.getValue()+step.getValue());
            break;
        case 'R':
            velocity.setValue(velocity.getValue()-step.getValue());
            break;
        case 'Q':
            ab = abscissa.getValue();
            for(unsigned int i=0; i<abscissa.getValue().size(); i++)
            {
                ab[i] += step.getValue();
                if (ab[i] > this->fromModel->getSize())
                    ab[i] = (Real)this->fromModel->getSize();
            }
            abscissa.setValue(ab);
            break;
        case 'W':
            ab = abscissa.getValue();
            for(unsigned int i=0; i<abscissa.getValue().size(); i++)
            {
                ab[i] -= step.getValue();
                if (ab[i] < 0.0)
                    ab[i] = 0.0;
            }
            abscissa.setValue(ab);
            break;
        case 'L':
            ang = angle.getValue();
            for(unsigned int i=0; i<angle.getValue().size(); i++)
            {
                ang[i] += angularStep.getValue();
            }
            angle.setValue(ang);
            rotateElements();
            break;
        case 'M':
            ang = angle.getValue();
            for(unsigned int i=0; i<angle.getValue().size(); i++)
            {
                ang[i] -= angularStep.getValue();
            }
            angle.setValue(ang);
            rotateElements();
            break;
        }

    }
}

template <class TIn, class TOut>
void CurveMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings()) return;
    std::vector< sofa::defaulttype::Vector3 > points;
    sofa::defaulttype::Vector3 point;

    const VecCoord& x = this->toModel->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i=0; i<x.size(); i++)
    {
        point = DataTypes::getCPos(x[i]);
        points.push_back(point);
    }
    vparams->drawTool()->drawPoints(points, 5, sofa::defaulttype::Vec<4,float>(1,1,0,1));

}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
