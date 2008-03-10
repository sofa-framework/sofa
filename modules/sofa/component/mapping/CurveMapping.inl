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

#include <sofa/component/mapping/CurveMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/rmath.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/simulation/tree/AnimateBeginEvent.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

template <class BasicMapping>
typename CurveMapping<BasicMapping>::Real CurveMapping<BasicMapping>::advanceAbscissa(Real ab, Real dist)
{
    int integer = helper::rfloor(ab);
    if (integer < 0) integer = 0;
    else if (integer > (int)this->lengthElements.size()-1) integer = this->lengthElements.size()-1;
    double fraction = ab - integer;
    while (integer < this->lengthElements.size()-1 && lengthElements[integer] * (1-fraction) < dist)
    {
        dist -= lengthElements[integer] * (1-fraction);
        ++integer;
        fraction = 0;
    }
    fraction += dist / lengthElements[integer];
    return integer + fraction;
}

template <class BasicMapping>
void CurveMapping<BasicMapping>::init()
{
    int n = (int) numNodes.getValue();
    this->toModel->resize(n);

    lengthElements.resize(n-1);
    InVecCoord& x0 = *this->fromModel->getX0();

    for (int i=0; i<n-1; i++)
        lengthElements[i] = (x0[i+1]-x0[i]).norm();

    helper::vector<Real> ab;

    ab = abscissa.getValue();
    ab.resize(n);

    if (stepNode.getValue() != 0)
    {
        for (unsigned int i=1; i<ab.size(); i++)
            ab[i] += ab[i-1] + stepNode.getValue();
    }
    else if (distNode.getValue() != 0)
    {
        for (unsigned int i=1; i<ab.size(); i++)
            ab[i] += advanceAbscissa( ab[i-1], distNode.getValue());
    }

    abscissa.setValue(ab);

    old_integer.resize(this->toModel->getSize());
    fill(old_integer.begin(), old_integer.end(), -1);

    old_angle.resize(this->toModel->getSize());
    fill(old_angle.begin(), old_angle.end(), 0.0);
    /*

     	VecCoord& xto = *this->toModel->getX();
     	InVecCoord& xfrom = *this->fromModel->getX();

    	std::ofstream catheterDataFile("/media/data/sofa-dev/trunk/Sofa/applications/projects/runSofa/rest_position.dat", std::fstream::out | std::fstream::binary);

    	apply(xto, xfrom);

    	catheterDataFile << "BeamModel_Name toto"
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
    this->BasicMapping::init();
}

template <class BasicMapping>
void CurveMapping<BasicMapping>::storeResetState()
{
    reset_abscissa = abscissa.getValue();
}

template <class BasicMapping>
void CurveMapping<BasicMapping>::reset()
{
    abscissa.setValue(reset_abscissa );
    fill(old_integer.begin(), old_integer.end(), -1);
    fill(old_angle.begin(), old_angle.end(), 0.0);
}

Quat computeOrientation(const Vec3d& AB, const Quat& Q)
{
    Vec3d PQ = AB;
    Quat quat = Q;

    Vec3d x = quat.rotate(Vec3d(1,0,0));
    PQ.normalize();

    if (dot(x, PQ) > 0.99)
        return Q;

    Vec3d y;
    double alpha;

    if (dot(x, PQ) < -0.99)
    {
        y = quat.rotate(Vec3d(0,0,1));
        alpha = M_PI;
    }
    else
    {
        y = cross(x, PQ);
        y.normalize();
        alpha = acos(dot(x, PQ));
    }

    Quat qaux = Quat(y, alpha);

    return (qaux * quat);
}

template <class BasicMapping>
void CurveMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(abscissa.getValue().size());
    for (unsigned int i=0; i<out.size(); i++)
    {
        int integer = helper::rfloor(abscissa.getValue()[i]);
        if (integer < 0) integer = 0;
        else if (integer > (int)in.size()-2) integer = in.size()-2;
        double fraction = abscissa.getValue()[i] - integer;
        if (fraction > 1.0) fraction = 1.0;
        {
            InCoord A, B, AB;
            A = in[integer];
            B = in[integer+1];
            AB = B - A;
            out[i].getCenter() = A + (AB * fraction);
            //if (integer != old_integer[i]) // rigid position has changed
            {
                out[i].getOrientation() = computeOrientation(AB, out[i].getOrientation());

            }
            Real Dtheta = angle.getValue()[i] - old_angle[i];
            Quat angularRotation = Quat(Vec3d(0.0,0.0,1.0), Dtheta);
            out[i].getOrientation() = angularRotation * out[i].getOrientation();
            out[i].getOrientation().normalize();
            old_integer[i] = integer;
            old_angle[i] = angle.getValue()[i];
        }
    }
}

template <class BasicMapping>
void CurveMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(abscissa.getValue().size());
    for (unsigned int i=0; i<out.size(); i++)
    {
        out[i] = typename Out::Deriv();
        int integer = helper::rfloor(abscissa.getValue()[i]);
        if (integer < 0) integer = 0;
        else if (integer > (int)in.size()-2) integer = in.size()-2;
        double fraction = abscissa.getValue()[i] - integer;
        if (fraction > 1.0) fraction = 1.0;
        {
            typename In::Deriv A, B, AB;
            A = in[integer];
            B = in[integer+1];
            AB = B - A;
            out[i].getVCenter() = A + (AB * fraction);
            //out[i].getOrientation() = computeOrientation(AB, out[i].getOrientation());
        }
    }
}

template <class BasicMapping>
void CurveMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const unsigned int pathsize = this->fromModel->getSize();
    out.resize(pathsize);
    for (unsigned int i=0; i<in.size(); i++)
    {
        int integer = helper::rfloor(abscissa.getValue()[i]);
        if (integer < 0) integer = 0;
        else if (integer > (int)pathsize-2) integer = pathsize-2;
        double fraction = abscissa.getValue()[i] - integer;
        if (fraction > 1.0) fraction = 1.0;
        {
            out[integer] += in[i].getVCenter() * (1-fraction);
            out[integer+1] += in[i].getVCenter() * (fraction);
        }
    }
}

template <class BaseMapping>
void CurveMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
}

template <class BaseMapping>
void CurveMapping<BaseMapping>::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/*sofa::simulation::tree::AnimateBeginEvent* ev = */dynamic_cast<sofa::simulation::tree::AnimateBeginEvent*>(event))
    {
        if (fabs(velocity.getValue()) > 0.00001)
        {
            helper::vector<Real> ab;
            ab = abscissa.getValue();
            Real s = velocity.getValue(); // * (Real)ev->getDt();
//            std::cout << "abscissa += "<<s<<std::endl;
            for(unsigned int i=0; i<abscissa.getValue().size(); i++)
            {
                //ab[i] += s;
                ab[i] = advanceAbscissa(ab[i], s);
                if (ab[i] > this->fromModel->getSize())
                    ab[i] = this->fromModel->getSize();
//                std::cout << "abscissa["<<i<<"] = "<<ab[i]<<std::endl;
            }
            abscissa.setValue(ab);
        }
    }
    else if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
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
                    ab[i] = this->fromModel->getSize();
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
            break;
        case 'M':
            ang = angle.getValue();
            for(unsigned int i=0; i<angle.getValue().size(); i++)
            {
                ang[i] -= angularStep.getValue();
            }
            angle.setValue(ang);
            break;
        }

    }
}

template <class BaseMapping>
void CurveMapping<BaseMapping>::draw()
{
    glPointSize(5);
    glColor4f (1,1,0,1);
    glBegin (GL_POINTS);
    const VecCoord& x = *this->toModel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::glVertexT(x[i]);
    }
    glEnd();
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
