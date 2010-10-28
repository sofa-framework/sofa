/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINTSET_STOPPERCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_STOPPERCONSTRAINT_INL

#include <sofa/component/constraintset/StopperConstraint.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>
namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
void StopperConstraint<DataTypes>::init()
{
    assert(this->mstate);

    this->getContext()->get(ode_integrator);

    if(ode_integrator!= NULL)
        std::cout<<"ode_integrator found named :"<< ode_integrator->getName()<<std::endl;
    else
        std::cout<<"no ode_integrator found"<<std::endl;

    helper::WriteAccessor<Data<VecCoord> > xData = *this->mstate->write(core::VecCoordId::position());
    VecCoord& x = xData.wref();
    if (x[index.getValue()].x() < min.getValue())
        x[index.getValue()].x() = (Real) min.getValue();
    if (x[index.getValue()].x() > max.getValue())
        x[index.getValue()].x() = (Real) max.getValue();
}

template<class DataTypes>
void StopperConstraint<DataTypes>::buildConstraintMatrix(unsigned int &constraintId, core::ConstMultiVecCoordId)
{
    int tm;
    Coord cx = Coord(1.0);

    tm = index.getValue();

    assert(this->mstate);

    helper::WriteAccessor<Data<MatrixDeriv> > cData = *this->mstate->write(core::MatrixDerivId::holonomicC());
    MatrixDeriv& c = cData.wref();

    cid = constraintId;
    constraintId += 1;

    c.writeLine(cid).addCol(tm, cx);
}

template<class DataTypes>
void StopperConstraint<DataTypes>::getConstraintValue(defaulttype::BaseVector* v, bool freeMotion)
{
    if (!freeMotion)
        sout<<"WARNING has to be implemented for method based on non freeMotion"<<sendl;

    if (freeMotion)
    {
        dfree = (*this->mstate->getXfree())[index.getValue()];
    }
    else
    {
        serr<<"WARNING: StopperConstraint with no freeMotion not implemented "<<sendl;
        /*
        double positionFactor = ode_integrator->getIntegrationFactor(0, 0);
        double velocityFactor = ode_integrator->getIntegrationFactor(1, 0);
        std::cout<<"dt found = "<<dt<<std::endl;

        dfree = (*this->object2->getX())[m2.getValue()] * positionFactor + (*this->object2->getV())[m2.getValue()]*velocityFactor
        	  - (*this->object1->getX())[m1.getValue()] * positionFactor - (*this->object1->getV())[m1.getValue()]*velocityFactor  ;

        dt = 1.0; // ode_integrator->getSolutionIntegrationFactor(0) * 2;
        */
    }

    v->set(cid, dfree[0]);
}

//template<class DataTypes>
//void StopperConstraint<DataTypes>::getConstraintId(long* id, unsigned int &offset)
//{
//	id[offset++] = cid;
//}

#ifdef SOFA_DEV
template<class DataTypes>
void StopperConstraint<DataTypes>::getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
//	resTab[offset] = new BilateralConstraintResolution3Dof();
//	offset += 3;

    for(int i=0; i<1; i++)
        resTab[offset++] = new StopperConstraintResolution1Dof(min.getValue(), max.getValue());
}
#endif

template<class DataTypes>
void StopperConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowInteractionForceFields()) return;

    /*
    	glDisable(GL_LIGHTING);
    	glPointSize(10);
    	glBegin(GL_POINTS);
    	glColor4f(1,0,1,1);
    	helper::gl::glVertexT((*this->object1->getX())[m1.getValue()]);
    	helper::gl::glVertexT((*this->object2->getX())[m2.getValue()]);
    	glEnd();
    	glPointSize(1);
    */
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
