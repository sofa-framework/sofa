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
#include <sofa/component/controller/LCPForceFeedback.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/mastersolver/MasterContactSolver.h>
#include <sofa/helper/LCPcalc.h>

using namespace std;
using namespace sofa::defaulttype;

namespace sofa
{
namespace component
{
namespace controller
{


void LCPForceFeedback::init()
{
    this->ForceFeedback::init();

//	BaseObject* object2 = static_cast<BaseObject*>(context->getObject(classid(sofa::component::odesolver::MasterContactSolver)));

    mastersolver = context->get<sofa::component::odesolver::MasterContactSolver>();

    mState = dynamic_cast<MechanicalState<Rigid3dTypes> *> (this->getContext()->getMechanicalState());
    if (!mState)
        serr << "LCPForceFeedback has no binding MechanicalState" << sendl;


    if (!mastersolver)
        serr << "LCPForceFeedback has no binding MasterContactSolver" << sendl;

    lcp = mastersolver->getLCP();

    sout << "init LCPForceFeedback done " << sendl;
};

void LCPForceFeedback::computeForce(double x, double y, double z, double /*u*/, double /*v*/, double /*w*/, double /*q*/, double& fx, double& fy, double& fz)
{
    if (!f_activate.getValue())
    {
        return;
    }
    static double mx = (*mState->getX())[0].getCenter()[0];
    static double my = (*mState->getX())[0].getCenter()[1];
    static double mz = (*mState->getX())[0].getCenter()[2];

    static component::odesolver::LCP* lcp_buf = NULL;

    static RigidTypes::VecConst c;
    static std::vector<int> id_buf;

    if (lcp_buf == NULL)
    {
        lcp_buf = mastersolver->getLCP();
        mx = (*mState->getX())[0].getCenter()[0];
        my = (*mState->getX())[0].getCenter()[1];
        mz = (*mState->getX())[0].getCenter()[2];
    }

    lcp = mastersolver->getLCP();

    if (lcp_buf!=lcp)
    {
        //////////////////// NEW LCP //////////////////////
        //sout<<"new LCP detected"<<sendl;

        mx = (*mState->getX())[0].getCenter()[0];
        my = (*mState->getX())[0].getCenter()[1];
        mz = (*mState->getX())[0].getCenter()[2];

        // copy of the constraints that correspond to the lcp in vecConst c
        c.clear();
        id_buf.clear();
        for(unsigned int c1 = 0; c1 < mState->getC()->size(); c1++)
        {
            int indexC1 = mState->getConstraintId()[c1];
            id_buf.push_back(indexC1);
            RigidTypes::SparseVecDeriv v;
            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=(*mState->getC())[c1].data();

            for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                v.add(itConstraint->first, itConstraint->second);
            }
            c.push_back(v);
        }

    }
    else
    {
        //sout<<"old LCP "<<sendl;
    }
    lcp_buf = lcp;
    //lcp->wait();
    //lcp->lock();




/////// Copy the constraint buffer /////////
    RigidTypes::VecConst* constraints = &c;
///////////////////////////////////////////

/////// Fordebug /////////
//	if(lcp)
//		sout<<"numConst" <<constraints->size()<<sendl;
//	else
//		sout<<"WARNING : LCP is null"<<sendl;
/////////////////////////

//	sout << "LCPForceFeedback::computeForce " << constraints->size() << sendl;

    if(lcp)
    {
        if ((lcp)->getMu() > 0.0 && constraints->size())
        {

            //RigidTypes::VecDeriv DX;
            //DX.resize(3);
            const unsigned int numConstraints = constraints->size();
            RigidTypes::VecDeriv force;

            if(!force.size())
                force.resize((*mState->getX()).size());


            //DX[0][0] = x - mx;
            //DX[1][0] = y - my;
            //DX[2][0] = z - mz;

            double dx = (x - mx);
            double dy = (y - my);
            double dz = (z - mz);

            //sout << "two !" << endl;

            for(unsigned int c1 = 0; c1 < numConstraints; c1++)
            {
                int indexC1 = id_buf[c1];
                ConstraintIterator itConstraint;
                std::pair< ConstraintIterator, ConstraintIterator > iter=(*constraints)[c1].data();

                for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
                {
                    //sout << "constraint ID :  " << indexC1 << endl;
                    (lcp)->getDfree()[indexC1] += itConstraint->second[0] * dx;
                    (lcp)->getDfree()[indexC1] += itConstraint->second[1] * dy;
                    (lcp)->getDfree()[indexC1] += itConstraint->second[2] * dz;
                    //sout << "data : " << constraints[c1][i].data[0] << " " << constraints[c1][i].data[1] << " " << constraints[c1][i].data[2] << endl;
                }
            }

            //sout << "three !" << endl;

            double tol = lcp->getTolerance();
            int max = 100;

            tol *= 0.001;
            //helper::nlcp_gaussseidel((lcp)->getNbConst(), (lcp)->getDfree(), (lcp)->getW(), (lcp)->getF(), (lcp)->getMu(), tol, max, true);
            helper::nlcp_gaussseidelTimed((lcp)->getNbConst(), (lcp)->getDfree(), (lcp)->getW(), (lcp)->getF(), (lcp)->getMu(), tol, max, true, 0.0008);
            //helper::afficheLCP((lcp)->getDfree(), (lcp)->getW(), (lcp)->getF(),(lcp)->getNbConst());


            //sout << "four !" << endl;

            for(unsigned int c1 = 0; c1 < numConstraints; c1++)
            {
                int indexC1 = id_buf[c1];


                ConstraintIterator itConstraint;
                std::pair< ConstraintIterator, ConstraintIterator > iter=(*constraints)[c1].data();

                for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
                {
                    (lcp)->getDfree()[indexC1] -= itConstraint->second[0] * dx;
                    (lcp)->getDfree()[indexC1] -= itConstraint->second[1] * dy;
                    (lcp)->getDfree()[indexC1] -= itConstraint->second[2] * dz;
                }
            }

            //sout << "five !" << endl;

            for(unsigned int c1 = 0; c1 < numConstraints; c1++)
            {
                int indexC1 = id_buf[c1];
                if ((lcp)->getF()[indexC1] != 0.0)
                {
                    ConstraintIterator itConstraint;
                    std::pair< ConstraintIterator, ConstraintIterator > iter=(*constraints)[c1].data();

                    for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
                    {
                        force[0] += itConstraint->second * (lcp)->getF()[indexC1];
                    }
                }
            }

            //sout << "six !" << endl;

            fx = force[0][0]*forceCoef.getValue() ;//0.0003;
            fy = force[0][1]*forceCoef.getValue() ;//0.0003;
            fz = force[0][2]*forceCoef.getValue();//0.0003;

            //sout << "seven !" << endl;

            //sout << "haptic forces : " << fx << " " << fy << " " << fz << endl;
            //sout << "forces : " << force << end;
            //sout << "haptic diff : " << DX[0][0] << " " << DX[1][0] << " " << DX[2][0]  << endl;
        }
        //sout << "eight" << endl;

    }

    //sout << "nine !" << endl;
    //lcp->unlock();

};



int lCPForceFeedbackClass = sofa::core::RegisterObject("LCP force feedback for the omni")
        .add< LCPForceFeedback >();

SOFA_DECL_CLASS(LCPForceFeedback)

} // namespace controller
} // namespace component
} // namespace sofa
