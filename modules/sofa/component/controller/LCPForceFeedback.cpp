/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

using namespace std;

namespace sofa
{
namespace component
{
namespace controller
{


void LCPForceFeedback::init()
{
    this->ForceFeedback::init();
    OmniDriver* driver = context->get<OmniDriver>();

//	BaseObject* object2 = static_cast<BaseObject*>(context->getObject(classid(sofa::component::odesolver::MasterContactSolver)));

    mastersolver = context->get<sofa::component::odesolver::MasterContactSolver>();

    driver->setForceFeedback(this);

    mState = dynamic_cast<MechanicalState<Rigid3dTypes> *> (this->getContext()->getMechanicalState());
    if (!mState)
        std::cerr << "WARNING - LCPForceFeedback has no binding MechanicalState\n";


    if (!mastersolver)
        std::cerr << "WARNING - LCPForceFeedback has no binding MasterContactSolver\n" ;

    lcp = mastersolver->getLCP();

    cout << "init LCPForceFeedback " << driver << " done " << std::endl;
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
        //std::cout<<"new LCP detected"<<std::endl;

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
            int sizeC1 = (*mState->getC())[c1].size();
            RigidTypes::SparseVecDeriv v;
            for(int i = 0; i < sizeC1; i++)
            {
                RigidTypes::SparseDeriv d((*mState->getC())[c1][i].index, (*mState->getC())[c1][i].data);
                v.push_back(d);
            }
            c.push_back(v);
        }

    }
    else
    {
        //std::cout<<"old LCP "<<std::endl;
    }
    lcp_buf = lcp;
    //lcp->wait();
    //lcp->lock();




/////// Copy the constraint buffer /////////
    RigidTypes::VecConst* constraints = &c;
///////////////////////////////////////////

/////// Fordebug /////////
//	if(lcp)
//		std::cout<<"numConst" <<constraints->size()<<std::endl;
//	else
//		std::cout<<"WARNING : LCP is null"<<std::endl;
/////////////////////////

//	std::cout << "LCPForceFeedback::computeForce " << constraints->size() << std::endl;

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

            //cout << "two !" << endl;

            for(unsigned int c1 = 0; c1 < numConstraints; c1++)
            {
                int indexC1 = id_buf[c1];
                int sizeC1 = (*constraints)[c1].size();
                for(int i = 0; i < sizeC1; i++)
                {
                    //cout << "constraint ID :  " << indexC1 << endl;
                    (lcp)->getDfree()[indexC1] += (*constraints)[c1][i].data[0] * dx;
                    (lcp)->getDfree()[indexC1] += (*constraints)[c1][i].data[1] * dy;
                    (lcp)->getDfree()[indexC1] += (*constraints)[c1][i].data[2] * dz;
                    //cout << "data : " << constraints[c1][i].data[0] << " " << constraints[c1][i].data[1] << " " << constraints[c1][i].data[2] << endl;
                }
            }

            //cout << "three !" << endl;

            double tol = lcp->getTolerance();
            int max = 100;

            tol *= 0.001;
            //helper::nlcp_gaussseidel((lcp)->getNbConst(), (lcp)->getDfree(), (lcp)->getW(), (lcp)->getF(), (lcp)->getMu(), tol, max, true);
            helper::nlcp_gaussseidelTimed((lcp)->getNbConst(), (lcp)->getDfree(), (lcp)->getW(), (lcp)->getF(), (lcp)->getMu(), tol, max, true, 0.0008);
            //helper::afficheLCP((lcp)->getDfree(), (lcp)->getW(), (lcp)->getF(),(lcp)->getNbConst());


            //cout << "four !" << endl;

            for(unsigned int c1 = 0; c1 < numConstraints; c1++)
            {
                int indexC1 = id_buf[c1];
                int sizeC1 = (*constraints)[c1].size();
                for(int i = 0; i < sizeC1; i++)
                {
                    (lcp)->getDfree()[indexC1] -= (*constraints)[c1][i].data[0] * dx;
                    (lcp)->getDfree()[indexC1] -= (*constraints)[c1][i].data[1] * dy;
                    (lcp)->getDfree()[indexC1] -= (*constraints)[c1][i].data[2] * dz;
                }
            }

            //cout << "five !" << endl;

            for(unsigned int c1 = 0; c1 < numConstraints; c1++)
            {
                int indexC1 = id_buf[c1];
                if ((lcp)->getF()[indexC1] != 0.0)
                {
                    int sizeC1 = (*constraints)[c1].size();

                    for(int i = 0; i < sizeC1; i++)
                    {
                        force[0] += (*constraints)[c1][i].data * (lcp)->getF()[indexC1];
                    }
                }
            }

            //cout << "six !" << endl;

            fx = force[0][0]*forceCoef.getValue() ;//0.0003;
            fy = force[0][1]*forceCoef.getValue() ;//0.0003;
            fz = force[0][2]*forceCoef.getValue();//0.0003;

            //cout << "seven !" << endl;

            //cout << "haptic forces : " << fx << " " << fy << " " << fz << endl;
            //cout << "forces : " << force << end;
            //cout << "haptic diff : " << DX[0][0] << " " << DX[1][0] << " " << DX[2][0]  << endl;
        }
        //cout << "eight" << endl;

    }

    //cout << "nine !" << endl;
    //lcp->unlock();

};



int lCPForceFeedbackClass = sofa::core::RegisterObject("LCP force feedback for the omni")
        .add< LCPForceFeedback >();

SOFA_DECL_CLASS(LCPForceFeedback)

} // namespace controller
} // namespace component
} // namespace sofa
