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
#define SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_CPP

#include "UncoupledConstraintCorrection.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/mass/UniformMass.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;

template<>
SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::init()
{
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());

    double dt = this->getContext()->getDt();

    VecReal usedComp;

    if (compliance.getValue().size() != 7)
    {
        using sofa::component::mass::UniformMass;
        using sofa::defaulttype::Rigid3Types;
        using sofa::defaulttype::Rigid3Mass;
        using sofa::simulation::Node;

        Node *node = dynamic_cast< Node * >(getContext());
        const Rigid3Mass *massValue;
        bool destroyMassValue = false;

        //Should use the BaseMatrix API to get the Mass
        //void getElementMass(unsigned int index, defaulttype::BaseMatrix *m)
        if (node != NULL)
        {
            core::behavior::BaseMass *m = node->mass;
            UniformMass< Rigid3Types, Rigid3Mass > *um = dynamic_cast< UniformMass< Rigid3Types, Rigid3Mass >* > (m);

            if (um)
                massValue = &(um->getMass());
            else
                serr << "WARNING : no mass found" << sendl;
        }
        else
        {
            massValue = new Rigid3Mass();
            destroyMassValue = true;
            serr << "\n WARNING : node is not found => massValue could be false in getCompliance function" << sendl;
        }

        const double dt2 = dt * dt;

        usedComp.push_back(dt2 / massValue->mass);
        usedComp.push_back(dt2 * massValue->invInertiaMassMatrix[0][0]);
        usedComp.push_back(dt2 * massValue->invInertiaMassMatrix[0][1]);
        usedComp.push_back(dt2 * massValue->invInertiaMassMatrix[0][2]);
        usedComp.push_back(dt2 * massValue->invInertiaMassMatrix[1][1]);
        usedComp.push_back(dt2 * massValue->invInertiaMassMatrix[1][2]);
        usedComp.push_back(dt2 * massValue->invInertiaMassMatrix[2][2]);
        compliance.setValue(usedComp);

        if (destroyMassValue)
        {
            delete massValue;
        }
    }
    else
    {
        sout << "COMPLIANCE VALUE FOUND" << sendl;
    }
}


template<>
SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::getCompliance(defaulttype::BaseMatrix *W)
{
    const MatrixDeriv& constraints = *mstate->getC();

    Deriv weightedNormal;
    Deriv comp_wN;

    VecReal usedComp = compliance.getValue();

    MatrixDerivRowConstIterator rowIt = constraints.begin();
    MatrixDerivRowConstIterator rowItEnd = constraints.end();

    while (rowIt != rowItEnd)
    {
        const unsigned int indexCurRowConst = rowIt.index();

        MatrixDerivColConstIterator colIt = rowIt.begin();
        MatrixDerivColConstIterator colItEnd = rowIt.end();

        while (colIt != colItEnd)
        {
            unsigned int dof = colIt.index();
            Deriv n = colIt.val();

#ifdef DEBUG
            std::cout << "    [ " << dof << "]=" << n << std::endl;
#endif

            weightedNormal.getVCenter() = n.getVCenter();
            weightedNormal.getVOrientation() = n.getVOrientation();

            // compliance * weightedNormal
            comp_wN.getVCenter() = weightedNormal.getVCenter() * usedComp[0];

            const double wn3 = weightedNormal[3];
            const double wn4 = weightedNormal[4];
            const double wn5 = weightedNormal[5];

            comp_wN[3] =  usedComp[1] * wn3 +  usedComp[2] * wn4 +  usedComp[3] * wn5;
            comp_wN[4] =  usedComp[2] * wn3 +  usedComp[4] * wn4 +  usedComp[5] * wn5;
            comp_wN[5] =  usedComp[3] * wn3 +  usedComp[5] * wn4 +  usedComp[6] * wn5;

            // serr << " - " << weightedNormal << sendl;
            // InvM_wN = weightedNormal / (*massValue);
            // InvM_wN *= dt * dt ;

            MatrixDerivRowConstIterator rowIt2 = rowIt;

            while (rowIt2 != rowItEnd)
            {
                const unsigned int indexCurColConst = rowIt2.index();

                MatrixDerivColConstIterator colIt2 = rowIt2.begin();
                MatrixDerivColConstIterator colIt2End = rowIt2.end();

                while (colIt2 != colIt2End)
                {
                    unsigned int dof2 = colIt2.index();
                    Deriv n2 = colIt2.val();

                    if (dof == dof2)
                    {
                        double w = n2 * comp_wN;

                        W->add(indexCurRowConst, indexCurColConst, w);

                        if (indexCurRowConst != indexCurColConst)
                            W->add(indexCurColConst, indexCurRowConst, w);
                    }

                    ++colIt2;
                }

                ++rowIt2;
            }

            ++colIt;
        }

        ++rowIt;
    }

#ifdef DEBUG
    // std::cout << "Wnew = " << Wnew << std::endl;
#endif
}

/*
template<>
SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection<defaulttype::Rigid3Types>::getCompliance(defaulttype::BaseMatrix*W)
{
	const VecConst& constraints = *mstate->getC();
	Deriv weighedNormal;
	Deriv Comp_wN;

	//serr<<"getCompliance : ";
	// debug //
	//sofa::component::linearsolver::FullMatrix<double> Wnew;
	//Wnew.resize((int)W->rowSize(), (int) W->colSize())	;

	unsigned int numConstraints = constraints.size();
	//serr<<" numConstraints :"<<numConstraints;
	VecReal UsedComp = compliance.getValue();
	//serr<<" UsedComp ok"<<sendl;

	#ifdef DEBUG
	std::cout<<"numConstraints ="<<numConstraints<<std::endl;
	#endif

	for(unsigned int curRowConst=0; curRowConst < numConstraints; curRowConst++)
	{
		int indexCurRowConst = mstate->getConstraintId()[curRowConst];

#ifdef DEBUG
		std::cout<<"constraint["<<curRowConst<<"] : index :  "<<indexCurRowConst<<std::endl;
#endif

		ConstraintIterator itConstraint;
		std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[curRowConst].data();

		for (itConstraint=iter.first;itConstraint!=iter.second;itConstraint++)
		{
			unsigned int dof = itConstraint->first;
			Deriv n = itConstraint->second;
#ifdef DEBUG
			std::cout<<"    [ "<<dof<<"]="<<n<<std::endl;
#endif
			weighedNormal.getVCenter() = n.getVCenter(); // weighed normal
			weighedNormal.getVOrientation() = n.getVOrientation();

			// compliance * weighedNormal
			Comp_wN.getVCenter() = weighedNormal.getVCenter() * UsedComp[0];

			Comp_wN[3] =  UsedComp[1] * weighedNormal[3] +  UsedComp[2] * weighedNormal[4] +  UsedComp[3] * weighedNormal[5];
			Comp_wN[4] =  UsedComp[2] * weighedNormal[3] +  UsedComp[4] * weighedNormal[4] +  UsedComp[5] * weighedNormal[5];
			Comp_wN[5] =  UsedComp[3] * weighedNormal[3] +  UsedComp[5] * weighedNormal[4] +  UsedComp[6] * weighedNormal[5];
			//serr<<" - "<<weighedNormal<<sendl;
			//InvM_wN = weighedNormal / (*massValue);
			//InvM_wN *= dt*dt ;

			int indexCurColConst;

			for(unsigned int curColConst = curRowConst; curColConst < numConstraints; curColConst++)
			{
				indexCurColConst = mstate->getConstraintId()[curColConst];

				ConstraintIterator itConstraint2;
				std::pair< ConstraintIterator, ConstraintIterator > iter2=constraints[curColConst].data();

				for (itConstraint2=iter2.first;itConstraint2!=iter2.second;itConstraint2++)
				{
					unsigned int dof2 = itConstraint2->first;
					Deriv n2 = itConstraint2->second;
					//W[indexCurRowConst][indexCurColConst] +=  constraints[curColConst][j].data * InvM_wN;

					if (dof == dof2)
					{
						double w =  n2 * Comp_wN;
						// debug //
						//Wnew.add(indexCurRowConst, indexCurColConst, w);
						W->add(indexCurRowConst, indexCurColConst, w);
						if (indexCurRowConst != indexCurColConst)
							W->add(indexCurColConst, indexCurRowConst, w);
					}
				}
			}
		}

	}
#ifdef DEBUG
	//std::cout<<" Wnew = "<<Wnew<<std::endl;
#endif
}
*/

template<>
SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::getComplianceMatrix(defaulttype::BaseMatrix *m) const
{
    const VecReal &comp = compliance.getValue();
    const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
    const unsigned int numDofs = comp.size() / 7;

    m->resize(dimension * numDofs, dimension * numDofs);

    /// @TODO Optimization
    for (unsigned int d = 0; d < numDofs; ++d)
    {
        const unsigned int d6 = 6 * d;
        const unsigned int d7 = 7 * d;
        const SReal invM = comp[d7];

        m->set(d6, d6, invM);
        m->set(d6 + 1, d6 + 1, invM);
        m->set(d6 + 2, d6 + 2, invM);

        m->set(d6 + 3, d6 + 3, comp[d7 + 1]);

        m->set(d6 + 3, d6 + 4, comp[d7 + 2]);
        m->set(d6 + 4, d6 + 3, comp[d7 + 2]);

        m->set(d6 + 3, d6 + 5, comp[d7 + 3]);
        m->set(d6 + 5, d6 + 3, comp[d7 + 3]);

        m->set(d6 + 4, d6 + 4, comp[d7 + 4]);

        m->set(d6 + 4, d6 + 5, comp[d7 + 5]);
        m->set(d6 + 5, d6 + 4, comp[d7 + 5]);

        m->set(d6 + 5, d6 + 5, comp[d7 + 6]);
    }

    /*
    for (unsigned int l=0;l<s;++l)
    {
    	for (unsigned int c=0;c<s;++c)
    	{
    		if (l==c)
    			m->set(l,c,comp[l]);
    		else
    			m->set(l,c,0);
    	}
    }
    */
}


template<>
SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::applyContactForce(const defaulttype::BaseVector *f)
{
    VecDeriv& force = *mstate->getExternalForces();
    const MatrixDeriv& constraints = *mstate->getC();

    unsigned int dof;
    Deriv weightedNormal;

    VecReal usedComp = compliance.getValue();

    force.resize((*mstate->getX()).size());

    MatrixDerivRowConstIterator rowIt = constraints.begin();
    MatrixDerivRowConstIterator rowItEnd = constraints.end();

    while (rowIt != rowItEnd)
    {
        const double fC1 = f->element(rowIt.index());

        if (fC1 != 0.0)
        {
            MatrixDerivColConstIterator colIt = rowIt.begin();
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            while (colIt != colItEnd)
            {
                dof = colIt.index();
                weightedNormal = colIt.val();

                force[dof].getVCenter() += weightedNormal.getVCenter() * fC1;
                force[dof].getVOrientation() += weightedNormal.getVOrientation() * fC1;

                ++colIt;
            }
        }

        ++rowIt;
    }

    // std::cout << "force -- resultante: " << force[0].getVCenter() << "   -- moment: " << force[0].getVOrientation() << std::endl;

    VecDeriv& dx = *mstate->getDx();
    VecCoord& x = *mstate->getX();
    VecDeriv& v = *mstate->getV();
    VecDeriv& v_free = *mstate->getVfree();
    VecCoord& x_free = *mstate->getXfree();

    const double dt = this->getContext()->getDt();

    // Euler integration... will be done in the "integrator" as soon as it exists !
    dx.resize(v.size());

    for (unsigned int i = 0; i < dx.size(); i++)
    {
        x[i] = x_free[i];
        v[i] = v_free[i];

        // compliance * force
        dx[i].getVCenter() = force[i].getVCenter() * usedComp[0];
        dx[i][3] =  usedComp[1] * force[i][3] +  usedComp[2] * force[i][4] +  usedComp[3] * force[i][5];
        dx[i][4] =  usedComp[2] * force[i][3] +  usedComp[4] * force[i][4] +  usedComp[5] * force[i][5];
        dx[i][5] =  usedComp[3] * force[i][3] +  usedComp[5] * force[i][4] +  usedComp[6] * force[i][5];
        dx[i] *= (1.0 / dt);
        v[i] += dx[i];
        dx[i] *= dt;
        x[i] += dx[i];
    }

//	std::cout << "dx -- resultante: " << dx[0].getVCenter() << "   -- moment: " << dx[0].getVOrientation() << std::endl;
//	std::cout << "UncoupledConstraintCorrection<defaulttype::Rigid3Types>: x = " << x << " \n        xfree = " << x_free << std::endl;
//	simulation::tree::MechanicalPropagateAndAddDxVisitor(dx).execute(this->getContext());
////////////////////////////////////////////////////////////////////
}


//template<>
//SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::applyContactForce(const defaulttype::BaseVector *f)
//{
//	VecDeriv& force = *mstate->getExternalForces();
//	const VecConst& constraints = *mstate->getC();
//	Deriv weighedNormal;
//
//	VecReal UsedComp = compliance.getValue();
//	double dt = this->getContext()->getDt();
//
//	//force.resize(0);
//	//force.resize(1);
//	//force[0] = Deriv();
//	force.resize((*mstate->getX()).size());
//
//	int numConstraints = constraints.size();
//
//	for(int c1 = 0; c1 < numConstraints; c1++)
//	{
//		int indexC1 = mstate->getConstraintId()[c1];
//		double fC1 = f->element(indexC1);
//
//		if (fC1 != 0.0)
//		{
//			ConstraintIterator itConstraint;
//			std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();
//
//			for (itConstraint=iter.first;itConstraint!=iter.second;itConstraint++)
//			{
//				unsigned int dof = itConstraint->first;
//				weighedNormal = itConstraint->second; // weighted normal
//				force[dof].getVCenter() += weighedNormal.getVCenter() * fC1;
//				force[dof].getVOrientation() += weighedNormal.getVOrientation() * fC1;
//			}
//		}
//	}
//
////	std::cout<<"force -- resultante: "<<force[0].getVCenter()<<"   -- moment: "<<force[0].getVOrientation()<<std::endl;
//
//	VecDeriv& dx = *mstate->getDx();
//	VecCoord& x = *mstate->getX();
//	VecDeriv& v = *mstate->getV();
//	VecDeriv& v_free = *mstate->getVfree();
//	VecCoord& x_free = *mstate->getXfree();
//
//	// Euler integration... will be done in the "integrator" as soon as it exists !
//	dx.resize(v.size());
//
//	for (unsigned int i=0; i<dx.size(); i++)
//	{
//		x[i] = x_free[i];
//		v[i] = v_free[i];
//
//		// compliance * force
//		dx[i].getVCenter() = force[i].getVCenter() * UsedComp[0];
//		dx[i][3] =  UsedComp[1] * force[i][3] +  UsedComp[2] * force[i][4] +  UsedComp[3] * force[i][5];
//		dx[i][4] =  UsedComp[2] * force[i][3] +  UsedComp[4] * force[i][4] +  UsedComp[5] * force[i][5];
//		dx[i][5] =  UsedComp[3] * force[i][3] +  UsedComp[5] * force[i][4] +  UsedComp[6] * force[i][5];
//		dx[i] *= (1.0/dt);
//		v[i] += dx[i];
//		dx[i] *= dt;
//		x[i] += dx[i];
//	}
//
////	std::cout<<"dx -- resultante: "<<dx[0].getVCenter()<<"   -- moment: "<<dx[0].getVOrientation()<<std::endl;
////	std::cout<<"UncoupledConstraintCorrection<defaulttype::Rigid3Types>: x= "<<x <<" \n        xfree = "<<x_free<<std::endl;
////	simulation::tree::MechanicalPropagateAndAddDxVisitor(dx).execute(this->getContext());
//////////////////////////////////////////////////////////////////////
//}


template<>
SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::setConstraintDForce(double * df, int begin, int end, bool update)
{
    const MatrixDeriv& constraints = *mstate->getC();
    const VecReal usedComp = compliance.getValue();

    if (!update)
        return;

    for (int id = begin; id <= end; id++)
    {
        int c = id_to_localIndex[id];

        MatrixDerivRowConstIterator curConstraint = constraints.readLine(c);

        if (curConstraint != constraints.end())
        {
            MatrixDerivColConstIterator colIt = curConstraint.begin();
            MatrixDerivColConstIterator colItEnd = curConstraint.end();

            while (colIt != colItEnd)
            {
                Deriv n = colIt.val();
                unsigned int dof = colIt.index();

                constraint_force[dof] += n * df[id];

                Deriv dx;
                dx.getVCenter() = constraint_force[dof].getVCenter() * compliance.getValue()[0];

                defaulttype::Vec3d wrench = constraint_force[dof].getVOrientation();
                dx.getVOrientation()[0] = usedComp[1] * wrench[0] + usedComp[2] * wrench[1] + usedComp[3] * wrench[2];
                dx.getVOrientation()[1] = usedComp[2] * wrench[0] + usedComp[4] * wrench[1] + usedComp[5] * wrench[2];
                dx.getVOrientation()[2] = usedComp[3] * wrench[0] + usedComp[5] * wrench[1] + usedComp[6] * wrench[2];

                constraint_disp[dof] = dx;

                ++colIt;
            }
        }
    }
}


//template<>
//SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::setConstraintDForce(double * df, int begin, int end, bool update)
//{
//	const VecConst& constraints = *mstate->getC();
//	if (!update)
//		return;
//	// debug
//	//if (end<6)
//	//	std::cout<<"addDf - df["<<begin<<" to "<<end<<"] ="<< df[begin] << " " << df[begin+1] << " "<< df[begin+2] << std::endl;
//
//	for ( int id_=begin; id_<=end; id_++)
//	{
//		int c = id_to_localIndex[id_];
//		ConstraintIterator itConstraint;
//		std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c].data();
//		for (itConstraint=iter.first;itConstraint!=iter.second;itConstraint++)
//		{
//			Deriv n = itConstraint->second;
//			unsigned int dof = itConstraint->first;
//
//			constraint_force[dof] += n * df[id_];
//
//			Deriv DX;
//			DX.getVCenter() = constraint_force[dof].getVCenter() * compliance.getValue()[0];
//			defaulttype::Vec3d wrench = constraint_force[dof].getVOrientation();
//			DX.getVOrientation()[0] = compliance.getValue()[1]*wrench[0] +  compliance.getValue()[2]*wrench[1] + compliance.getValue()[3]*wrench[2] ;
//			DX.getVOrientation()[1] = compliance.getValue()[2]*wrench[0] +  compliance.getValue()[4]*wrench[1] + compliance.getValue()[5]*wrench[2] ;
//			DX.getVOrientation()[2] = compliance.getValue()[3]*wrench[0] +  compliance.getValue()[5]*wrench[1] + compliance.getValue()[6]*wrench[2] ;
//
//			constraint_disp[dof] = DX;
//		}
//	}
//}


///////////////////// ATTENTION : passer un indice début - fin (comme pour force et déplacement) pour calculer le block complet
///////////////////// et pas uniquement la diagonale.
template<>
SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
{
    const MatrixDeriv& constraints = *mstate->getC();
    const VecReal usedComp = compliance.getValue();

    Deriv weightedNormal, C_n;

    for (int id1 = begin; id1 <= end; id1++)
    {
        int c1 = id_to_localIndex[id1];

        MatrixDerivRowConstIterator curConstraint = constraints.readLine(c1);

        if (curConstraint != constraints.end())
        {
            MatrixDerivColConstIterator colIt = curConstraint.begin();
            MatrixDerivColConstIterator colItEnd = curConstraint.end();

            while (colIt != colItEnd)
            {
                weightedNormal = colIt.val();
                unsigned int dof1 = colIt.index();

                C_n.getVCenter() = weightedNormal.getVCenter() * compliance.getValue()[0];
                defaulttype::Vec3d wrench = weightedNormal.getVOrientation() ;

                C_n.getVOrientation()[0] = usedComp[1] * wrench[0] + usedComp[2] * wrench[1] + usedComp[3] * wrench[2];
                C_n.getVOrientation()[1] = usedComp[2] * wrench[0] + usedComp[4] * wrench[1] + usedComp[5] * wrench[2];
                C_n.getVOrientation()[2] = usedComp[3] * wrench[0] + usedComp[5] * wrench[1] + usedComp[6] * wrench[2];

                for (int id2 = id1; id2 <= end; id2++)
                {
                    int c2 = id_to_localIndex[id2];

                    MatrixDerivRowConstIterator curConstraint2 = constraints.readLine(c2);

                    if (curConstraint2 != constraints.end())
                    {
                        MatrixDerivColConstIterator colIt2 = curConstraint2.begin();
                        MatrixDerivColConstIterator colItEnd2 = curConstraint2.end();

                        while (colIt2 != colItEnd2)
                        {
                            unsigned int dof2 = colIt2.index();

                            if (dof1 == dof2)
                            {
                                Deriv n2 = colIt2.val();
                                double w = n2 * C_n;

                                W->add(id1, id2, w);

                                if (id1 != id2)
                                    W->add(id2, id1, w);
                            }

                            ++colIt2;
                        }
                    }
                }

                ++colIt;
            }
        }
    }
}


/////////////////////// ATTENTION : passer un indice début - fin (comme pour force et déplacement) pour calculer le block complet
/////////////////////// et pas uniquement la diagonale.
//template<>
//SOFA_COMPONENT_CONSTRAINTSET_API void UncoupledConstraintCorrection< defaulttype::Rigid3Types >::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
//{
//	//std::cout<<"UncoupledConstraintCorrection<defaulttype::Rigid3Types>::getBlockDiagonalCompliance"<<std::endl;
//
//	const VecConst& constraints = *mstate->getC();
//
//	//std::cout<<" begin = "<<begin<<"  - end = "<<std::endl;
//
//	Deriv weighedNormal, C_n;
//
//	//std::cerr<<" weighedNormal, C_n "<<std::endl;
//
//	for (int id1=begin; id1<=end; id1++)
//	{
//		//std::cerr<<"constraint : "<<id1;
//		int c1 = id_to_localIndex[id1];
//		//std::cerr<<" local index : "<<c1<<std::endl;
//
//		ConstraintIterator itConstraint1;
//		std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();
//
//		for (itConstraint1=iter.first;itConstraint1!=iter.second;itConstraint1++)
//		{
//			weighedNormal = itConstraint1->second;
//			unsigned int dof1 = itConstraint1->first;
//
//			C_n.getVCenter() = weighedNormal.getVCenter() * compliance.getValue()[0];
//			defaulttype::Vec3d wrench = weighedNormal.getVOrientation() ;
//			C_n.getVOrientation()[0] = compliance.getValue()[1]*wrench[0] +  compliance.getValue()[2]*wrench[1] + compliance.getValue()[3]*wrench[2] ;
//			C_n.getVOrientation()[1] = compliance.getValue()[2]*wrench[0] +  compliance.getValue()[4]*wrench[1] + compliance.getValue()[5]*wrench[2] ;
//			C_n.getVOrientation()[2] = compliance.getValue()[3]*wrench[0] +  compliance.getValue()[5]*wrench[1] + compliance.getValue()[6]*wrench[2] ;
//
//			//std::cout<<"C_n : "<<C_n<<std::endl;
//
//			for (int id2= id1; id2<=end; id2++)
//			{
//				int c2 = id_to_localIndex[id2];
//				ConstraintIterator itConstraint2;
//				std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c2].data();
//				for (itConstraint2=iter.first;itConstraint2!=iter.second;itConstraint2++)
//				{
//
//					unsigned int dof2 = itConstraint2->first;
//
//					if (dof1 == dof2)
//					{
//						Deriv n2 = itConstraint2->second;
//						double w = n2 * C_n;
//						// debug
//						//std::cout<<"W("<<id1<<","<<id2<<") += "<< w ;
//						W->add(id1, id2, w);
//						if (id1 != id2)
//							W->add(id2, id1, w);
//					}
//				}
//			}
//		}
//	}
//}


SOFA_DECL_CLASS(UncoupledConstraintCorrection)

int UncoupledConstraintCorrectionClass = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
#ifndef SOFA_FLOAT
        .add< UncoupledConstraintCorrection< Vec1dTypes > >()
//.add< UncoupledConstraintCorrection< Vec2dTypes > >()
        .add< UncoupledConstraintCorrection< Vec3dTypes > >()
//.add< UncoupledConstraintCorrection< Vec6dTypes > >()
//.add< UncoupledConstraintCorrection< Rigid2dTypes > >()
        .add< UncoupledConstraintCorrection< Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< UncoupledConstraintCorrection< Vec1fTypes > >()
//.add< UncoupledConstraintCorrection< Vec2fTypes > >()
        .add< UncoupledConstraintCorrection< Vec3fTypes > >()
//.add< UncoupledConstraintCorrection< Vec6fTypes > >()
//.add< UncoupledConstraintCorrection< Rigid2fTypes > >()
        .add< UncoupledConstraintCorrection< Rigid3fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec1dTypes >;
//template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec2dTypes >;
template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec3dTypes >;
//template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec6dTypes >;
//template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Rigid2dTypes >;
template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec1fTypes >;
//template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec2fTypes >;
template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec3fTypes >;
//template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Vec6fTypes >;
//template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Rigid2fTypes >;
template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection< Rigid3fTypes >;
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa
