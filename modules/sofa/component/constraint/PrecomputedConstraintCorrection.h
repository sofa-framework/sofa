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
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTCORRECTION_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTCORRECTION_H

#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>

#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::core;
using namespace sofa::core::componentmodel;
using namespace sofa::defaulttype;
/**
 *  \brief Component computing contact forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class PrecomputedConstraintCorrection : public componentmodel::behavior::BaseConstraintCorrection
{
public:
    typedef TDataTypes DataTypes;
    typedef typename componentmodel::behavior::BaseConstraintCorrection Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename defaulttype::SparseConstraint<Deriv> SparseConstraint;
    typedef typename SparseConstraint::const_data_iterator ConstConstraintIterator;
    typedef typename SparseConstraint::data_iterator ConstraintIterator;
    typedef typename DataTypes::SparseVecDeriv Const;

    /// element rotation matrix
    typedef typename Coord::value_type Real;
    typedef MatNoInit<3, 3, Real> Transformation;

    bool   _rotations;
    DataPtr<bool> f_rotations;
    bool   _restRotations;
    DataPtr<bool> f_restRotations;
    Data<bool> recompute;
    Data<std::string> filePrefix;

    PrecomputedConstraintCorrection(behavior::MechanicalState<DataTypes> *mm = NULL);

    virtual ~PrecomputedConstraintCorrection();

    virtual void bwdInit();

    /// Retrieve the associated MechanicalState
    behavior::MechanicalState<DataTypes>* getMState() { return mstate; }

    virtual void getCompliance(defaulttype::BaseMatrix* W);
    virtual void applyContactForce(const defaulttype::BaseVector *f);


    virtual void rotateConstraints();
    virtual void rotateResponse();
    virtual void resetContactForce();

    virtual void draw();


    // new API for non building the constraint system during solving process //

    virtual void resetForUnbuiltResolution(double * f, std::list<int>& /*renumbering*/)  ;

    virtual bool hasConstraintNumber(int index) ;  // virtual ???

    virtual void addConstraintDisplacement(double *d, int begin,int end) ;

    virtual void setConstraintDForce(double *df, int begin, int end, bool update) ;

    virtual void getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end) ;
    /////////////////////////////////////////////////////////////////////////////////

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const PrecomputedConstraintCorrection<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }


protected:
    behavior::MechanicalState<DataTypes> *mstate;

    struct InverseStorage
    {
        Real* data;
        int nbref;
        InverseStorage() : data(NULL), nbref(0) {}
    };
    std::string invName;
    InverseStorage* invM;
    const Real* appCompliance;

    static std::map<std::string, InverseStorage>& getInverseMap()
    {
        static std::map<std::string, InverseStorage> registry;
        return registry;
    }

    static InverseStorage* getInverse(std::string name);

    static void releaseInverse(std::string name, InverseStorage* inv);

    unsigned int nbRows, nbCols, dof_on_node, nbNodes;
    helper::vector<int> _indexNodeSparseCompliance;
    helper::vector<Deriv> _sparseCompliance;
    Real Fbuf[6], DXbuf;



    /* optimization : buf of result = Compliance * VecConst on a sparse structure
    	typedef struct {
    		Deriv Cn;
    		int nodeId;
    		int constraintId;
    	}SparseCompliance;

    	SparseCompliance _sparseCompliance[10000];
    	*/

    //Deriv **_sparseCompliance;


    // new :  for non building the constraint system during solving process //
    //VecDeriv constraint_disp, constraint_force;
    //std::list<int> constraint_dofs;		// list of indices of each point which is involve with constraint
    helper::vector<int> id_to_localIndex;	// table that gives the local index of a constraint given its id
    sofa::helper::vector<unsigned int>* localConstraintId;
    linearsolver::FullMatrix<Real> localW;
    double* constraint_force;

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
