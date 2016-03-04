/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

//-------------------------------------------------------------------------
//						--   Description   --
//	LMLConstraint imports from a LML file (see LMLReader class) imposed displacements,
//  including translations and fixed points, and traduces it to sofa constraints.
//  It inherits from Constraint sofa core class.
//-------------------------------------------------------------------------

#ifndef SOFAPML_LMLCONSTRAINT_H
#define SOFAPML_LMLCONSTRAINT_H


#include "sofa/core/behavior/Constraint.h"
#include "sofa/core/behavior/MechanicalState.h"
#include "sofa/core/visual/VisualModel.h"
#include <SofaPML/config.h>

#include <vector>
#include <map>

#include <Loads.h>


namespace sofa
{

namespace filemanager
{

namespace pml
{

//using namespace sofa::core;
//using namespace sofa::core::behavior;
//using namespace std;

template<class DataTypes>
class LMLConstraint : public sofa::core::behavior::Constraint<DataTypes> //, public sofa::core::VisualModel
{
    SOFA_CLASS(SOFA_TEMPLATE(LMLConstraint, DataTypes), SOFA_TEMPLATE(sofa::core::behavior::Constraint, DataTypes));
public :
    ///template types
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord::iterator VecCoordIterator;
    typedef typename DataTypes::VecDeriv::iterator VecDerivIterator;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

    ///constructor
    LMLConstraint(Loads* loadsList, const std::map<unsigned int, unsigned int> &atomIndexToDOFIndex, sofa::core::behavior::MechanicalState<DataTypes> *mm);

    ~LMLConstraint() { /*delete loads;*/}

    /// return the targets list
    std::vector<unsigned int> getTargets() {return targets;}

    ///fix or translate a point
    LMLConstraint<DataTypes>* addConstraint(unsigned int index, Deriv trans);
    LMLConstraint<DataTypes>* removeConstraint(int index);

    /// Constraint inherits
    void projectResponse(VecDeriv& dx); ///< project dx to constrained space
    virtual void projectVelocity(VecDeriv& ) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& x); ///< project x to constrained space (x models a position)

    virtual void getConstraintViolation(const core::ConstraintParams* /*cParams*/, defaulttype::BaseVector* /*resV*/, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
    {
        serr << "LMLConstraint::getConstraintViolation() not implemented" << sendl;
    }


    virtual void buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, DataMatrixDeriv& /*c*/, unsigned& /*cIndex*/, const DataVecCoord& /*x*/)
    {
        serr << "LMLConstraint::buildConstraintMatrix() not implemented" << sendl;
    }

    /// -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

private:

    /// fix a point on the axe specified (0=x, 1=y, 2=z)
    void fixDOF(int index, int axe);

    sofa::core::behavior::MechanicalState<DataTypes> * mmodel;
    /// the set of vertex targets
    std::vector<unsigned int> targets;
    /// list of translations
    VecDeriv translations;
    /// list of fixed directions
    VecDeriv directionsNULLs;
    VecDeriv initPos;

    /// the lml loads
    Loads * loads;
    ///link between PML object indexes and sofa Dofs Indexes
    std::map<unsigned int, unsigned int> atomToDOFIndexes;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_FILEMANAGER_PML)
extern template class SOFA_BUILD_FILEMANAGER_PML_API LMLConstraint<Vec3Types>;
#endif

}
}
}
#endif //LMLCONSTRAINT_H
