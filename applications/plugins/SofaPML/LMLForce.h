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
//	LMLForce imports from a LML file (see LMLReader class) the forces
//  applied on DOFs, and traduces it to sofa Forcefield.
//  It inherits from ForceField sofa core class.
//-------------------------------------------------------------------------

#ifndef SOFAPML_LMLFORCE_H
#define SOFAPML_LMLFORCE_H

#include <Loads.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/visual/VisualModel.h>
#include <SofaPML/config.h>

#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace std;

template<class DataTypes>
class LMLForce : public ForceField<DataTypes> //, public VisualModel
{
    SOFA_CLASS(SOFA_TEMPLATE(LMLForce, DataTypes), SOFA_TEMPLATE(ForceField, DataTypes));
public :
    ///template types
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord::iterator VecCoordIterator;
    typedef typename DataTypes::VecDeriv::iterator VecDerivIterator;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    ///constructors
    LMLForce(MechanicalState<DataTypes> *mm = NULL)
        : ForceField<DataTypes>(mm)
    {}
    LMLForce(Loads* loadsList, const map<unsigned int, unsigned int> &atomIndexToDOFIndex, MechanicalState<DataTypes> *mm);

    ~LMLForce() { /*delete loads;*/}

    /// return targets list
    std::vector<unsigned int> getTargets() {return targets;}

    ///add a new target (dof index)
    void addTarget(unsigned int i) { targets.push_back(i); forces.push_back(Deriv() ); }

    /// -- ForceField Inherits
    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);
    virtual void addDForce (VecDeriv& , const VecDeriv& ) {}
    virtual double getPotentialEnergy(const VecCoord& ) const {return 0;}

    virtual void addForce(const MechanicalParams* /*mparams*/, DataVecDeriv& /*f*/ , const DataVecCoord& /*x*/ , const DataVecDeriv& /*v*/)
    {
        serr << "LMLForce::addForce() not implemented" << sendl;
    }
    virtual void addDForce(const MechanicalParams* /*mparams*/, DataVecDeriv& /*df*/, const DataVecDeriv& /*dx*/)
    {
        serr << "LMLForce::addDForce() not implemented" << sendl;
    }
    virtual double getPotentialEnergy(const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/) const
    {
        serr << "LMLForce::getPotentialEnergy() not implemented" << sendl;
        return 0.;
    }
    /// -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

protected:

    MechanicalState<DataTypes> * mmodel;
    /// list of forces targets
    std::vector<unsigned int> targets;
    /// list of force directions
    VecDeriv forces;
    /// LML loads
    Loads* loads;
    /// link between PML object indexes and sofa dofs indexs
    map<unsigned int, unsigned int> atomToDOFIndexes;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_FILEMANAGER_PML)
extern template class SOFA_BUILD_FILEMANAGER_PML_API LMLForce<Vec3Types>;
#endif

}
}
}
#endif

