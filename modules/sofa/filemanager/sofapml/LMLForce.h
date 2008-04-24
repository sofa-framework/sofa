/***************************************************************************
								   LMLForce
                             -------------------
    begin             : August 12th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2007/02/25 13:51:44 $
    Version           : $Revision: 0.2 $
 ***************************************************************************/

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

#ifndef LMLFORCE_H
#define LMLFORCE_H

#include <Loads.h>

#include "sofa/core/componentmodel/behavior/ForceField.h"
#include "sofa/core/componentmodel/behavior/MechanicalState.h"
#include "sofa/core/VisualModel.h"

#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::core;
using namespace sofa::core::componentmodel::behavior;
using namespace std;

template<class DataTypes>
class LMLForce : public ForceField<DataTypes>, public VisualModel
{
public :
    ///template types
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord::iterator VecCoordIterator;
    typedef typename DataTypes::VecDeriv::iterator VecDerivIterator;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    ///constructors
    LMLForce(MechanicalState<DataTypes> *mm = NULL)
        : ForceField<DataTypes>(mm)
    {}
    LMLForce(Loads* loadsList, const map<unsigned int, unsigned int> &atomIndexToDOFIndex, MechanicalState<DataTypes> *mm);

    ~LMLForce() { delete loads;}

    /// return targets list
    std::vector<unsigned int> getTargets() {return targets;}

    ///add a new target (dof index)
    void addTarget(unsigned int i) { targets.push_back(i); forces.push_back(Deriv() ); }

    /// -- ForceField Inherits
    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);
    virtual void addDForce (VecDeriv& , const VecDeriv& ) {}
    virtual sofa::defaulttype::Vector3::value_type getPotentialEnergy(const VecCoord& ) {return 0;}

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

}
}
}
#endif

