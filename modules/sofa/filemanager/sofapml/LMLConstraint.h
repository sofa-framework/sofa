/***************************************************************************
								LMLConstraint
                             -------------------
    begin             : August 9th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2006/08/09 9:58:16 $
    Version           : $Revision: 0.1 $
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
//	LMLConstraint imports from a LML file (see LMLReader class) imposed displacements,
//  including translations and fixed points, and traduces it to sofa constraints.
//  It inherits from Constraint sofa core class.
//-------------------------------------------------------------------------

#ifndef LMLCONSTRAINT_H
#define LMLCONSTRAINT_H


#include "sofa/core/componentmodel/behavior/Constraint.h"
#include "sofa/core/componentmodel/behavior/MechanicalState.h"
#include "sofa/core/VisualModel.h"

#include <vector>
#include <map>

#include <Loads.h>


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
class LMLConstraint : public Constraint<DataTypes>, public VisualModel
{
public :
    ///template types
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord::iterator VecCoordIterator;
    typedef typename DataTypes::VecDeriv::iterator VecDerivIterator;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    ///constructor
    LMLConstraint(Loads* loadsList, const map<unsigned int, unsigned int> &atomIndexToDOFIndex, MechanicalState<DataTypes> *mm);

    ~LMLConstraint() { delete loads;}

    /// return the targets list
    std::vector<unsigned int> getTargets() {return targets;}

    ///fix or translate a point
    LMLConstraint<DataTypes>* addConstraint(unsigned int index, Deriv trans);
    LMLConstraint<DataTypes>* removeConstraint(int index);

    /// Constraint inherits
    void projectResponse(VecDeriv& dx); ///< project dx to constrained space
    virtual void projectVelocity(VecDeriv& ) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& x); ///< project x to constrained space (x models a position)

    /// -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

private:

    /// fix a point on the axe specified (0=x, 1=y, 2=z)
    void fixDOF(int index, int axe);

    MechanicalState<DataTypes> * mmodel;
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
    map<unsigned int, unsigned int> atomToDOFIndexes;
};

}
}
}
#endif //LMLCONSTRAINT_H
