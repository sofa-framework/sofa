/***************************************************************************
								   LMLForce
                             -------------------
    begin             : August 12th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2006/02/25 13:51:44 $
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

#include "LMLForce.inl"
#include "sofa/defaulttype/Vec3Types.h"
#include "sofa/component/MechanicalObject.h"
#include <sofa/core/ObjectFactory.h>
using namespace sofa::defaulttype;
using namespace sofa::component;

namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(LMLForce)


template class LMLForce<Vec3Types>;
}
}
}

