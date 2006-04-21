#ifndef SOFA_CORE_BASICMAPPING_H
#define SOFA_CORE_BASICMAPPING_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include "Sofa/Abstract/BaseObject.h"
#include "Sofa/Abstract/BehaviorModel.h"

namespace Sofa
{

namespace Core
{

/*! \class BasicMapping
  *  \brief An interface to convert a model to an other model
  *  \author Fonteneau Sylvere
  *  \version 0.1
  *  \date    02/22/2004
  *
  *	 <P> This Interface is used for the Mappings. A Mapping can convert one model to an other. <BR>
  *  For example, we can have a mapping from BehaviorModel to a VisualModel. <BR>
  *
  */

class BasicMapping : public virtual Abstract::BaseObject
{
public:
    virtual ~BasicMapping() { }

    // Mapping Interface
    virtual void init() = 0;

    /*! \fn void updateMapping()
     *  \brief apply the transformation from a model to an other model (like apply displacement from BehaviorModel to VisualModel)
     */
    virtual void updateMapping() = 0;

    virtual Abstract::Base* getFrom() = 0;
    virtual Abstract::Base* getTo() = 0;
};

} // namespace Core

} // namespace Sofa

#endif
