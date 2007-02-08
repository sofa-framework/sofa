#ifndef SOFA_CORE_BASEMAPPING_H
#define SOFA_CORE_BASEMAPPING_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/BehaviorModel.h>

namespace sofa
{

namespace core
{

/*! \class BaseMapping
  *  \brief An interface to convert a model to an other model
  *  \author Fonteneau Sylvere
  *  \version 0.1
  *  \date    02/22/2004
  *
  *	 <P> This Interface is used for the Mappings. A Mapping can convert one model to an other. <BR>
  *  For example, we can have a mapping from BehaviorModel to a VisualModel. <BR>
  *
  */

class BaseMapping : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseMapping() { }

    // Mapping Interface
    virtual void init() = 0;

    /*! \fn void updateMapping()
     *  \brief apply the transformation from a model to an other model (like apply displacement from BehaviorModel to VisualModel)
     */
    virtual void updateMapping() = 0;

    virtual objectmodel::BaseObject* getFrom() = 0;
    virtual objectmodel::BaseObject* getTo() = 0;
};

} // namespace core

} // namespace sofa

#endif
