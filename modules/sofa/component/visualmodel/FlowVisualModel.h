/*
 * FlowVisualModel.h
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#ifndef FLOWVISUALMODEL_H_
#define FLOWVISUALMODEL_H_

#include <sofa/component/component.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

template <class DataTypes>
class SOFA_COMPONENT_VISUALMODEL_API FlowVisualModel : public core::VisualModel
{
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> FluidState;
protected:
    FluidState* fstate;

public:
    FlowVisualModel();
    virtual ~FlowVisualModel();

    void init();
    void initVisual();
    void draw();
};
/*
#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_FLUIDSOLIDINTERACTIONFORCEFIELD_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class FlowVisualModel<defaulttype::Vec3dTypes>;
extern template class FlowVisualModel<defaulttype::Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class FlowVisualModel<defaulttype::Vec3fTypes>;
extern template class FlowVisualModel<defaulttype::Vec2fTypes>;
#endif
#endif
*/
}

}

}

#endif /* FLOWVISUALMODEL_H_ */
