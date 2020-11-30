#pragma once
#include <sofa/core/datatypes/DataMaterial.h>

namespace sofa::core::objectmodel
{

template<>
bool Data<sofa::helper::types::Material>::operator==(const sofa::helper::types::Material& value) const { return false; }

template<>
bool Data<sofa::helper::types::Material>::operator!=(const sofa::helper::types::Material& value) const { return false; }

}
