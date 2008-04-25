#ifndef SOFA_TYPEDEF
#define SOFA_TYPEDEF


//Compilation WITHOUT double
#ifndef SOFA_DOUBLE
#include <sofa/component/typedef/Sofa_float.h>
#endif

//Compilation WITHOUT float
#ifndef SOFA_FLOAT
#include <sofa/component/typedef/Sofa_double.h>
#endif

#ifndef SOFA_DOUBLE
#ifndef SOFA_FLOAT
#include <sofa/component/typedef/Mapping_combination.h>
#endif
#endif


//---------------------
//Laparoscopic
typedef sofa::component::MechanicalObject<sofa::defaulttype::LaparoscopicRigid3Types> MechanicalObjectLaparoscopicRigid3;

#endif
