%module helper

%include "std_string.i"

%{
#include "sofa/helper/GenerateRigid.h"

%}

namespace sofa
{
namespace helper
{

struct GenerateRigidInfo
{
    sofa::defaulttype::Matrix3 inertia;
    sofa::defaulttype::Quaternion inertia_rotation;
    sofa::defaulttype::Vector3 inertia_diagonal;
    sofa::defaulttype::Vector3 com;
    SReal mass;
};

bool generateRigid( sofa::helper::GenerateRigidInfo& res
                  , const std::string& meshFilename
                  , SReal density
                  , const sofa::defaulttype::Vector3& scale = sofa::defaulttype::Vector3(1,1,1)
                  , const sofa::defaulttype::Vector3& rotation /*Euler angles*/ = sofa::defaulttype::Vector3(0,0,0)
                  );

}
}