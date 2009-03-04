#include <sofa/component/visualmodel/OglAttribute.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS ( OglFloatAttribute );
SOFA_DECL_CLASS ( OglFloat2Attribute );
SOFA_DECL_CLASS ( OglFloat3Attribute );
SOFA_DECL_CLASS ( OglFloat4Attribute );

int OglFloatAttributeClass = core::RegisterObject ( "OglFloatAttribute" ).add< OglFloatAttribute >();
int OglFloat2AttributeClass = core::RegisterObject ( "OglFloat2Attribute" ).add< OglFloat2Attribute >();
int OglFloat3AttributeClass = core::RegisterObject ( "OglFloat3Attribute" ).add< OglFloat3Attribute >();
int OglFloat4AttributeClass = core::RegisterObject ( "OglFloat4Attribute" ).add< OglFloat4Attribute >();

SOFA_DECL_CLASS ( OglIntAttribute );
SOFA_DECL_CLASS ( OglInt2Attribute );
SOFA_DECL_CLASS ( OglInt3Attribute );
SOFA_DECL_CLASS ( OglInt4Attribute );

int OglIntAttributeClass = core::RegisterObject ( "OglIntAttribute" ).add< OglIntAttribute >();
int OglInt2AttributeClass = core::RegisterObject ( "OglInt2Attribute" ).add< OglInt2Attribute >();
int OglInt3AttributeClass = core::RegisterObject ( "OglInt3Attribute" ).add< OglInt3Attribute >();
int OglInt4AttributeClass = core::RegisterObject ( "OglInt4Attribute" ).add< OglInt4Attribute >();

SOFA_DECL_CLASS ( OglUIntAttribute );
SOFA_DECL_CLASS ( OglUInt2Attribute );
SOFA_DECL_CLASS ( OglUInt3Attribute );
SOFA_DECL_CLASS ( OglUInt4Attribute );

int OglUIntAttributeClass = core::RegisterObject ( "OglUIntAttribute" ).add< OglUIntAttribute >();
int OglUInt2AttributeClass = core::RegisterObject ( "OglUInt2Attribute" ).add< OglUInt2Attribute >();
int OglUInt3AttributeClass = core::RegisterObject ( "OglUInt3Attribute" ).add< OglUInt3Attribute >();
int OglUInt4AttributeClass = core::RegisterObject ( "OglUInt4Attribute" ).add< OglUInt4Attribute >();

}

}

}
