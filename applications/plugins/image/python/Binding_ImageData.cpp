
#include <SofaPython/PythonMacros.h>
#include <SofaPython/Binding_SofaModule.h>
#include <SofaPython/PythonFactory.h>

#include "Binding_ImageData.h"
#include <SofaPython/Binding_Data.h>
#include <SofaPython/PythonToSofa.inl>


// kind of old-style templates
#define IMAGETYPE ImageC
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageUC
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageI
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageUI
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageS
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageUS
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageL
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageUL
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageF
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageD
#include "Binding_ImageData.inl"
#undef IMAGETYPE
#define IMAGETYPE ImageB
#include "Binding_ImageData.inl"
#undef IMAGETYPE

