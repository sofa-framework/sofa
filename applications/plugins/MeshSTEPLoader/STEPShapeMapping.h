#ifndef SOFA_COMPONENT_MAPPING_STEPSHAPEMAPPING_H
#define SOFA_COMPONENT_MAPPING_STEPSHAPEMAPPING_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/State.h>
#include <SofaBaseTopology/MeshTopology.h>
#include "SingleComponent.h"
#include "MeshSTEPLoader.h"
#include <sofa/core/objectmodel/BaseObjectDescription.h>
namespace sofa
{
namespace component
{
namespace engine
{

class STEPShapeExtractor : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(STEPShapeExtractor,sofa::core::DataEngine);

public:

    typedef core::topology::Topology::Triangle Triangle;
    STEPShapeExtractor(loader::MeshSTEPLoader* loader=NULL,topology::MeshTopology* topology=NULL);

    void init();
    void update();

    Data<unsigned int> shapeNumber; ///< Shape number to be loaded
    Data<unsigned int > indexBegin; ///< The begin index for this shape with respect to the global mesh
    Data<unsigned int > indexEnd; ///< The end index for this shape with respect to the global mesh
public:

    template< class T>
    static std::string shortName( const T* /*ptr*/ = NULL, core::objectmodel::BaseObjectDescription* = NULL )
    {
        return std::string("stepShapeExtractor");
    }

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the input and output attributes and check
    /// if they are compatible with the input and output model types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        loader::MeshSTEPLoader* loader = NULL;
        topology::MeshTopology* topology = NULL;

        std::string inPath, outPath;

        if (arg->getAttribute("input"))
            inPath = arg->getAttribute("input");

        context->findLinkDest(loader, inPath, NULL);

        if (arg->getAttribute("output"))
            outPath = arg->getAttribute("output");

        context->findLinkDest(topology, outPath, NULL);

        if (loader == NULL)
        {
            context->serr << "Cannot create "<<className(obj)<<" as input model is missing or invalid." << context->sendl;
            return false;
        }

        if (topology == NULL)
        {
            context->serr << "Cannot create "<<className(obj)<<" as output model is missing or invalid." << context->sendl;
            return false;
        }

        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the input and output attributes to
    /// find the input and output models of this mapping.
    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();

        if (context)
            context->addObject(obj);

        if (arg)
        {
            std::string inPath, outPath;
            if (arg->getAttribute("input"))
                inPath = arg->getAttribute("input");

            if (arg->getAttribute("output"))
                outPath = arg->getAttribute("output");

            obj->loader.setPath( inPath );
            obj->topology.setPath( outPath );

            obj->parse(arg);
        }

        return obj;
    }

protected:
    SingleLink<STEPShapeExtractor, loader::MeshSTEPLoader, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> loader;
    SingleLink<STEPShapeExtractor, topology::MeshTopology, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> topology;

};

}

}

}

#endif
