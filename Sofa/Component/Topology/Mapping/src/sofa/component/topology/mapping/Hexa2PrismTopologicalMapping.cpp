#include <sofa/component/topology/mapping/Hexa2PrismTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>

#include <sofa/component/topology/container/grid/GridTopology.h>
#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::mapping
{

void registerHexa2PrismTopologicalMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Topological mapping where HexahedronSetTopology is converted to PrismSetTopology")
        .add< Hexa2PrismTopologicalMapping >());
}

Hexa2PrismTopologicalMapping::Hexa2PrismTopologicalMapping()
    : sofa::core::topology::TopologicalMapping()
    , d_swapping(initData(&d_swapping, false, "swapping", "Boolean enabling to swap hexa-edges\n in order to avoid bias effect"))
{
    m_inputType = geometry::ElementType::HEXAHEDRON;
    m_outputType = geometry::ElementType::PRISM;
}

Hexa2PrismTopologicalMapping::~Hexa2PrismTopologicalMapping()
{
}

void Hexa2PrismTopologicalMapping::init()
{
    using namespace container::dynamic;

    Inherit1::init();

    if (toModel == nullptr)
    {
        msg_error() << "No target topology container found.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // INITIALISATION of PRISM mesh from HEXAHEDRAL mesh:

    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    auto Loc2GlobVec = sofa::helper::getWriteOnlyAccessor(Loc2GlobDataVec);
    Loc2GlobVec.clear();
    Glob2LocMap.clear();

    const size_t nbcubes = fromModel->getNbHexahedra();

    // These values are only correct if the mesh is a grid topology
    int nx = 2;
    int ny = 1;
    {
        const auto* grid = dynamic_cast<container::grid::GridTopology*>(fromModel.get());
        if (grid != nullptr)
        {
            nx = grid->getNx() - 1;
            ny = grid->getNy() - 1;
        }
    }

    static constexpr int numberPrismsInHexa = 2;
    Loc2GlobVec.reserve(nbcubes * numberPrismsInHexa);

    const bool swapping = d_swapping.getValue();

    // Tessellation of each cube into 2 triangular prisms
    // Hexahedron vertices:
    //     4-------7
    //    /|      /|
    //   0-------3 |
    //   | 5----|--6
    //   |/     | /
    //   1-------2
    //
    // Decomposition into 2 prisms:
    // - Prism 1: vertices [0, 1, 5] (bottom triangle) and [3, 2, 6] (top triangle)
    // - Prism 2: vertices [0, 5, 4] (bottom triangle) and [3, 6, 7] (top triangle)
    // This ensures face consistency between neighboring hexahedra

    for (size_t i = 0; i < nbcubes; ++i)
    {
        core::topology::BaseMeshTopology::Hexa c = fromModel->getHexahedron(i);

        bool swapped = false;

        if (swapping)
        {
            if (!((i % nx) & 1))
            {
                // swap all points on the X edges
                std::swap(c[0], c[1]);
                std::swap(c[3], c[2]);
                std::swap(c[4], c[5]);
                std::swap(c[7], c[6]);
                swapped = !swapped;
            }
            if (((i / nx) % ny) & 1)
            {
                // swap all points on the Y edges
                std::swap(c[0], c[3]);
                std::swap(c[1], c[2]);
                std::swap(c[4], c[7]);
                std::swap(c[5], c[6]);
                swapped = !swapped;
            }
            if ((i / (nx * ny)) & 1)
            {
                // swap all points on the Z edges
                std::swap(c[0], c[4]);
                std::swap(c[1], c[5]);
                std::swap(c[2], c[6]);
                std::swap(c[3], c[7]);
                swapped = !swapped;
            }
        }

        if (!swapped)
        {
            // Standard decomposition ensuring face consistency between neighbors
            toModel->addPrism(c[0], c[1], c[5], c[3], c[2], c[6]);  // Prism 1
            toModel->addPrism(c[0], c[5], c[4], c[3], c[6], c[7]);  // Prism 2
        }
        else
        {
            // Swapped decomposition
            toModel->addPrism(c[0], c[1], c[5], c[3], c[2], c[6]);  // Prism 1
            toModel->addPrism(c[0], c[5], c[4], c[3], c[6], c[7]);  // Prism 2
        }

        for (int j = 0; j < numberPrismsInHexa; ++j)
        {
            Loc2GlobVec.push_back(i);
        }
        Glob2LocMap[i] = static_cast<unsigned int>(Loc2GlobVec.size()) - 1;
    }

    // Need to fully init the target topology
    toModel->init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

Index Hexa2PrismTopologicalMapping::getFromIndex(Index /*ind*/)
{
    return sofa::InvalidID;
}

void Hexa2PrismTopologicalMapping::updateTopologicalMappingTopDown()
{
    msg_warning() << "Method Hexa2PrismTopologicalMapping::updateTopologicalMappingTopDown() not yet implemented!";
    // TODO...
}

} // namespace sofa::component::topology::mapping
