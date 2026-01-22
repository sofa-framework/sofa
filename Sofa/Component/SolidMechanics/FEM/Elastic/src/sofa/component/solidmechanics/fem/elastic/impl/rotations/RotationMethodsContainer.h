#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <variant>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class RotationMethodsContainer
 * @brief A container for rotation computation methods in corotational formulations
 *
 * This class provides a flexible mechanism to select and execute different rotation computation
 * strategies for elements in corotational finite element simulations. The rotation methods
 * calculate the element's rotation relative to its initial configuration, which is essential
 * for corotational force fields.
 *
 * @tparam DataTypes The data type used throughout the simulation (e.g., sofa::defaulttype::Vec3Types for 3D)
 * @tparam ElementType The element type (e.g., sofa::geometry::Triangle, sofa::geometry::Tetrahedron)
 * @tparam Methods... A pack of rotation method classes implementing the `computeRotation` interface
 *
 * Key Features:
 * - Supports multiple rotation computation strategies via std::variant
 * - Provides GUI-driven method selection through a Data-driven mechanism
 *
 * Technical Details:
 * - The active method is stored in `m_rotationComputer` (std::variant)
 * - Method selection is driven by the `d_rotationMethod` data object (index-based)
 * - The `selectRotationMethod` function updates the active method based on the selected index
 * - The `MAKE_SELECTABLE_ITEMS` macro generates GUI items for method selection
 *
 * Why Multiple Methods?
 * Corotational formulations often require different rotation strategies depending on computational efficiency requirements.
 *
 * This container enables seamless switching between methods without reconfiguring the entire simulation.
 */
template <class DataTypes, class ElementType, class... Methods>
struct RotationMethodsContainer
{
private:
    std::variant<Methods...> m_rotationComputer;

public:

    static constexpr auto NumberOfMethods = std::variant_size_v<decltype(m_rotationComputer)>;

    explicit RotationMethodsContainer(sofa::core::objectmodel::BaseObject* parent)
        : d_rotationMethod(parent->initData(&d_rotationMethod, "rotationMethod", ("The method used to compute the element rotations.\n" + RotationMethodsItems::dataDescription()).c_str()))
    {}

    using RotationMatrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;
    static constexpr std::size_t NumberOfNodesInElement = ElementType::NumberOfNodes;

    /**
     * Computes the current rotation state relative to the initial configuration.
     *
     * @param rotationMatrix Output: Current rotation matrix (relative to initial config)
     * @param initialRotationMatrix Initial rotation matrix (for reference)
     * @param nodesPosition Current positions of element nodes
     * @param nodesRestPosition Rest positions of element nodes (initial configuration)
     */
    void computeRotation(RotationMatrix& rotationMatrix, const RotationMatrix& initialRotationMatrix,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesPosition,
        const std::array<sofa::Coord_t<DataTypes>, NumberOfNodesInElement>& nodesRestPosition)
    {
        std::visit(
            [&](auto& rotationComputer)
            {
                rotationComputer.computeRotation(rotationMatrix, initialRotationMatrix, nodesPosition, nodesRestPosition);
            },
            m_rotationComputer);
    }

    struct RotationMethodsItems final : sofa::helper::SelectableItem<RotationMethodsItems>
    {
        using sofa::helper::SelectableItem<RotationMethodsItems>::SelectableItem;
        using sofa::helper::SelectableItem<RotationMethodsItems>::operator=;
        using sofa::helper::SelectableItem<RotationMethodsItems>::operator==;
        static constexpr std::array s_items{Methods::getItem()...};
        static_assert(std::is_same_v<typename decltype(s_items)::value_type, sofa::helper::Item>);
    };
    sofa::Data< RotationMethodsItems > d_rotationMethod;

    /**
     * Selects the rotation method based on the current index
     *
     * @note This function is typically called by the GUI to update the active method.
     */
    void selectRotationMethod()
    {
        const std::size_t selectedId = d_rotationMethod.getValue();

        if (selectedId < NumberOfMethods)
        {
            doSelectRotationMethod<NumberOfMethods - 1>(selectedId);
        }
    }

private:

    /**
     * Recursively selects the method at the given index
     * @tparam Id The index of the method among the available methods
     * @param selectedId The index of the method to select
     */
    template<std::size_t Id>
    void doSelectRotationMethod(const std::size_t selectedId)
    {
        if (selectedId == Id)
        {
            using SelectedMethod = std::variant_alternative_t<Id, decltype(m_rotationComputer)>;
            m_rotationComputer.template emplace<SelectedMethod>();
        }
        else
        {
            if constexpr (Id > 0)
            {
                doSelectRotationMethod<Id - 1>(selectedId);
            }
        }
    }
};

}
