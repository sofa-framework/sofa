#pragma once
#include <sofa/helper/SelectableItem.h>

namespace sofa::component::solidmechanics::fem::elastic
{

constexpr std::string_view parallelComputeStrategy = "parallel";
constexpr std::string_view sequencedComputeStrategy = "sequenced";

MAKE_SELECTABLE_ITEMS(ComputeStrategy,
    sofa::helper::Item{parallelComputeStrategy, "The algorithm is executed in parallel"},
    sofa::helper::Item{sequencedComputeStrategy, "The algorithm is executed sequentially"},
);

}
