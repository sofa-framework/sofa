#include <sofa/helper/ComponentChange.h>
#include <iostream>

using sofa::helper::lifecycle::deprecatedComponents;

int main(int argc, char **argv)
{
    for (const auto& component : deprecatedComponents)
    {
        std::cout << component.first << std::endl;
    }

    return 0;
}
