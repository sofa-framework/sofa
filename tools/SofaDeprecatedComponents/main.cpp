#include <sofa/helper/deprecated.h>
#include <SofaUserInteraction/DisabledContact.h>
#include <iostream>

using sofa::helper::deprecated::components;

int main(int argc, char **argv)
{
//    std::cout << sofa::component::collision::DisabledContact<void,void>::HeaderFileLocation << std::endl;

//    const std::type_info& r1 = typeid(sofa::component::collision::DisabledContact<int, int>);
//    std::cout << "type : " << r1 << std::endl;

    for (const auto& component : components)
    {
        std::cout << component.first << std::endl;
    }

    return 0;
}
