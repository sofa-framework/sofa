#include <iostream>
#include "gtest/gtest.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    // Set LC_CTYPE according to the environnement variable.
    setlocale(LC_CTYPE, "");

    return RUN_ALL_TESTS();
}
