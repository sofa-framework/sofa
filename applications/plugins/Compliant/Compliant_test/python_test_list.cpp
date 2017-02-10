#include <SofaTest/Python_test.h>
#include <sofa/helper/system/FileSystem.h>

namespace sofa {


static bool ends_with(const std::string& suffix, const std::string& full){
    const std::size_t lf = full.length();
    const std::size_t ls = suffix.length();
    
    if(lf < ls) return false;
    
    return (0 == full.compare(lf - ls, ls, suffix));
}

static bool starts_with(const std::string& prefix, const std::string& full){
    const std::size_t lf = full.length();
    const std::size_t lp = prefix.length();
    
    if(lf < lp) return false;
    
    return (0 == full.compare(0, lp, prefix));
}


static bool is_scene_file(const std::string& filename) {
    return starts_with("scene_", filename) && ends_with(".py", filename);
}

static bool is_test_file(const std::string& filename) {
    return starts_with("test_", filename) && ends_with(".py", filename);
}


// these are sofa scenes
static struct Tests : public Python_test_list
{
    Tests()
    {
        static const std::string scenePath = std::string(COMPLIANT_TEST_PYTHON_DIR);

        std::vector<std::string> files;
        helper::system::FileSystem::listDirectory(scenePath, files);

        for(const std::string& file : files) {
            if(is_scene_file(file) ) {
                addTest(file, scenePath);
            }
        }
        
    }
} tests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_scene_test,
                        ::testing::ValuesIn(tests.list));

TEST_P(Python_scene_test, sofa_python_scene_tests)
{
    run(GetParam());
}




////////////////////////


// these are just python files loaded in the sofa python environment (paths...)
static struct Tests2 : public Python_test_list
{
    Tests2()
    {
        static const std::string testPath = std::string(COMPLIANT_TEST_PYTHON_DIR);

        std::vector<std::string> files;
        helper::system::FileSystem::listDirectory(testPath, files);

        for(const std::string& file : files) {
            if(is_test_file(file) ) {
                addTest(file, testPath);
            }
        }
        
    }
} tests2;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_test,
                        ::testing::ValuesIn(tests2.list));

TEST_P(Python_test, sofa_python_tests)
{
    run(GetParam());
}




} // namespace sofa
