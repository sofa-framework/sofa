# These are real modules. If you need components from there you
# need to first add find_package(XXX) in your CMakeLists.txt

# Common
sofa_add_collection(SofaSimpleFem SofaSimpleFem ON)
sofa_add_collection(SofaRigid SofaRigid ON)
sofa_add_collection(SofaDeformable SofaDeformable ON)
sofa_add_collection(SofaObjectInteraction SofaObjectInteraction ON)
sofa_add_collection(SofaMeshCollision SofaMeshCollision ON)
sofa_add_collection(SofaEngine SofaEngine ON)
sofa_add_collection(SofaExplicitOdeSolver SofaExplicitOdeSolver ON)
sofa_add_collection(SofaImplicitOdeSolver SofaImplicitOdeSolver ON)
sofa_add_collection(SofaLoader SofaLoader ON)
