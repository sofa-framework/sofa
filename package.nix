{
  boost,
  cmake,
  cxxopts,
  eigen,
  #fetchFromGitHub,
  glew,
  gtest,
  lib,
  qt6Packages,
  libGL,
  metis,
  stdenv,
  tinyxml-2,
  zlib,
}:

stdenv.mkDerivation {
  pname = "sofa";
  version = "24.06.00";

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./applications
      ./Authors.txt
      ./cmake
      ./CHANGELOG.md
      ./CMakeLists.txt
      ./CMakePresets.json
      ./examples
      ./extlibs
      ./LICENSE-LGPL.md
      ./package.cmake
      ./README.md
      ./scripts
      ./share
      ./Sofa
      ./tools
    ];
  };

  propagatedNativeBuildInputs = [
    cmake
    qt6Packages.wrapQtAppsHook
  ];
  propagatedBuildInputs = [
    boost
    cxxopts
    eigen
    glew
    gtest
    qt6Packages.libqglviewer
    qt6Packages.qtbase
    libGL
    metis
    tinyxml-2
    zlib
  ];

  cmakeFlags = [
    (lib.cmakeBool "SOFA_ALLOW_FETCH_DEPENDENCIES" false)
  ];

  doCheck = true;

  postFixup = lib.optionalString stdenv.hostPlatform.isDarwin ''
    install_name_tool -change \
      $out/lib/libSceneChecking.24.12.99.dylib \
      $out/plugins/SceneChecking/lib/libSceneChecking.24.12.99.dylib \
      $out/bin/.runSofa-24.12.99-wrapped
  '';

  meta = {
    description = "SOFA is an open-source framework for interactive physics simulation, with emphasis on biomechanical and robotic simulations";
    homepage = "https://github.com/sofa-framework/sofa";
    license = lib.licenses.lgpl21Only;
    maintainers = with lib.maintainers; [ nim65s ];
    mainProgram = "runSofa";
    platforms = lib.platforms.unix ++ lib.platforms.windows;
  };
}
