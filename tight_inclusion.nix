{
  lib,

  stdenv,
  fetchFromGitHub,

  cmake,

  eigen,
  spdlog,
}:

let
  cf = fetchFromGitHub {
    owner = "conda-forge";
    repo = "tight-inclusion-feedstock";
    rev = "ad0bb48ec12aa991ae208252cdd7e61fa37bdc10";
    hash = "sha256-4v6h1c2fe0DK5z/s86B7oUyUBAYXTnMHzOpBM91NEpY=";
  };

in

stdenv.mkDerivation (finalAttrs: {
  pname = "tight-inclusion";
  version = "1.0.6";

  src = fetchFromGitHub {
    owner = "Continuous-Collision-Detection";
    repo = "Tight-Inclusion";
    tag = "v${finalAttrs.version}";
    hash = "sha256-Yq79qShpA5VmPan+QV05GrGrmsoUuVFCrJDK7F207Qs=";
  };

  patches = [
    "${cf}/recipe/patches/0001-Export-target-install-header-in-right-dir-still-need.patch"
    "${cf}/recipe/patches/0002-Export-symbols-on-windows.patch"
    "${cf}/recipe/patches/0003-patch-for-conda-forge-package.patch"
    "${cf}/recipe/patches/0004-add-cmake-config-file-homogenize-project-name.patch"
    "${cf}/recipe/patches/0005-fix-installation-of-configured-config.hpp-file.patch"
  ];

  postPatch = ''
    # we don't need the bin
    substituteInPlace CMakeLists.txt \
      --replace-fail \
        "add_subdirectory(app)" \
        ""
  '';

  nativeBuildInputs = [
    cmake
  ];

  propagatedBuildInputs = [
    eigen
    spdlog
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DTIGHT_INCLUSION_TOPLEVEL_PROJECT=ON"
  ];

  meta = {
    description = "Conservative continuous collision detection (CCD) method with support for minimum separation";
    homepage = "https://github.com/Continuous-Collision-Detection/Tight-Inclusion";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ nim65s ];
    platforms = lib.platforms.unix;
  };
})
