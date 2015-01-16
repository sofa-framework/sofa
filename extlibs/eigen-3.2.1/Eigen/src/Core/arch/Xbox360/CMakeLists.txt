FILE(GLOB Eigen_Core_arch_Xbox360_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_arch_Xbox360_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/arch/Xbox360 COMPONENT Devel
)
