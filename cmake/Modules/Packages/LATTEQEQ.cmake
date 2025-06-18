enable_language(Fortran)

# using lammps in a super-build setting
if(TARGET LATTEQEQ::latteqeq)
  target_link_libraries(lammps PRIVATE LATTEQEQ::latteqeq)
  return()
endif()

find_package(LATTEQEQ 1.2.2 CONFIG)
if(LATTEQEQ_FOUND)
  set(DOWNLOAD_LATTEQEQ_DEFAULT OFF)
else()
  set(DOWNLOAD_LATTEQEQ_DEFAULT ON)
endif()
### (TODO) LATTEQEQ DOWNLOAD

#option(DOWNLOAD_LATTEQEQ "Download the LATTEQEQ library instead of using an already installed one" ${DOWNLOAD_LATTEQEQ_DEFAULT})
#if(DOWNLOAD_LATTEQEQ)
#  message(STATUS "LATTEQEQ download requested - we will build our own")
#  set(LATTEQEQ_URL "https://github.com/lanl/LATTEQEQ/archive/v1.2.2.tar.gz" CACHE STRING "URL for LATTEQEQ tarball")
#  set(LATTEQEQ_MD5 "820e73a457ced178c08c71389a385de7" CACHE STRING "MD5 checksum of LATTEQEQ tarball")
#  mark_as_advanced(LATTEQEQ_URL)
#  mark_as_advanced(LATTEQEQ_MD5)
#
#  # CMake cannot pass BLAS or LAPACK library variable to external project if they are a list
#  list(LENGTH BLAS_LIBRARIES} NUM_BLAS)
#  list(LENGTH LAPACK_LIBRARIES NUM_LAPACK)
#  if((NUM_BLAS GREATER 1) OR (NUM_LAPACK GREATER 1))
#    message(FATAL_ERROR "Cannot compile downloaded LATTEQEQ library due to a technical limitation")
#  endif()
#
#  include(ExternalProject)
#  ExternalProject_Add(latteqeq_build
#    URL     ${LATTEQEQ_URL}
#    URL_MD5 ${LATTEQEQ_MD5}
#    SOURCE_SUBDIR cmake
#    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${CMAKE_REQUEST_PIC} -DCMAKE_INSTALL_LIBDIR=lib
#    -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}
#    -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} -DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}
#    -DCMAKE_Fortran_FLAGS_${BTYPE}=${CMAKE_Fortran_FLAGS_${BTYPE}} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
#    BUILD_BYPRODUCTS <INSTALL_DIR>/lib/liblatteqeq.a
#  )
#  ExternalProject_get_property(latteqeq_build INSTALL_DIR)
#  add_library(LAMMPS::LATTEQEQ UNKNOWN IMPORTED)
#  set_target_properties(LAMMPS::LATTEQEQ PROPERTIES
#    IMPORTED_LOCATION "${INSTALL_DIR}/lib/liblatteqeq.a"
#    INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}")
#  target_link_libraries(lammps PRIVATE LAMMPS::LATTEQEQ)
#  add_dependencies(LAMMPS::LATTEQEQ latteqeq_build)
#else()
#  find_package(LATTEQEQ 1.2.2 REQUIRED CONFIG)
#  target_link_libraries(lammps PRIVATE LATTEQEQ::latteqeq)
#endif()
find_package(LATTEQEQ 1.2.2 REQUIRED CONFIG)
target_link_libraries(lammps PRIVATE LATTEQEQ::latteqeq)
