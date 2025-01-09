set(PYTHON_ENABLE_GRIDFORCE_DEFAULT OFF)
if(PKG_ML-PACE)
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.14)
    find_package(Python COMPONENTS NumPy QUIET)
  else()
    # assume we have NumPy
    set(Python_NumPy_FOUND ON)
  endif()
  if(Python_NumPy_FOUND)
    set(PYTHON_ENABLE_GRIDFORCE_DEFAULT ON)
  endif()
endif()

option(PYTHON_ENABLE_GRIDFORCE "Build PYTHON package with grid ace support" ${PYTHON_ENABLE_GRIDFORCE_DEFAULT})

if(CMAKE_VERSION VERSION_LESS 3.12)
  if(NOT PYTHON_VERSION_STRING)
    set(Python_ADDITIONAL_VERSIONS 3.12 3.11 3.10 3.9 3.8 3.7 3.6)
    # search for interpreter first, so we have a consistent library
    find_package(PythonInterp) # Deprecated since version 3.12
    if(PYTHONINTERP_FOUND)
      set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
    endif()
  endif()
  # search for the library matching the selected interpreter
  set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
  find_package(PythonLibs REQUIRED) # Deprecated since version 3.12
  if(NOT (PYTHON_VERSION_STRING STREQUAL PYTHONLIBS_VERSION_STRING))
    message(FATAL_ERROR "Python Library version ${PYTHONLIBS_VERSION_STRING} does not match Interpreter version ${PYTHON_VERSION_STRING}")
  endif()
  target_include_directories(lammps PRIVATE ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(lammps PRIVATE ${PYTHON_LIBRARIES})
else()
  if(NOT Python_INTERPRETER)
    # backward compatibility
    if(PYTHON_EXECUTABLE)
      set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
    endif()
    find_package(Python COMPONENTS Interpreter)
  endif()
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  target_link_libraries(lammps PRIVATE Python::Python)
endif()
target_compile_definitions(lammps PRIVATE -DLMP_PYTHON)

if(PYTHON_ENABLE_GRIDFORCE)
  if(NOT PKG_ML-PACE)
    message(FATAL_ERROR "Must enable ML-PACE package for including grid-based descriptor support in PYTHON")
  endif()
  execute_process(
    COMMAND python -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE npOUTPUT
  )
  string(STRIP ${npOUTPUT} npOUTPUT)
  target_include_directories(lammps
    PRIVATE
    ${npOUTPUT}
  )
  target_compile_definitions(lammps PRIVATE -DPYTHON_GRIDFORCE)
endif()
