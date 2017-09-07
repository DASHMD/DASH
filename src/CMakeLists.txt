file (GLOB_RECURSE SOURCES "*.cpp" "*.cu")
list (REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

set(INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}) 
set(INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/GPUArrays) 
set(INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/Integrators)
set(INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/Fixes) 
set(INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/DataStorageUser) 
set(INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/Evaluators) 
set(INC_DIRS ${INC_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/BondedForcers) 

SET(PY_INC_DIRS "${INC_DIRS}" PARENT_SCOPE)
include_directories(${INC_DIRS})

configure_file (
    "${PROJECT_SOURCE_DIR}/Config.h.in"
    "${PROJECT_BINARY_DIR}/Config.h"
)
MESSAGE(STATUS ${PYTHON_LIBRARIES})
MESSAGE(STATUS ${CUDA_LIBRARIES})
cuda_add_library (${MD_ENGINE_LIB_NAME} ${SOURCES} SHARED)
#PYTHON_LIBRARIES IS NO VALUE
target_link_libraries (${MD_ENGINE_LIB_NAME} ${PYTHON_LIBRARIES}
                                            ${Boost_LIBRARIES}
											 #${PugiXML_LIBRARIES}
                                             ${CUDA_LIBRARIES}
                                             ${CUDA_CUFFT_LIBRARIES})

# TODO: Why does install(TARGETS ...) not work?
#install(TARGETS ${MD_ENGINE_LIB_NAME} LIBRARY DESTINATION lib)
install(FILES ${CMAKE_BINARY_DIR}/lib${MD_ENGINE_LIB_NAME}.so DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

cuda_add_executable (testing main.cpp)
target_link_libraries(testing ${MD_ENGINE_LIB_NAME})

#cuda_add_executable (testing main.cpp)
#target_link_libraries (testing ${MD_ENGINE_LIB_NAME})

