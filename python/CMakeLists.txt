find_program (PYTHON_BIN "python")
if (PYTHON_BIN)
    set (SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set (SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set (PYTHON_LIB_NAME "DASH.so")

    get_filename_component (SIM_SOURCES DASH.cpp ABSOLUTE)

    configure_file (${SETUP_PY_IN} ${SETUP_PY})

    add_custom_target (PythonLib ALL
                       ${PYTHON_BIN} ${SETUP_PY} build # Build command
                       SOURCES ${SIM_SOURCES})

    add_dependencies (PythonLib ${MD_ENGINE_LIB_NAME})

    install (CODE "execute_process(COMMAND ${PYTHON_BIN} ${SETUP_PY} install)")
else (PYTHON_BIN)
    message (WARNING "Python binary not found.")
endif (PYTHON_BIN)
