#project(cuvistaGui)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

#qt_standard_project_setup()
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

qt_add_executable(cuvistaGui
    cuvistaGui.ui
    eula.ui
    player.ui
    progress.ui
    cuvistaGui.qrc
    cuvistaGui.cpp
    Eula.cpp
    main.cpp
    player.cpp
    PlayerWidget.cpp
    progress.cpp
    UserInputGui.cpp
    ImageLabel.cpp
    ClickLabel.cpp
    MessagePrinterGui.cpp
)

target_include_directories(cuvistaGui PRIVATE "${PROJECT_SOURCE_DIR}/Deshaker")
target_include_directories(cuvistaGui PRIVATE "${PROJECT_SOURCE_DIR}/cuvistaGui")
set_target_properties(cuvistaGui PROPERTIES WIN32_EXECUTABLE ON)

target_link_libraries(cuvistaGui PRIVATE
    Common
    Matrix 
    CudaStuff 
    Deshaker 
    "${CORE_LIBS}"
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
)

install(TARGETS
cuvistaGui
)
qt_generate_deploy_app_script(
    TARGET cuvistaGui
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT "${deploy_script}")
