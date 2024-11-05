//
// Created by LiYifan on 2024/6/11.
//

// Control a simple pendulum 控制简单的摆

//伺服控制（Servo Control）
//伺服控制是一种在自动控制系统中广泛使用的控制机制，旨在使系统的输出（例如电机的位置、速度或扭瘦）精确地跟踪或达到某个期望的值或轨迹。伺服控制系统通常包含：
//
//反馈元素：用于监测系统的当前状态，如位置、速度等。
//控制器：基于反馈信号与期望值之间的差异，计算出控制信号。
//执行机构：根据控制器的指令，如电机，来调整系统状态，推动系统向期望状态转移。
//伺服控制的关键特点是反馈控制机制，它使得系统能够自动地调整控制动作来修正任何偏离预定目标的行为。
//
//PD控制（Proportional-Derivative Control）
//PD控制是一种简单而有效的反馈控制策略，它通过比例（P）和微分（D）两个主要元素来构成其控制逻辑。
//PD控制的目的是减少系统输出与目标值之间的差距，通常用于减少系统的稳态误差（通过比例控制）并提高系统的响应速度与稳定性（通过微分控制）。
//PD控制器的控制规律一般形式为：
//[ u(t) = K_p e(t) + K_d \frac{de(t)}{dt} ]
//
//( u(t) ) 是控制器输出，
//( K_p ) 是比例增益，
//( e(t) ) 是控制偏差，即期望值与实际值之间的差距，
//( K_d ) 是微分增益，
//( \frac{de(t)}{dt} ) 是偏差的变化率，即误差的微分。
//PD控制的一个关键优点是它的简单性，使其容易实现和调节。比例项有助于减小误差，而微分项则提供对系统变化的预测，有助于减少或消除振荡，提升系统的稳定性。

#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right) {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

/// pendulum
void set_torque_control(const mjModel* m, int actuator_no, int flag)
{
//    当flag为0时，执行器的gainprm（增益参数）被设置为0,意味着该执行器不会对模型施加任何额外的力或扭矩，
//    这种模式通常用于模拟自由运动，或者在外部直接通过程序设定执行器力的情况下使用。
//    当flag不为0时（比如为1），执行器的gainprm被设置为1。这意味着执行器将根据其它配置参数正常工作，
//    且具备基本的增益，使得可以通过一定的控制策略来操控模型。这在实现一些基本控制逻辑，如角速度控制、位置控制等场景下很有用。
    if (flag==0)
        m->actuator_gainprm[10*actuator_no+0] = 0;
    else
        m->actuator_gainprm[10*actuator_no+0] = 1;
}

void set_velocity_servo(const mjModel* m, int actuator_no, double kv) // 调节执行器的速度以达到期望值。
{
    m->actuator_gainprm[10*actuator_no+0] = kv;
    m->actuator_biasprm[10*actuator_no+2] = -kv;
}

void set_position_servo(const mjModel* m, int actuator_no, double kp) // 调整执行器的位置以匹配给定的目标。
{
    m->actuator_gainprm[10*actuator_no+0] = kp;
    m->actuator_biasprm[10*actuator_no+1] = -kp;
//    actuator_gainprm: 这个数组用于设置执行器的增益参数。在这里，kp是增益值，它是反馈回路中用于决定执行器输出力度的一个系数。通过改变kp的值，你可以控制执行器反应的敏感程度。
//    actuator_biasprm: 这个数组则用于设置执行器的偏差参数。-kp在这里作为一个负反馈，可能用于创造一个期望的偏差或补偿偏移。
}

void mycontroller(const mjModel* m, mjData* d) // 每个模拟步骤，mycontroller函数都会被调用，以便对模型施加控制。
{
    int i;
    int actuator_no; // 一个索引或标识符，用于指定我们正在配置或控制的是哪一个执行器（例如电机或驱动器）。
    //0 = torque actuator
    // 使用PD控制计算控制力矩并应用到此执行器上。这里使用的是一个简单的PD控制公式
    // -10*(d->qpos[0]-0)-1*d->qvel[0]，其中d->qpos[0]是当前位置，d->qvel[0]是当前速度。
    actuator_no = 0;
    int flag = 0;
    set_torque_control(m, actuator_no, flag);
    d->ctrl[0] = -10*(d->qpos[0]-0)-1*d->qvel[0];//PD control
    //这里的-10*(d->qpos[0]-0)部分是位置控制的比例项，-1*d->qvel[0]部分是速度控制的微分项。
    // 对于-10*(d->qpos[0]-0)，0可以被看作是目标位置，这段代码试图将关节（或执行器控制的物体）控制到原点位置（即d->qpos[0]为0的点）。
    //d->ctrl[0] = -10*(d->sensordata[0]-0)-1*d->sensordata[1];

    //1 = position servo
//    选择第二个执行器(actuator_no = 1)，并设置其位置控制增益为0（本质上不控制位置）。
//    然后，利用PD控制重新设置，增益设为10，这意味着位置控制被启用，并且期望达到一个特定的位置。
    actuator_no = 1;
    double kp = 0;
    set_position_servo(m, actuator_no, kp);
    //for (i=0;i<10;i++)
    // {
    //   //printf("%f \n", m->actuator_gainprm[10*actuator_no+i]);
    //   //printf("%f \n", m->actuator_biasprm[10*actuator_no+i]);
    // }

    //printf("*********** \n");
    d->ctrl[1] = 0.5; // 目标位置

    //2= velocity servo
    //执行器会尝试将速度控制在一个给定的速度值。
    actuator_no = 2;
    double kv = 0;
    set_velocity_servo(m, actuator_no, kv);
    d->ctrl[2] = 0.2; // 目标速度

    //PD control
    actuator_no = 1;
    double kp2 = 10;
    set_position_servo(m, actuator_no, kp2);
    actuator_no = 2;
    double kv2 = 1;
    set_velocity_servo(m, actuator_no, kv2);
    d->ctrl[1] = -0.5;
    d->ctrl[2] = 0;
}
/// pendulum

// main function
int main(int argc, const char** argv) {
    // check command-line arguments
    std::printf( "argc: \n");

//  if (argc!=2) {
//    std::printf(" USAGE:  basic modelfile\n");
//    return 0;
//  }

/// pendulum
    //argv[1] = "hello.xml";
    argv[1] = "pendulum.xml";
/// pendulum

    // load and compile model
    char error[1000] = "Could not load binary model";
    if (std::strlen(argv[1])>4 && !std::strcmp(argv[1]+std::strlen(argv[1])-4, ".mjb")) {
        m = mj_loadModel(argv[1], 0);
    } else {
        m = mj_loadXML(argv[1], 0, error, 1000);
    }
    if (!m) {
        mju_error("Load model error: %s", error);
    }

    // make data
    d = mj_makeData(m);

    // init GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    /// pendulum
    double arr_view[] = {90, -5, 5, 0.012768, -0.000000, 1.254336};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    d->qpos[0]=1.57; //pi/2
    mjcb_control = mycontroller;
    /// pendulum

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window)) {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0/60.0) {
            mj_step(m, d);
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return 1;
}
