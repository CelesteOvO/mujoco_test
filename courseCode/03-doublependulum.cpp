//
// Created by LiYifan on 2024/6/11.
//
// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Control a double pendulum 控制双摆

#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#define ndof 2 //number of degrees of freedom
//related to writing data to a file
double simend = 5; //simulation end time
FILE *fid;
int loop_index = 0;
const int data_frequency = 50; //frequency at which data is written to a file
char datapath[] = "doublependulum_data.csv";


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

///double pendulum

void init_save_data()
{
    //write name of the variable here (header)
    fprintf(fid,"t, ");
    fprintf(fid,"PE, KE, TE, ");
    fprintf(fid,"q1, q2, ");

    //Don't remove the newline
    fprintf(fid,"\n");
}

void save_data(const mjModel* m, mjData* d)
{
    //data here should correspond to headers in init_save_data()
    //seperate data by a space %f followed by space
    fprintf(fid,"%f, ",d->time);
    fprintf(fid,"%f, %f, %f, ",d->energy[0],d->energy[1],d->energy[0]+d->energy[1]);
    fprintf(fid,"%f, %f ",d->qpos[0],d->qpos[1]);
    //Don't remove the newline
    fprintf(fid,"\n");
}

void mycontroller(const mjModel* m, mjData* d) {
    //write control here
    mj_energyPos(m, d);
    mj_energyVel(m, d);
    printf("%f %f %f %f \n",d->time,d->energy[0],d->energy[1],d->energy[0]+d->energy[1]);

    // check equations
    // M*qacc + qfrc_bias = qfrc_applied + ctrl
//    M: 是质量矩阵（也称为惯性矩阵），它表示系统的质量分布。在多自由度系统中，这是一个矩阵，而在单一自由度系统中，这就是质量的标量值。
//    qacc: 代表关节（或系统）的加速度，是我们要求解的未知量之一。
//    qfrc_bias: 由于系统中的各种非保守力（如摩擦力、科里奥利力等）产生的偏置力。它也包括了重力的效应。
//    qfrc_applied: 代表系统所受到的外部施加力，可以是由用户输入或环境因素（如碰撞）产生的力。
//    ctrl: 控制输入，即我们通过控制器施加到系统以执行特定任务（如移动到特定位置、跟踪特定轨迹等）的力或扭矩。
//   方程 M*qacc + qfrc_bias = qfrc_applied + ctrl本质上是牛顿第二定律(F=ma)的一个变种，适用于多自由度机械系统。
//   这里面的“力”不仅包括外部作用力（qfrc_applied和ctrl），还有因系统内部机制产生的偏置力（qfrc_bias）。
//   动力学方程的目的是找到系统的加速度qacc，这样就可以通过积分来获得系统的速度和位置，进而进行模拟或控制。
    //const int ndof = 2; 两个自由度
    double dense_M[ndof * ndof] = {0};
    // 显示质量矩阵
    mj_fullM(m,dense_M, d->qM);
    double M[ndof][ndof]={0};
    M[0][0] = dense_M[0];
    M[0][1] = dense_M[1];
    M[1][0] = dense_M[2];
    M[1][1] = dense_M[3];
    // printf("%f %f \n",M[0][0],M[0][1]);
    // printf("%f %f \n",M[1][0],M[1][1]);

    // 得到加速度
    double qddot[ndof]={0};
    qddot[0]=d->qacc[0];
    qddot[1]=d->qacc[1];

    // M * qddot + f = qfrc_applied + ctrl 和前面那个公式一样

    double f[ndof]={0};
    f[0] = d->qfrc_bias[0];
    f[1] = d->qfrc_bias[1];

    // 计算等式左边
    double lhs[ndof]={0};
    mju_mulMatVec(lhs,dense_M,qddot,2,2); //lhs = M*qddot
    lhs[0] = lhs[0] + f[0]; //lhs = M*qddot + f
    lhs[1] = lhs[1] + f[1];

    // printf("%f %f \n",lhs[0], lhs[1]);

    // qfrc_applied 和 f 抵消, 所以不会运动
    // 两个关节上的力恰好等于外力
    /*d->qfrc_applied[0] = f[0];
    d->qfrc_applied[1] = f[1];*/

    d->qfrc_applied[0] = 0.1*f[0];
    d->qfrc_applied[1] = 0.5*f[1];

    // 等式右边
    double rhs[ndof]={0};
    rhs[0] = d->qfrc_applied[0];
    rhs[1] = d->qfrc_applied[1];

    // 可以看到一行的两个数是相等的
    // printf("%f %f \n",lhs[0], rhs[0]);
    // printf("%f %f \n",lhs[1], rhs[1]);

    // control
    double Kp1 = 100, Kp2 = 100;
    double Kv1 = 10, Kv2 = 10;
    double qref1 = -0.5, qref2 = -1.6;

    //PD control
    // d->qfrc_applied[0] = -Kp1*(d->qpos[0]-qref1)-Kv1*d->qvel[0];
    // d->qfrc_applied[1] = -Kp2*(d->qpos[1]-qref2)-Kv2*d->qvel[1];

    //coriolis + gravity + PD control
    // d->qfrc_applied[0] = f[0]-Kp1*(d->qpos[0]-qref1)-Kv1*d->qvel[0];
    // d->qfrc_applied[1] = f[1]-Kp2*(d->qpos[1]-qref2)-Kv2*d->qvel[1];

    //Feedback linearization
    //M*(-kp( ... ) - kv(...) + f)
    double tau[2]={0};
    tau[0]=-Kp1*(d->qpos[0]-qref1)-Kv1*d->qvel[0];
    tau[1]=-Kp2*(d->qpos[1]-qref2)-Kv2*d->qvel[1];

    mju_mulMatVec(tau,dense_M,tau,2,2); //lhs = M*tau
    tau[0] += f[0];
    tau[1] += f[1];
    d->qfrc_applied[0] = tau[0];
    d->qfrc_applied[1] = tau[1];

    if ( loop_index % data_frequency == 0)
        save_data(m,d);
    loop_index = loop_index + 1;
}
///double pendulum

// main function
int main(int argc, const char** argv) {
    // check command-line arguments
    std::printf( "argc: \n");

//  if (argc!=2) {
//    std::printf(" USAGE:  basic modelfile\n");
//    return 0;
//  }

    ///double pendulum
    //argv[1] = "hello.xml";
    argv[1] = "doublependulum.xml";
    ///double pendulum

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

    ///double pendulum
    double arr_view[] = {89.608063, -11.588379, 5, 0.000000, 0.000000, 2.000000};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    // install control callback
    mjcb_control = mycontroller;

    fid = fopen(datapath,"w");
    init_save_data();

    d->qpos[0] = 0.5;
    //d->qpos[1] = 0;
    ///double pendulum

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

        ///double pendulum
        if (d->time>=simend)
        {
            fclose(fid);
            break;
        }
        ///double pendulum

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
