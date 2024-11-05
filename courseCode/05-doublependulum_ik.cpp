#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

// Manipulator drawing 机械手画圆

//simulation end time
double simend = 20;
double qinit[2] = {0,1.25};
double r = 0.5; // 圆的半径
double omega = 0.5;

double x_0, y_0;

//related to writing data to a file
FILE *fid;
int loop_index = 0;
const int data_frequency = 10; //frequency at which data is written to a file
char datapath[] = "doublependulum_ik_data.csv";

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

/// double pendulum fsm
void init_save_data()
{
    //write name of the variable here (header)
    fprintf(fid,"t, ");
    fprintf(fid,"x, y ");

    //Don't remove the newline
    fprintf(fid,"\n");
}

void save_data(const mjModel* m, mjData* d)
{
    //data here should correspond to headers in init_save_data()
    //seperate data by a space %f followed by space
    fprintf(fid,"%f, ",d->time);
    fprintf(fid,"%f, %f ",d->sensordata[0],d->sensordata[2]);

    //Don't remove the newline
    fprintf(fid,"\n");
}
/// double pendulum fsm

/// double pendulum fsm
void set_position_servo(const mjModel* m,int actuator_no,double kp)
{
    m->actuator_gainprm[10*actuator_no+0]=kp;
    m->actuator_biasprm[10*actuator_no+1]=-kp;
}

void set_velocity_servo(const mjModel* m,int actuator_no,double kv)
{
    m->actuator_gainprm[10*actuator_no+0]=kv;
    m->actuator_biasprm[10*actuator_no+2]=-kv;
}

// 初始化控制器
void init_controller(const mjModel* m, mjData* d)
{
    //mj_step(m,d);
    mj_forward(m,d);
    printf("position = %f %f \n",d->sensordata[0],d->sensordata[2]);

    //x0+r = d->sensordata[0];
    //y0 = d->sensordata[2]

    x_0 = d->sensordata[0] - r;
    y_0 = d->sensordata[2];
}

void mycontroller(const mjModel* m, mjData* d)
{
    //write control here
    //printf("position = %f %f %f \n",d->sensordata[0],d->sensordata[1],d->sensordata[2]);
    //printf("velocity = %f %f %f \n",d->sensordata[3],d->sensordata[4],d->sensordata[5]);

    //void mj_jac(const mjModel* m, const mjData* d,mjtNum* jacp, mjtNum* jacr, const mjtNum point[3], int body);
    double jacp[6]={0};
    double point[3]={d->sensordata[0],d->sensordata[1],d->sensordata[2]};
    int body = 2;
    mj_jac(m,d,jacp,NULL,point,body);
    // printf("J = \n");//3x2
    // printf("%f %f \n", jacp[0],jacp[1]);
    // printf("%f %f \n", jacp[2],jacp[3]);
    // printf("%f %f \n", jacp[4],jacp[5]);

    // 计算速度的Jacobian
    double J[4]={ jacp[0],jacp[1],jacp[4],jacp[5]}; // 摆在xz平面上运动,和y没有关系,打印下来的值都是0,所以不使用2 3
    // 计算等式右边的qdot
    double qdot[2] = {d->qvel[0],d->qvel[1]};
    double xdot[2] ={0};
    //xdot = J*qdot 计算J和qdot的乘积
    mju_mulMatVec(xdot,J,qdot,2,2);
    // printf("velocity using jacobian: %f %f \n",xdot[0],xdot[1]);
    // printf("velocity using sensordata= %f %f \n",d->sensordata[3],d->sensordata[5]);
    // 上面这两行的数字应该是相同的

    // d->ctrl[0] = qinit[0];
    // d->ctrl[2] = qinit[1];

    int i;
    // 计算J的逆矩阵
    double det_J = J[0]*J[3]-J[1]*J[2];
    double J_temp[] = {J[3],-J[1],-J[2],J[0]};
    double J_inv[4]={};
    for (i=0;i<4;i++)
        J_inv[i] = J_temp[i]/det_J;

    // 根据圆的方程计算xy
    double x,y;
    x = x_0 + r*cos(omega*d->time);
    y = y_0 + r*sin(omega*d->time);

    // x减去当前传感器的位置
    double dr[] = {x- d->sensordata[0],y - d->sensordata[2]};
    double dq[2] ={};

    //dq = Jinv*dr
    mju_mulMatVec(dq,J_inv,dr,2,2);
    printf("%f %f \n", dq[0],dq[1]);

    //q -> q+dq
    //ctrl = q 更新q
    d->ctrl[0] = d->qpos[0]+dq[0];
    d->ctrl[2] = d->qpos[1]+dq[1];

    //write data here (dont change/dete this function call; instead write what you need to save in save_data)
    if ( loop_index%data_frequency==0)
    {
        save_data(m,d);
    }
    loop_index = loop_index + 1;
}
/// double pendulum fsm

// main function
int main(int argc, const char** argv) {
    // check command-line arguments
    std::printf( "argc: \n");

//  if (argc!=2) {
//    std::printf(" USAGE:  basic modelfile\n");
//    return 0;
//  }

/// double pendulum fsm
    //argv[1] = "hello.xml";
    argv[1] = "doublependulum_ik.xml";
/// double pendulum fsm
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

    /// double pendulum fsm
    double arr_view[] = {89.608063, -11.588379, 5, 0.000000, 0.000000, 1.000000};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    // install control callback
    mjcb_control = mycontroller;
    d->qpos[0] = qinit[0];
    d->qpos[1] = qinit[1];

    fid = fopen(datapath,"w");
    init_save_data();
    init_controller(m,d);
    /// double pendulum fsm

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

        /// double pendulum fsm
        if (d->time>=simend)
        {
            fclose(fid);
            break;
        }
        /// double pendulum fsm

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
