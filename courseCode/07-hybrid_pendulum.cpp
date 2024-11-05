#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

// Gymnast swing/release on a bar

//simulation end time
double simend = 5;

int fsm;
#define fsm_swing 0
#define fsm_free 1

//related to writing data to a file
FILE *fid;
int loop_index = 0;
const int data_frequency = 10; //frequency at which data is written to a file
char datapath[] = "hybrid_pendulum.csv";

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

/// hybrid_pendulum
void init_save_data()
{
    //write name of the variable here (header)
    fprintf(fid,"t, ");

    //Don't remove the newline
    fprintf(fid,"\n");
}

void save_data(const mjModel* m, mjData* d)
{
    //data here should correspond to headers in init_save_data()
    //seperate data by a space %f followed by space
    fprintf(fid,"%f ",d->time);

    //Don't remove the newline
    fprintf(fid,"\n");
}
/// hybrid_pendulum

/// hybrid_pendulum
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

void init_controller(const mjModel* m, mjData* d)
{
    fsm = fsm_swing;
}

void mycontroller(const mjModel* m, mjData* d)
{
    //write control here

    const int nv = 6;
    const int nv1 = 3;
    double M[nv*nv]={0};
    mj_fullM(m,M,d->qM);
    //original system
    // M0 M1 M2
    // M3 M4 M5
    // M6 M7 M8

    //two pendulums
    // M0 M1 M2 M3 M4 M5
    // M6 M7 M8 M9 M10 M11
    // M12 M13 M14 M15 M16 M17
    // ...
    // ..
    // ...
    // printf("%f %f %f \n",M[0],M[1],M[2]);
    // printf("%f %f %f \n",M[3],M[4],M[5]);
    // printf("%f %f %f \n",M[6],M[7],M[8]);
    double M1[nv1*nv1]={0};
    M1[0] = M[0];  M1[1] = M[1];  M1[2] = M[2];
    M1[3] = M[6];  M1[4] = M[7];  M1[5] = M[8];
    M1[6] = M[12];  M1[7] = M[13];  M1[8] = M[14];

    double qddot[nv]={0};
    qddot[0] = d->qacc[0];
    qddot[1] = d->qacc[1];
    qddot[2] = d->qacc[2];

    double f[nv]={0};
    f[0] = d->qfrc_bias[0];
    f[1] = d->qfrc_bias[1];
    f[2] = d->qfrc_bias[2];

    double lhs[3]={0};
    double M_qddot[3]={0}; //=M*qddot
    mju_mulMatVec(M_qddot,M1,qddot,3,3);
    //lhs = M*qddot + f
    lhs[0]=M_qddot[0]+f[0];
    lhs[1]=M_qddot[1]+f[1];
    lhs[2]=M_qddot[2]+f[2];

    //Fx, Fy, tau_y
    double tau[nv] = {0};
    tau[0] = d->qfrc_applied[0]; //Fx
    tau[1] = d->qfrc_applied[1]; //Fz
    tau[2] = d->qfrc_applied[2]; //tau_y

    int i;
    double J0[3*nv]={0};
    for (i=0;i<3*nv;i++)
        J0[i] = d->efc_J[i];



    double JT_F[3]={0};
    double J1[nv1*nv1]={0};
    J1[0] = J0[0];  J1[1] = J0[1];  J1[2] = J0[2];
    J1[3] = J0[6];  J1[4] = J0[7];  J1[5] = J0[8];
    J1[6] = J0[12];  J1[7] = J0[13];  J1[8] = J0[14];
    //J[3*nv] = J[3x6]
    // J0 J1 J2 J3 J4 J5
    // J6 J7 J8 J9 J10 J11
    // J12 J13 J14 J15 J16 J17

    double F0[3]={0};
    F0[0] = d->efc_force[0];
    F0[1] = d->efc_force[1];
    F0[2] = d->efc_force[2];
    mju_mulMatTVec(JT_F,J1,F0,3,3);

    // mju_mulMatTVec(JT_F,J0,F0,3,3);

    // double F0[3]={0};
    // F0[0] = d->efc_force[0];
    // F0[1] = d->efc_force[1];
    // F0[2] = d->efc_force[2];
    // mju_mulMatTVec(JT_F,J1,F0,3,3);
    // double rhs[3] = {0};
    // rhs[0] = tau[0] + JT_F[0];
    // rhs[1] = tau[1] + JT_F[1];
    // rhs[2] = tau[2] + JT_F[2];

    //verify equations
    // printf("eqn1: %f %f \n",lhs[0],rhs[0]);
    // printf("eqn2: %f %f \n",lhs[1],rhs[1]);
    // printf("eqn3: %f %f \n",lhs[2],rhs[2]);

    //transitions
    if (fsm==fsm_swing && d->qpos[5]>1)
    {
        fsm = fsm_free;
    }

    //actions
    if (fsm==fsm_swing)
    {
        d->qfrc_applied[2] = -1*(d->qvel[2]-5);



        d->qfrc_applied[3] =  JT_F[0];
        d->qfrc_applied[4] =  JT_F[1];
        d->qfrc_applied[5] =  JT_F[2] + d->qfrc_applied[2];
    }
    if (fsm==fsm_free)
    {
        d->qfrc_applied[3] =  0;
        d->qfrc_applied[4] =  0;
        d->qfrc_applied[5] =  0;
    }

    //printf("*****\n");
    //write data here (dont change/dete this function call; instead write what you need to save in save_data)
    if ( loop_index%data_frequency==0)
    {
        save_data(m,d);
    }
    loop_index = loop_index + 1;
}
/// hybrid_pendulum

// main function
int main(int argc, const char** argv) {
    // check command-line arguments
    std::printf( "argc: \n");

//  if (argc!=2) {
//    std::printf(" USAGE:  basic modelfile\n");
//    return 0;
//  }

/// hybrid_pendulum
    //argv[1] = "hello.xml";
    argv[1] = "hybrid_pendulum.xml";
/// hybrid_pendulum
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

    /// hybrid_pendulum
    double arr_view[] = {89.608063, -11.588379, 7, 0.000000, 0.000000, 2.000000}; //view the left side (for ll, lh, left_side)
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
    init_controller(m,d);

    d->qpos[2] = -1.57;
    d->qpos[5] = d->qpos[2];
    /// hybrid_pendulum

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

        /// hybrid_pendulum
        if (d->time>=simend)
        {
            fclose(fid);
            break;
        }
        /// hybrid_pendulum

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
