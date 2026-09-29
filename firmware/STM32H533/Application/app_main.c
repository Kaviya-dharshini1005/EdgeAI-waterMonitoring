#include <tk/tkernel.h>
#include <tm/tmonitor.h>

#include "autoencoder.h"
#include "threshold.h"
#include "normalization.h"
#include "main.h"
#include "stm32h5xx_hal_adc.h"

extern ADC_HandleTypeDef hadc1;


/* ================================================================
   SYSTEM STATES
   ================================================================ */

#define STATE_SAFE        0
#define STATE_ALERT       1
#define STATE_CRITICAL    2


/* ================================================================
   SYSTEM PARAMETERS
   ================================================================ */

#define HISTORY_SIZE      10
#define MSG_POOL_SIZE     8



/* Slope thresholds */
#define PH_SLOPE_THRESHOLD     0.30f
#define TDS_SLOPE_THRESHOLD    50.0f
#define TURB_SLOPE_THRESHOLD   1.0f


/* ================================================================
   DATA NORMALIZATION PARAMETERS
   These values correspond to the training dataset preprocessing.
   ================================================================ */

#define PH_MIN            0.22749905f
#define PH_MAX            13.17540172f

#define TDS_MIN           728.75082960f
#define TDS_MAX           56488.67241000f

#define TURBIDITY_MIN     1.49220661f
#define TURBIDITY_MAX     6.49424947f


/* ================================================================
   pH SENSOR CALIBRATION
   ================================================================ */

#define PH_NEUTRAL_VOLTAGE   0.46f
#define PH_SLOPE              0.09f


/* ================================================================
   OPTIONAL SIMULATION MODE
   0 = REAL SENSORS
   1 = SIMULATION
   ================================================================ */

LOCAL volatile INT simulation_mode = 0;

LOCAL float simulated_pH        = 7.1f;
LOCAL float simulated_TDS       = 180.0f;
LOCAL float simulated_turbidity = 1.2f;


/* ================================================================
   SENSOR MESSAGE

   T_MSG MUST BE THE FIRST MEMBER because the object is sent
   through the μT-Kernel mailbox.
   ================================================================ */

typedef struct {

    T_MSG   msg;

    float   pH;
    float   TDS;
    float   turbidity;

    INT     state;

    /*
       0 = no alert
       1 = slope detection
       2 = TinyML anomaly
    */
    INT     alert_reason;

} SENSOR_MSG;


/* ================================================================
   GLOBAL SYSTEM STATE
   ================================================================ */

LOCAL INT current_state = STATE_SAFE;

LOCAL INT alert_count = 0;
LOCAL INT safe_count  = 0;


/* ================================================================
   SENSOR HISTORY
   ================================================================ */

LOCAL float pH_history[HISTORY_SIZE] = {

    7.1f, 7.1f, 7.1f, 7.1f, 7.1f,
    7.1f, 7.1f, 7.1f, 7.1f, 7.1f

};

LOCAL float TDS_history[HISTORY_SIZE] = {

    180.0f, 180.0f, 180.0f, 180.0f, 180.0f,
    180.0f, 180.0f, 180.0f, 180.0f, 180.0f

};

LOCAL float turb_history[HISTORY_SIZE] = {

    1.2f, 1.2f, 1.2f, 1.2f, 1.2f,
    1.2f, 1.2f, 1.2f, 1.2f, 1.2f

};


/*
   history_index points to the NEXT location to be written.
*/

LOCAL INT history_index = 0;


/* ================================================================
   TASK IDS
   ================================================================ */

LOCAL ID tskid_sensor;
LOCAL ID tskid_process;
LOCAL ID tskid_comm;
LOCAL ID tskid_power;


/* ================================================================
   MAILBOX IDS
   ================================================================ */

LOCAL ID mbxid_sensor;
LOCAL ID mbxid_comm;


/* ================================================================
   SYNCHRONIZATION OBJECTS
   ================================================================ */

LOCAL ID mtx_history;
LOCAL ID sem_msgpool;


/* ================================================================
   MESSAGE POOL

   Each sensor message occupies one slot.

   The semaphore initially contains MSG_POOL_SIZE resources.
   SensorReadTask acquires one slot.
   CommunicationTask releases it after the message is completely
   consumed.
   ================================================================ */

LOCAL SENSOR_MSG msg_pool[MSG_POOL_SIZE];

LOCAL INT msg_next = 0;


/* ================================================================
   TASK FUNCTION DECLARATIONS
   ================================================================ */

LOCAL void SensorReadTask(INT stacd, void *exinf);

LOCAL void DataProcessingTask(INT stacd, void *exinf);

LOCAL void CommTask(INT stacd, void *exinf);

LOCAL void PowerSaveTask(INT stacd, void *exinf);


/* ================================================================
   TASK CREATION STRUCTURES
   ================================================================ */

LOCAL T_CTSK ctsk_sensor = {

    .itskpri = 8,
    .stksz   = 2048,
    .task    = SensorReadTask,
    .tskatr  = TA_HLNG | TA_RNG3

};


LOCAL T_CTSK ctsk_process = {

    .itskpri = 6,
    .stksz   = 2048,
    .task    = DataProcessingTask,
    .tskatr  = TA_HLNG | TA_RNG3

};


LOCAL T_CTSK ctsk_comm = {

    .itskpri = 5,
    .stksz   = 1024,
    .task    = CommTask,
    .tskatr  = TA_HLNG | TA_RNG3

};


LOCAL T_CTSK ctsk_power = {

    .itskpri = 4,
    .stksz   = 512,
    .task    = PowerSaveTask,
    .tskatr  = TA_HLNG | TA_RNG3

};


/* ================================================================
   ADC DRIVER
   ================================================================ */

LOCAL uint32_t read_adc_channel(uint32_t channel)

{

    ADC_ChannelConfTypeDef sConfig = {0};

    sConfig.Channel      = channel;

    sConfig.Rank         = ADC_REGULAR_RANK_1;

    sConfig.SamplingTime = ADC_SAMPLETIME_92CYCLES_5;


    HAL_ADC_ConfigChannel(&hadc1, &sConfig);


    uint32_t sum = 0;


    for (INT i = 0; i < 10; i++)

    {

        HAL_ADC_Start(&hadc1);


        if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK)

        {

            sum += HAL_ADC_GetValue(&hadc1);

        }


        HAL_ADC_Stop(&hadc1);

    }


    return sum / 10;

}


/* ================================================================
   pH SENSOR
   ================================================================ */

LOCAL float read_pH(void)

{

    if (simulation_mode)

        return simulated_pH;


    uint32_t raw =
        read_adc_channel(ADC_CHANNEL_0);


    float voltage =
        (raw / 4095.0f) * 3.3f;


    /*
       The hardware voltage divider attenuates the sensor output.
       Recover the approximate original sensor voltage.
    */

    float v_actual_sensor =
        voltage * 2.0f;


    float pH =
        7.0f -
        ((v_actual_sensor - PH_NEUTRAL_VOLTAGE)
        / PH_SLOPE);


    if (pH < 0.0f)

        pH = 0.0f;


    if (pH > 14.0f)

        pH = 14.0f;


    return pH;

}


/* ================================================================
   TDS SENSOR
   ================================================================ */

LOCAL float read_TDS(void)

{

    if (simulation_mode)

        return simulated_TDS;


    uint32_t raw =
        read_adc_channel(ADC_CHANNEL_1);


    float voltage =
        (raw / 4095.0f) * 3.3f;


    float tds =
        voltage * 500.0f;


    return tds;

}


/* ================================================================
   TURBIDITY SENSOR
   ================================================================ */

LOCAL float read_turbidity(void)

{

    if (simulation_mode)

        return simulated_turbidity;


    uint32_t raw =
        read_adc_channel(ADC_CHANNEL_5);


    float voltage =
        (raw / 4095.0f) * 3.3f;


    float ntu =

        TURBIDITY_MAX -

        (voltage / 2.1f) *
        (TURBIDITY_MAX - TURBIDITY_MIN);


    if (ntu < TURBIDITY_MIN)

        ntu = TURBIDITY_MIN;


    if (ntu > TURBIDITY_MAX)

        ntu = TURBIDITY_MAX;


    return ntu;

}


/* ================================================================
   SLOPE / TREND DETECTION

   Compares:
       latest 3 samples
   against:
       previous 3 samples

   The absolute difference between their averages is returned.
   ================================================================ */

LOCAL float calculate_slope(float *buf)

{

    float latest_sum = 0.0f;

    float prior_sum  = 0.0f;


    tk_loc_mtx(
        mtx_history,
        TMO_FEVR
    );


    /*
       history_index points to the next slot.

       Therefore:
       newest sample = history_index - 1
    */

    INT newest_idx =

        (history_index - 1 + HISTORY_SIZE)
        % HISTORY_SIZE;


    /*
       Latest three chronological samples.
    */

    for (INT i = 0; i < 3; i++)

    {

        INT idx =

            (newest_idx - i + HISTORY_SIZE)
            % HISTORY_SIZE;


        latest_sum += buf[idx];

    }


    /*
       Three samples immediately before the latest group.
    */

    for (INT i = 0; i < 3; i++)

    {

        INT idx =

            (newest_idx - 3 - i
            + HISTORY_SIZE * 2)
            % HISTORY_SIZE;


        prior_sum += buf[idx];

    }


    tk_unl_mtx(mtx_history);


    float latest_avg =
        latest_sum / 3.0f;


    float prior_avg =
        prior_sum / 3.0f;


    float diff =
        latest_avg - prior_avg;


    if (diff < 0.0f)

        diff = -diff;


    return diff;

}


/* ================================================================
   RTOS STATE: ALERT
   ================================================================ */

LOCAL void enter_alert_state(void)

{

    current_state = STATE_ALERT;


    tm_putstring(

        (UB*)
        "[RTOS-STATE] ALERT: "
        "Increasing monitoring priority\n"

    );


    tk_chg_pri(
        tskid_sensor,
        2
    );


    tk_chg_pri(
        tskid_process,
        2
    );


    tk_chg_pri(
        tskid_comm,
        2
    );


    /*
       Stop the low-priority background task while the
       system is responding to an abnormal condition.
    */

    tk_sus_tsk(
        tskid_power
    );

}


/* ================================================================
   RTOS STATE: CRITICAL
   ================================================================ */

LOCAL void enter_critical_state(void)

{

    current_state =
        STATE_CRITICAL;


    tm_putstring(

        (UB*)
        "[RTOS-STATE] CRITICAL: "
        "Maximum monitoring priority\n"

    );


    tk_chg_pri(
        tskid_sensor,
        1
    );


    tk_chg_pri(
        tskid_process,
        1
    );


    tk_chg_pri(
        tskid_comm,
        1
    );

}


/* ================================================================
   RTOS STATE: SAFE
   ================================================================ */

LOCAL void enter_safe_state(void)

{

    current_state =
        STATE_SAFE;


    alert_count = 0;


    tm_putstring(

        (UB*)
        "[RTOS-STATE] SAFE: "
        "Returning to baseline schedule\n"

    );


    tk_chg_pri(
        tskid_sensor,
        8
    );


    tk_chg_pri(
        tskid_process,
        6
    );


    tk_chg_pri(
        tskid_comm,
        5
    );


    /*
       Resume background task.
    */

    tk_rsm_tsk(
        tskid_power
    );

}


/* ================================================================
   TASK 1
   SENSOR READ TASK
   ================================================================ */

LOCAL void SensorReadTask(
    INT stacd,
    void *exinf
)

{

    while (1)

    {

        /*
           Acquire one free message-pool slot.
        */

        tk_wai_sem(
            sem_msgpool,
            1,
            TMO_FEVR
        );


        INT slot =

            msg_next % MSG_POOL_SIZE;


        msg_next++;


        SENSOR_MSG *m =
            &msg_pool[slot];


        /*
           Acquire sensors.
        */

        m->pH =
            read_pH();


        m->TDS =
            read_TDS();


        m->turbidity =
            read_turbidity();


        m->state =
            current_state;


        m->alert_reason =
            0;


        /*
           Update history under mutex.
        */

        tk_loc_mtx(
            mtx_history,
            TMO_FEVR
        );


        pH_history[history_index] =
            m->pH;


        TDS_history[history_index] =
            m->TDS;


        turb_history[history_index] =
            m->turbidity;


        history_index =

            (history_index + 1)
            % HISTORY_SIZE;


        tk_unl_mtx(
            mtx_history
        );


        /*
           Send sensor message to processing task.
        */

        tk_snd_mbx(

            mbxid_sensor,

            (T_MSG*)m

        );


        /*
           Adaptive sampling.
        */

        if (current_state ==
            STATE_SAFE)

        {

            /*
               Normal monitoring.
            */

            tk_dly_tsk(2000);

        }

        else if (current_state ==
                 STATE_ALERT)

        {

            /*
               Faster monitoring.
            */

            tk_dly_tsk(500);

        }

        else

        {

            /*
               Critical monitoring.
            */

            tk_dly_tsk(100);

        }

    }

}


/* ================================================================
   TASK 2
   DATA PROCESSING TASK

   Performs:

   1. Normalization
   2. TinyML inference
   3. Slope detection
   4. Persistence filtering
   5. State transition
   ================================================================ */

LOCAL void DataProcessingTask(
    INT stacd,
    void *exinf
)

{

    while (1)

    {

        SENSOR_MSG *m;


        /*
           Wait for sensor data.
        */

        tk_rcv_mbx(

            mbxid_sensor,

            (T_MSG**)&m,

            TMO_FEVR

        );


        /* ========================================================
           NORMALIZATION
           ======================================================== */




        /* ========================================================
           TINYML INPUT VECTOR

           IMPORTANT:
           This is an ARRAY, not a single float.
           ======================================================== */




        /* ========================================================
           TINYML AUTOENCODER
           ======================================================== */

        float reconstruction_error =
            AE_Predict(
                m->pH,
                m->TDS,
                m->turbidity
            );


        /* ========================================================
           ML ANOMALY DECISION
           ======================================================== */

        INT ml_anomaly =

            (reconstruction_error
             > ANOMALY_THRESHOLD)
             ? 1
             : 0;


        /* ========================================================
           SLOPE DETECTION
           ======================================================== */

        float ph_slope =

            calculate_slope(
                pH_history
            );


        float tds_slope =

            calculate_slope(
                TDS_history
            );


        float turb_slope =

            calculate_slope(
                turb_history
            );


        INT slope_anomaly = 0;


        /*
           Parameter-specific trend thresholds.
        */

        if (ph_slope >
            PH_SLOPE_THRESHOLD)

        {

            slope_anomaly = 1;

        }


        if (tds_slope >
            TDS_SLOPE_THRESHOLD)

        {

            slope_anomaly = 1;

        }


        if (turb_slope >
            TURB_SLOPE_THRESHOLD)

        {

            slope_anomaly = 1;

        }


        /* ========================================================
           DUAL DETECTION
           ======================================================== */

        if (slope_anomaly ||
            ml_anomaly)

        {

            /*
               Slope gets priority in the reason field if both
               mechanisms trigger simultaneously.
            */

            if (slope_anomaly)

                m->alert_reason = 1;

            else

                m->alert_reason = 2;


            /*
               Consecutive abnormal observation.
            */

            alert_count++;

            safe_count = 0;

        }

        else

        {

            /*
               A clean observation breaks the abnormal sequence.
            */

            alert_count = 0;

            safe_count++;


            /*
               Five consecutive safe observations restore SAFE.
            */

            if (safe_count >= 5 &&
                current_state != STATE_SAFE)

            {

                safe_count = 0;

                enter_safe_state();

            }

        }


        /* ========================================================
           STATE ESCALATION
           ======================================================== */

        if (alert_count >= 3 &&
            current_state == STATE_SAFE)

        {

            enter_alert_state();

        }


        else if (alert_count >= 7 &&
                 current_state == STATE_ALERT)

        {

            enter_critical_state();

        }


        /*
           Store the resulting state.
        */

        m->state =
            current_state;


        /*
           Send processed message to communication task.
        */

        tk_snd_mbx(

            mbxid_comm,

            (T_MSG*)m

        );

    }

}


/* ================================================================
   TASK 3
   COMMUNICATION TASK
   ================================================================ */

LOCAL void CommTask(
    INT stacd,
    void *exinf
)

{

    while (1)

    {

        SENSOR_MSG *m;


        /*
           Receive processed message.
        */

        tk_rcv_mbx(

            mbxid_comm,

            (T_MSG**)&m,

            TMO_FEVR

        );


        /* ========================================================
           SERIAL OUTPUT
           ======================================================== */

        INT pH_i =
            (INT)m->pH;


        INT pH_d =

            (INT)(
                (m->pH - pH_i)
                * 100.0f
            );


        INT TDS_i =
            (INT)m->TDS;


        INT TDS_d =

            (INT)(
                (m->TDS - TDS_i)
                * 100.0f
            );


        INT turb_i =
            (INT)m->turbidity;


        INT turb_d =

            (INT)(
                (m->turbidity - turb_i)
                * 100.0f
            );


        const UB *state_str;


        if (m->state ==
            STATE_SAFE)

        {

            state_str =
                (UB*)"SAFE";

        }

        else if (m->state ==
                 STATE_ALERT)

        {

            state_str =
                (UB*)"ALERT";

        }

        else

        {

            state_str =
                (UB*)"CRITICAL";

        }


        const UB *reason_str;


        if (m->alert_reason == 1)

        {

            reason_str =
                (UB*)" [SLOPE]";

        }

        else if (m->alert_reason == 2)

        {

            reason_str =
                (UB*)" [TINYML]";

        }

        else

        {

            reason_str =
                (UB*)"";

        }


        tm_printf(

            (UB*)
            "pH=%d.%02d  "
            "TDS=%d.%02d  "
            "Turb=%d.%02d  "
            "| %s%s\n",

            pH_i,
            pH_d,

            TDS_i,
            TDS_d,

            turb_i,
            turb_d,

            state_str,
            reason_str

        );


        /*
           Message has now been completely consumed.

           Return its pool slot to the semaphore.
        */

        tk_sig_sem(
            sem_msgpool,
            1
        );

    }

}


/* ================================================================
   TASK 4
   POWER MANAGEMENT TASK
   ================================================================ */

LOCAL void PowerSaveTask(
    INT stacd,
    void *exinf
)

{

    while (1)

    {

        tm_putstring(

            (UB*)
            "[POWER] Background system nominal - "
            "low-activity mode\n"

        );


        /*
           Background task sleeps for 10 seconds.
        */

        tk_dly_tsk(10000);

    }

}


/* ================================================================
   SYSTEM INITIALIZATION
   ================================================================ */

EXPORT INT usermain(void)

{

    tm_putstring(

        (UB*)
        "\n"
        "========================================\n"
        " SmartWater RTOS v1.0\n"
        " Adaptive Water Quality Monitoring\n"
        " STM32H533RE | uT-Kernel 3.0\n"
        "========================================\n"

    );


    /* ============================================================
       INITIALIZE AUTOENCODER
       ============================================================ */

    AE_Init();


    /* ============================================================
       CREATE SENSOR MAILBOX
       ============================================================ */

    T_CMBX cmbx = {

        .mbxatr =
            TA_TFIFO | TA_MFIFO

    };


    mbxid_sensor =
        tk_cre_mbx(&cmbx);


    mbxid_comm =
        tk_cre_mbx(&cmbx);


    if (mbxid_sensor < 0 ||
        mbxid_comm < 0)

    {

        tm_putstring(

            (UB*)
            "[ERROR] Mailbox creation failed\n"

        );

        return -1;

    }


    /* ============================================================
       CREATE HISTORY MUTEX
       ============================================================ */

    T_CMTX cmtx = {

        .mtxatr  = TA_TFIFO,
        .ceilpri = 1

    };


    mtx_history =
        tk_cre_mtx(&cmtx);


    if (mtx_history < 0)

    {

        tm_putstring(

            (UB*)
            "[ERROR] History mutex creation failed\n"

        );

        return -1;

    }


    /* ============================================================
       CREATE MESSAGE-POOL SEMAPHORE

       Initially all 8 message slots are available.
       ============================================================ */

    T_CSEM csem = {

        .sematr  = TA_TFIFO,
        .isemcnt = MSG_POOL_SIZE,
        .maxsem  = MSG_POOL_SIZE

    };


    sem_msgpool =
        tk_cre_sem(&csem);


    if (sem_msgpool < 0)

    {

        tm_putstring(

            (UB*)
            "[ERROR] Message semaphore creation failed\n"

        );

        return -1;

    }


    /* ============================================================
       CREATE TASKS
       ============================================================ */

    tskid_sensor =
        tk_cre_tsk(&ctsk_sensor);


    tskid_process =
        tk_cre_tsk(&ctsk_process);


    tskid_comm =
        tk_cre_tsk(&ctsk_comm);


    tskid_power =
        tk_cre_tsk(&ctsk_power);


    if (tskid_sensor < 0 ||
        tskid_process < 0 ||
        tskid_comm < 0 ||
        tskid_power < 0)

    {

        tm_putstring(

            (UB*)
            "[ERROR] Task creation failed\n"

        );

        return -1;

    }


    /* ============================================================
       START TASKS
       ============================================================ */

    tk_sta_tsk(
        tskid_sensor,
        0
    );


    tk_sta_tsk(
        tskid_process,
        0
    );


    tk_sta_tsk(
        tskid_comm,
        0
    );


    tk_sta_tsk(
        tskid_power,
        0
    );


    tm_putstring(

        (UB*)
        "[SYSTEM] All 4 RTOS tasks running\n"

    );


    tm_putstring(

        (UB*)
        "[SYSTEM] Sensor channels: A0 / A1 / A5\n"

    );


    tm_putstring(

        (UB*)
        "[SYSTEM] TinyML anomaly detection enabled\n"

    );


    tm_putstring(

        (UB*)
        "[SYSTEM] Slope trend detection enabled\n"

    );


    tm_putstring(

        (UB*)
        "[SYSTEM] Adaptive scheduling enabled\n"

    );


    tm_putstring(

        (UB*)
        "----------------------------------------\n"

    );


    /*
       Keep usermain alive indefinitely.
    */

    tk_slp_tsk(
        TMO_FEVR
    );


    return 0;

}
