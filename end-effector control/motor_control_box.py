import numpy as np

def motor_control(command_array):


    motor_min=370
    motor_max=450
    motor_array=np.int_(np.round(command_array))
    motor_array=np.clip(motor_array,motor_min,motor_max)


# default motor_num is 6
    motor1=motor_array[0]
    temp = bin(motor1)[2:].zfill(16)
    motor1SH = int(temp[0:8], 2)
    motor1SL = int(temp[8:16], 2)

    motor2=motor_array[1]
    temp = bin(motor2)[2:].zfill(16)
    motor2SH = int(temp[0:8], 2)
    motor2SL = int(temp[8:16], 2)

    motor3=motor_array[2]
    temp = bin(motor3)[2:].zfill(16)
    motor3SH = int(temp[0:8], 2)
    motor3SL = int(temp[8:16], 2)

    motor4=motor_array[3]
    temp = bin(motor4)[2:].zfill(16)
    motor4SH = int(temp[0:8], 2)
    motor4SL = int(temp[8:16], 2)

    motor5=motor_array[4]
    temp = bin(motor5)[2:].zfill(16)
    motor5SH = int(temp[0:8], 2)
    motor5SL = int(temp[8:16], 2)

    motor6=motor_array[5]
    temp = bin(motor6)[2:].zfill(16)
    motor6SH = int(temp[0:8], 2)
    motor6SL = int(temp[8:16], 2)


    DATA = np.array([106, 0, 0, 0, 0, 0, 0,
                     motor1SL, motor1SH, motor2SL, motor2SH, motor3SL, motor3SH,
                     motor4SL, motor4SH, motor5SL, motor5SH, motor6SL, motor6SH], 
                     dtype=np.uint8)
    return DATA









