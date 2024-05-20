################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/COM/Com.c \
../Core/Src/COM/Com_Cfg.c \
../Core/Src/COM/Com_PBCfg.c \
../Core/Src/COM/Std_Types.c 

OBJS += \
./Core/Src/COM/Com.o \
./Core/Src/COM/Com_Cfg.o \
./Core/Src/COM/Com_PBCfg.o \
./Core/Src/COM/Std_Types.o 

C_DEPS += \
./Core/Src/COM/Com.d \
./Core/Src/COM/Com_Cfg.d \
./Core/Src/COM/Com_PBCfg.d \
./Core/Src/COM/Std_Types.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/COM/%.o Core/Src/COM/%.su Core/Src/COM/%.cyclo: ../Core/Src/COM/%.c Core/Src/COM/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F303xE -c -I../Core/Inc -I../Drivers/STM32F3xx_HAL_Driver/Inc -I../Drivers/STM32F3xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F3xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-COM

clean-Core-2f-Src-2f-COM:
	-$(RM) ./Core/Src/COM/Com.cyclo ./Core/Src/COM/Com.d ./Core/Src/COM/Com.o ./Core/Src/COM/Com.su ./Core/Src/COM/Com_Cfg.cyclo ./Core/Src/COM/Com_Cfg.d ./Core/Src/COM/Com_Cfg.o ./Core/Src/COM/Com_Cfg.su ./Core/Src/COM/Com_PBCfg.cyclo ./Core/Src/COM/Com_PBCfg.d ./Core/Src/COM/Com_PBCfg.o ./Core/Src/COM/Com_PBCfg.su ./Core/Src/COM/Std_Types.cyclo ./Core/Src/COM/Std_Types.d ./Core/Src/COM/Std_Types.o ./Core/Src/COM/Std_Types.su

.PHONY: clean-Core-2f-Src-2f-COM

