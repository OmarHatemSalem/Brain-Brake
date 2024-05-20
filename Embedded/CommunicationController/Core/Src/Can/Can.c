#include "Can.h"
#include "../COM/Com_Cbk.h"
#include "stm32f3xx_hal.h"

#include <stdlib.h>
#include <string.h>


#define Can_GetCtrlPvtData(cont) (&CanUnit[cont])

uint32  MsgCode;              /* Received message buffer code */
uint32  MsgId;                /* Received message ID */
uint32  MsgLen;            /* Recieved message number of data bytes */
uint8   MsgData[8];           /* Received message data string*/
uint32  MsgTs;         /* Received message time */


//#include "stm32f3xx_hal_can.h"

CAN_HandleTypeDef hcan;



typedef enum {
    CAN_UNINIT = 0,
    CAN_READY
} Can_DriverStateType;

Can_DriverStateType Can_state = CAN_UNINIT;

Can_ReturnType Can_Write(Can_HwHandleType Hth, const Can_PduType* PduInfo) {
	// we only have one controller with id 0
	if (Hth != 0) {
		return CAN_NOT_OK;
	}
	CAN_TxHeaderTypeDef TxHeader;
	uint32_t TxMailbox;
	TxHeader.DLC = PduInfo->length;
	TxHeader.IDE = CAN_ID_STD;
	TxHeader.RTR = CAN_RTR_DATA;
	TxHeader.StdId = PduInfo->id;
	return HAL_CAN_AddTxMessage(&hcan, &TxHeader, PduInfo->sdu, &TxMailbox) == HAL_OK ? CAN_OK : CAN_NOT_OK;
}


void Can_Init(const Can_ConfigType *Config) {
    /* Do initial configuration of layer here */
	hcan.Instance = CAN;
	hcan.Init.Prescaler = 12;
	hcan.Init.Mode = CAN_MODE_NORMAL;
	hcan.Init.SyncJumpWidth = CAN_SJW_1TQ;
	hcan.Init.TimeSeg1 = CAN_BS1_2TQ;
	hcan.Init.TimeSeg2 = CAN_BS2_2TQ;
	hcan.Init.TimeTriggeredMode = DISABLE;
	hcan.Init.AutoBusOff = DISABLE;
	hcan.Init.AutoWakeUp = DISABLE;
	hcan.Init.AutoRetransmission = DISABLE;
	hcan.Init.ReceiveFifoLocked = DISABLE;
	hcan.Init.TransmitFifoPriority = DISABLE;

	HAL_CAN_Start(&hcan);

	Can_state = CAN_READY;

}



