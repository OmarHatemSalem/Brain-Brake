#ifndef COM_H
#define COM_H

#include "Com_Cfg.h"
#include "ComStack_Types.h"


#define COM_SW_MAJOR_VERSION   1
#define COM_SW_MINOR_VERSION   2
#define COM_SW_PATCH_VERSION   0
#define NO_PDU_REFERENCE 	   -1
#define COM_NO_FUNCTION_CALLOUT NULL


#define Com_SignalTypeToSize(type, len) \
	(type == UINT8   ? sizeof(uint8) : \
	type == UINT16  ? sizeof(uint16) : \
	type == UINT32  ? sizeof(uint32) : \
	type == UINT8_N  ? sizeof(uint8) * len : \
	type == SINT8   ? sizeof(sint8) : \
	type == SINT16  ? sizeof(sint16) : \
	type == SINT32 ? sizeof(sint32) : sizeof(boolean))

typedef uint16 Com_BitPositionType;

/* @req COM346 */
typedef boolean (*ComRxIPduCallout_type)(PduIdType PduId, const uint8 *IPduData);

/* @req COM700 */
typedef boolean (*ComTxIPduCallout_type)(PduIdType PduId, const uint8 *IPduData);

/* @req COM491 COM554 COM555 COM556*/
typedef void(*ComNotificationCallout_type)(void);

extern const ComRxIPduCallout_type ComRxIPduCallouts[];
extern const ComTxIPduCallout_type ComTxIPduCallouts[];
extern const ComNotificationCallout_type ComNotificationCallouts[];

typedef enum Com_StatusType {
	COM_UNINIT,
	COM_INIT
} Com_StatusType;

/* @req COM119 */
typedef enum Com_IPduSignalProcessing {
	IMMEDIATE,
	DEFERRED
} Com_IPduSignalProcessing;

/* @req COM493 */
typedef enum Com_IPduDirection {
	RECEIVE,
	SEND
} Com_IPduDirection;

/* @req COM232 */
typedef enum ComTransferProperty_type {
	PENDING, /* Transmission not triggered */
	TRIGGERED, /* Can be triggered by I-PDU */
	TRIGGERED_ON_CHANGE, /* Triggered if value is different than stored */
	TRIGGERED_ON_CHANGE_WITHOUT_REPETITION, /* Triggered once if value is different than stored */
	TRIGGERED_WITHOUT_REPETITION /* Can be triggered by I-PDU but once */
} ComTransferProperty_type;

/* @req COM137 */
typedef enum ComTxModeMode_type {
	DIRECT,
	MIXED,
	NONE,
	PERIODIC
} ComTxModeMode_type;

/* @req COM602 */
/* Filtering mechanisms used on sender side for Transmission Mode Conditions (TMC) but shall not filter out signals on sender side.*/
typedef enum ComFilterAlgorithm_type {
	ALWAYS,
	MASKED_NEW_DIFFERS_MASKED_OLD,
	MASKED_NEW_DIFFERS_X,
	MASKED_NEW_EQUALS_X,
	NEVER,
	NEW_IS_OUTSIDE,
	NEW_IS_WITHIN,
	ONE_EVERY_N
} ComFilterAlgorithm_type;

/* @req COM157 */
typedef enum ComSignalEndianess_type {
	COM_BIG_ENDIAN,
	COM_LITTLE_ENDIAN,
	COM_OPAQUE
} ComSignalEndianess_type;

/* @req COM291 */
/* reception deadline monitoring of the I-PDU using the smallest configured non zero timeout parameter */
typedef enum ComRxDataTimeoutAction_type {
	NONEX,
	REPLACE
} ComRxDataTimeoutAction_type;


/* @req COM127 */
/* Used to determine signedness of the signal and may be used to reserve storage */
typedef enum Com_SignalType {
	BOOLEAN,
	FLOAT32,
	FLOAT64,
	SINT16,
	SINT32,
	SINT64,
	SINT8,
	UINT16,
	UINT32,
	UINT64,
	UINT8,
	UINT8_DYN,
	UINT8_N
} Com_SignalType;

typedef uint16 Com_SignalIdType;
typedef uint16 Com_SignalGroupIdType;
typedef uint16 Com_IpduGroupIdType;
/* typedef uint8:1 Com_IpduGroupVector; Check*/
typedef uint8 Com_ServiceIdType;


/* #define BOOLEAN bool
#define UINT8 uint8
#define UINT16 uint16
#define UINT32 uint32
#define UINT8_N uint8
#define SINT8 sint8
#define SINT16 sint16
#define SINT32 sint32 */

#define COMServiceId_Init 0x01 
#define COMServiceId_DeInit 0x02 
#define COMServiceId_IpduGroupControl 0x03 
#define COMServiceId_ReceptionDMControl 0x06 
#define COMServiceId_GetStatus 0x07 
#define COMServiceId_GetVersionInfo 0x09 
#define COMServiceId_SendSignal 0x0A 
#define COMServiceId_ReceiveSignal 0x0B 
#define COMServiceId_UpdateShadowSignal 0x0C 
#define COMServiceId_SendSignalGroup 0x0D 
#define COMServiceId_ReceiveSignalGroup 0x0E 
#define COMServiceId_ReceiveShadowSignal 0x0F 
#define COMServiceId_InvalidateSignal 0x10 
#define COMServiceId_InvalidateShadowSignal 0x16 
#define COMServiceId_TriggerIPDUSend 0x17 
#define COMServiceId_MainFunctionRx 0x18 
#define COMServiceId_MainFunctionTx 0x19 
#define COMServiceId_MainFunctionRouteSignals 0x1A 
#define COMServiceId_InvalidateSignalGroup 0x1B
#define COMServiceId_ClearIpduGroupVector 0x1C 
#define COMServiceId_SetIpduGroup 0x1D 
#define COMServiceId_SendDynSignal 0x21 
#define COMServiceId_ReceiveDynSignal 0x22 
#define COMServiceId_SendSignalGroupArray 0x23 
#define COMServiceId_ReceiveSignalGroupArray 0x24 
#define COMServiceId_SwitchIpduTxMode 0x27 
#define COMServiceId_TriggerIPDUSendWithMetaData 0x28 
#define COMServiceId_TxConfirmation 0x40 
#define COMServiceId_TriggerTransmit 0x41 
#define COMServiceId_RxIndication 0x42 
#define COMServiceId_CopyTxData 0x43 
#define COMServiceId_CopyRxData 0x44 
#define COMServiceId_TpRxIndication 0x45 
#define COMServiceId_StartOfReception 0x46 
#define COMServiceId_TpTxConfirmation 0x48


typedef struct Com_BufferStateType {
	PduLengthType index;
	boolean isLocked;
} Com_BufferStateType;


/* @req COM351 */
/* Configuration container for Tx-mode for I-PDUs. */
typedef struct ComTxMode_type {
	ComTxModeMode_type ComTxModeMode; /* Transmission mode for this IPdu. */
	uint8 ComTxModeNumberOfRepetitions; /* @req COM281 */ /* Number of times the I-PDU will be sent in each I-PDU cycle. Set to 0 for DIRECT transmission mode and >0 for DIRECT/N-times mode. */
	uint32 ComTxModeRepetitionPeriod; /* @req COM282 */ /* Defines the period of the transmissions in DIRECT/N-times and MIXED transmission modes. */
	uint32 ComTxModeTimeOffset; /* @req COM180 */ /* Time before first transmission of this IPDU. (i.e. between the ipdu group start and this IPDU is sent for the first time. */
	uint32 ComTxModeTimePeriod; /* @req COM178 */ /* Period of cyclic transmission for PERIODIC or MIXED. */
} ComTxMode_type;

/* @req COM496 */
/* Configuartion container related to transmission parameters  */
typedef struct ComTxIPdu_type {
	uint32 ComTxIPduMinimumDelayFactor; /* Minimum delay between successive transmissions of the I-PDU. */
	uint8 ComTxIPduUnusedAreasDefault; /* COM will fill unused areas within an IPdu with this bit patter. */
	ComTxMode_type ComTxModeTrue; /* Transmission modes for the I-PDU.*/
	ComTxMode_type ComTxModeFalse;
} ComTxIPdu_type;


/** Configuration structure for signals and signal groups. */
typedef struct ComSignal_type {
    Com_BitPositionType ComBitPosition; /* @req COM259 */ /* Start bit/position of signal within I-PDU*/
    union {
        uint8 ComBitSize; /* @req COM158 */ /* Size of signal in bits */
        uint32 ComSignalLength; /* @req COM437 */ /* Size of signal in bytes for UINT8_N and UINT8_DYN */
    };
    uint32 ComErrorNotification; /** Notification function for error notification. */
    uint32 ComFirstTimeoutFactor; /* First timeout period for deadline monitoring. */
    uint16 ComHandleId; /* @req COM165 */ /* Identifier for the signal. */
    uint32 ComNotification; /* Tx and Rx notification function. */
    ComRxDataTimeoutAction_type ComRxDataTimeoutAction; /* @req COM412 */ /* Action to be performed when a reception timeout occurs. */
    ComSignalEndianess_type ComSignalEndianess; /* Endianess of the signal's network representation. */
    const void *ComSignalInitValue; /* Value used to initialized this signal. */
    Com_SignalType ComSignalType; /* Type of the signal. */
    uint32 ComTimeoutFactor; /* Timeout period for deadline monitoring. */
    uint32 ComTimeoutNotification; /* @req COM552 */ /* Timeout notification function. */
    ComTransferProperty_type ComTransferProperty;
    Com_BitPositionType ComUpdateBitPosition; /* @req COM257 */ /* Update bit's position, NULL if none */
    uint8 ComSignalUseUpdateBit;  /* Set when update bit is used.*/
    ComFilterAlgorithm_type ComFilter; /* Filter for this signal. */
    uint16 ComIPduHandleId; /* @req COM175 */ /*ID of the IPDU that this signal belongs to */
    uint8 Com_EOL; /* Marks the end of list for the signal configuration array. */
} ComSignal_type;

/* @req COM340 */
typedef struct ComIPdu_type {
	/** Callout function of this IPDU.
	 * The callout function is an optional function used both on sender and receiver side.
	 * If configured, it determines whether an IPdu is considered for further processing. If
	 * the callout return false the IPdu will not be received/sent.
	 */
	boolean (*ComIPduCallout)(PduIdType PduId, const uint8 *IPduData);
	uint8 IPduOutgoingId; /* Outgoing PDU ID*/
	Com_IPduSignalProcessing ComIPduSignalProcessing; /* Signal processing mode for this IPDU. */
	uint8 ComIPduSize; /* Size of the IPDU in bytes. Range 0-8 for CAN */
	Com_IPduDirection ComIPduDirection; /* The direction of the IPDU. Receive or Send. */
	ComTxIPdu_type ComTxIPdu; /* Container of transmission related parameters. */
	void *ComIPduDataPtr; /* Reference to the actual pdu data storage */
	void *ComIPduDeferredDataPtr;
	const ComSignal_type * const *ComIPduSignalRef; /* References to all signals contained in this I-PDU. */
	uint8 Com_EOL; /* Marks the end of list for this configuration array. */
} ComIPdu_type;

/* @req COM825 */
/* Top-level configuration container for COM. Exists once per configuration. */
typedef struct Com_ConfigType {
	ComIPdu_type *ComIPdu; /* IPDU definitions */
	ComSignal_type *ComSignal; /* Signal definitions */
	uint8 ComNumOfSignals; /* Number of signals */
	uint8 ComNumOfIPDUs; /* Number of I-PDUs */
} Com_ConfigType;


/* Startup and Control Services*/
void Com_Init(const Com_ConfigType * config); /*SID 0x01*/
void Com_DeInit(void);
Com_StatusType Com_GetStatus(void); /*SID 0x07, returns COM_INIT or COM_UNINIT*/
void Com_TxConfirmation(PduIdType TxPduId);
void Com_RxIndication(PduIdType RxPduId, const PduInfoType* PduInfoPtr);



/* Communication Services */
uint8 Com_SendSignal(Com_SignalIdType SignalId, const void* SignalDataPtr); /*SID 0x0a*/
uint8 Com_ReceiveSignal(Com_SignalIdType SignalId, void* SignalDataPtr); /*SID 0x0b*/

/* Helper functions */
void Com_WriteData(uint8 *pdu, uint8 *pduSignalMask, const uint8 *signalDataPtr, 
	uint8 destByteLength, Com_BitPositionType segmentStartBitOffset, uint8 segmentBitLength);
void Com_WriteToPDU(Com_SignalIdType signalId, const void *signalData, boolean *dataChanged);
boolean Com_BufferLocked(PduIdType id);
void Com_RxSignalProcess(ComIPdu_type *IPdu);
Com_BitPositionType Com_GetByteOffset(Com_BitPositionType BitNumber);

extern ComSignalEndianess_type Com_SystemEndianness;
extern Com_BufferStateType Com_BufferState[];

#define COM_BUSY 0x81
#define COM_SERVICE_NOT_AVAILABLE 0x80
#endif /*COM_H_*/
