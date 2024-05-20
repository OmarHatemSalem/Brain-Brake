#include "Com_PbCfg.h"

const uint8 Com_SignalInitValue_SpeedSignal[4] = {0x0};
const uint8 Com_SignalInitValue_DistSignal[4] = {0x0};
static uint8 ReadingsSignal_IPduBuffer[8];

/* I-PDU signal lists. */
static ComSignal_type ComSignal[] = {

    { /* Speed Signal */
        .ComHandleId                = COM_SIGNAL_ID_SPEEDSIGNAL,
        .ComIPduHandleId            = COM_PDU_ID_READINGSPDU,
        .ComFirstTimeoutFactor      = 0,
        .ComNotification            = COM_NO_FUNCTION_CALLOUT,
        .ComTimeoutFactor           = 0,
        .ComTimeoutNotification     = COM_NO_FUNCTION_CALLOUT,
        .ComErrorNotification       = COM_NO_FUNCTION_CALLOUT,
        .ComTransferProperty        = TRIGGERED,
        .ComUpdateBitPosition       = 0,
        .ComSignalUseUpdateBit      = FALSE,
        .ComSignalInitValue         = Com_SignalInitValue_SpeedSignal,
        .ComBitPosition             = 0,
        .ComBitSize                 = 32,
        .ComSignalEndianess         = COM_LITTLE_ENDIAN,
        .ComSignalType              = FLOAT32,
        .ComRxDataTimeoutAction     = NONE,
        .Com_EOL                    = FALSE
    },

    { /* Dist Signal */
        .ComHandleId                = COM_SIGNAL_ID_DISTSIGNAL,
        .ComIPduHandleId            = COM_PDU_ID_READINGSPDU,
        .ComFirstTimeoutFactor      = 0,
        .ComNotification            = 0,
        .ComTimeoutFactor           = 0,
        .ComTimeoutNotification     = COM_NO_FUNCTION_CALLOUT,
        .ComErrorNotification       = COM_NO_FUNCTION_CALLOUT,
        .ComTransferProperty        = TRIGGERED,
        .ComUpdateBitPosition       = 0,
        .ComSignalUseUpdateBit      = FALSE,
        .ComSignalInitValue         = Com_SignalInitValue_DistSignal,
        .ComBitPosition             = 32,
        .ComBitSize                 = 32,
        .ComSignalEndianess         = COM_LITTLE_ENDIAN,
        .ComSignalType              = FLOAT32,
        .ComRxDataTimeoutAction     = NONE,
        .Com_EOL                    = FALSE
    },

    {
        .Com_EOL                = TRUE
    }
};

/* Signal References */
static const ComSignal_type * const Readings_SignalRef[]={
        &ComSignal[COM_SIGNAL_ID_SPEEDSIGNAL],
        &ComSignal[COM_SIGNAL_ID_DISTSIGNAL],
        NULL
};
    
/* I-PDU definitions */
static ComIPdu_type ComIPdu[] = {

    {
        .ComIPduCallout =  NULL,
        .IPduOutgoingId =  COM_PDU_ID_READINGSPDU,
        .ComIPduSignalProcessing =  IMMEDIATE,
        .ComIPduSize =  8,
        .ComIPduDirection =  RECEIVE,
        .ComTxIPdu ={
            .ComTxIPduMinimumDelayFactor =  1,
            .ComTxIPduUnusedAreasDefault =  0x55,
            .ComTxModeTrue ={
                .ComTxModeMode =   DIRECT,
                .ComTxModeNumberOfRepetitions =   0,
            },
        },
        .ComIPduDataPtr =  ReadingsSignal_IPduBuffer,
        .ComIPduDeferredDataPtr =  NULL,
        .ComIPduSignalRef =  Readings_SignalRef,
        .Com_EOL =  FALSE,
    },

    {
        .Com_EOL =  TRUE
    }
};

const Com_ConfigType ComConfiguration = {
    .ComIPdu =  ComIPdu,
    .ComSignal =  ComSignal,
    .ComNumOfSignals = 2,
    .ComNumOfIPDUs = 1
};
