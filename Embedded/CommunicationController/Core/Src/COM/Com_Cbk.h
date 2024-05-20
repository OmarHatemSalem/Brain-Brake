#ifndef COM_CBK_H
#define COM_CBK_H

/* Callback functions and Notifications */
BufReq_ReturnType Com_StartOfReception(PduIdType id, const PduInfoType* info,PduLengthType TpSduLength, PduLengthType* bufferSizePtr); /*SID 0x46*/
BufReq_ReturnType Com_CopyRxData(PduIdType id, const PduInfoType* info, PduLengthType* bufferSizePtr); /*SID 0x44*/
BufReq_ReturnType Com_CopyTxData(PduIdType id, const PduInfoType* info, RetryInfoType* retry, PduLengthType* availableDataPtr); /*SID 0x43*/

#endif
