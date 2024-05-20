#include "Com.h"
#include "UserCbk.h"

/* Notifications */
const ComNotificationCallout_type ComNotificationCallouts [] = { signal_callback, NULL };

/* RX Callouts */
const ComRxIPduCallout_type ComRxIPduCallouts[] = { NULL };

/* TX Callouts */
const ComTxIPduCallout_type ComTxIPduCallouts[] = { NULL };
