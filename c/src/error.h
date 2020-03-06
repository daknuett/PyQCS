#ifndef GQCS_error_h_
#define GQCS_error_h_

volatile char * GQCS_err_msg;

#define GQCS_set_error(emsg) (GQCS_err_msg = emsg)

#endif
