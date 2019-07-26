#ifndef DSLASH_TABLE_H
#define DSLASH_TABLE_H

#include "neon_dslash_types.h"
#include "qmp.h"

namespace Chroma
{

class DslashTable {
public:
    DslashTable(int subgrid[]);
    ~DslashTable();

    // Accessors
    HalfSpinor* getChi1() {
        return chi1;
    }

    HalfSpinor* getChi2() {
        return chi2;
    }

    HalfSpinor*** getRecvBufptr() {
        return (HalfSpinor***)recv_bufptr;
    }

    HalfSpinor*** getSendBufptr() {
        return (HalfSpinor***)send_bufptr;
    }
    
    // Communications
    //
    inline
    void startReceives() {
        /* Prepost all receives */
        if (total_comm > 0) {

            // Use QMP Harness 
            if (QMP_start(recv_all_mh[0]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_start failed in forward direction");
                QMP_abort(1);
            }
	
            if (QMP_start(recv_all_mh[1]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_start failed in backward direction");
                QMP_abort(1);
            }
        }
    }

    inline void finishReceiveFromForward() 
    {  
        /* Finish all forward receives */
        if (total_comm > 0 ) { 
            if (QMP_wait(recv_all_mh[1]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_wait failed in forward direction");
                QMP_abort(1);
            }
        }
    }

    inline void finishReceiveFromBack() 
    { 
        if( total_comm > 0 ) { 
            /* Finish all forward receives */
            if (QMP_wait(recv_all_mh[0]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_wait failed in forward direction");
                QMP_abort(1);
            }
        }
    }
	

    inline void startSendBack() 
    { 
        if(total_comm > 0) {
            if (QMP_start(send_all_mh[1]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_start failed in forward direction");
                QMP_abort(1);
            }
        }
    }
    
    inline void startSendForward() 
    {
        if(total_comm > 0) {
            if (QMP_start(send_all_mh[0]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_start failed in forward direction");
                QMP_abort(1);
            }
        }
    }

    inline void finishSendBack() {
        if( total_comm > 0 ) {
            /* Finish all sends */
            if (QMP_wait(send_all_mh[1]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_wait failed in forward direction");
                QMP_abort(1);
            }
        }
    }

    inline void finishSendForward() {
        if( total_comm > 0 ) {
            /* Finish all sends */
            if (QMP_wait(send_all_mh[0]) != QMP_SUCCESS) {
                QMP_error("sse_su3dslash_wilson: QMP_wait failed in forward direction");
                QMP_abort(1);
            }
        }
    }
private:

    static QMP_mem_t* xchi;
    
    HalfSpinor *chi1;
    HalfSpinor *chi2;

    HalfSpinor* recv_bufptr[2][4];
    HalfSpinor* send_bufptr[2][4];

    HalfSpinor* send_bufs;
    HalfSpinor* recv_bufs;

    QMP_msgmem_t send_msg[2][4];
    QMP_msgmem_t recv_msg[2][4];

    QMP_msghandle_t send_mh[2][4];
    QMP_msghandle_t recv_mh[2][4];

    QMP_msghandle_t send_all_mh[4];
    QMP_msghandle_t recv_all_mh[4];

    int total_comm;   
};


} // namespace Chroma
#endif // DSLASH_TABLE_H
