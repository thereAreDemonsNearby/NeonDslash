#ifndef SHIFT_TABLE_H
#define SHIFT_TABLE_H

#include "neon_dslash_types.h"
#include <memory>
#include "qmp.h"

namespace Chroma
{

enum HalfSpinorOffsetType {
    DECOMP_SCATTER=0,
    DECOMP_HVV_SCATTER,
    RECONS_MVV_GATHER,
    RECONS_GATHER
};

struct InvTab { 
    int cb;
    int linearcb;
};

class ShiftTable {
public:
    
    ShiftTable(
        const int* _subgrid_size,
        HalfSpinor* chi1, 
        HalfSpinor* chi2,
        HalfSpinor* recv_bufs[2][4],
        HalfSpinor* send_bufs[2][4],
        void (*getSiteCoords)(int coord[], int node, int linearsite), 
        int (*getLinearSiteIndex)(const int coord[]),
        int (*getNodeNumber)(const int coord[])
        );

    ~ShiftTable() {
        free(xoffset_table);
        free(xsite_table);
    }

    inline
    int siteTable(int i) {
        return site_table[i];
    }



    HalfSpinor* halfspinorBufferOffset(HalfSpinorOffsetType type, int site, int mu) {
        //      std::cout << "type="<<type<<" site="<<site<<" mu="<<mu<<" index=" << (mu + 4*( site + subgrid_vol*(int)type)) << std::endl << std::flush;
	return offset_table[mu + 4*( site + subgrid_vol*(int)type) ];
    }

    inline int subgridVolCB() {
        return subgrid_vol_cb;
    }
private:
    /* Tables */
    HalfSpinor** xoffset_table;        /* Unaligned */
    HalfSpinor** offset_table;         /* Aligned */
    
    int *xsite_table;         /* Unaligned */
    int *site_table;          /* Aligned */
        
    int tot_size[4];          /* Class scope members */
    int subgrid_size[4];
    int subgrid_cb_size[4];
    int subgrid_vol;
    int subgrid_vol_cb;         /* Useful numbers */
    const int Nd;                   /* No of Dimensions */
    

    // This is not needed as it can be done transitively:
    // ie lookup the QDP index and then lookup the coord with that 
    inline
    void mySiteCoords4D(int gcoords[], int node, int linearsite)
    {
        int mu;
        int tmp_coord[4];
        int cb,cbb;
        int* log_coords=QMP_get_logical_coordinates_from(node);
        int my_node = QMP_get_node_number();


        for(mu=0; mu < 4; mu++) { 
            gcoords[mu] = log_coords[mu]*subgrid_size[mu];
        }
        free(log_coords);
 
        cb=linearsite/subgrid_vol_cb;
      
        crtesn4d(linearsite % subgrid_vol_cb, subgrid_cb_size, tmp_coord);

        // Add on position within the node
        // NOTE: the cb for the x-coord is not yet determined
        gcoords[0] += 2*tmp_coord[0];
        for(mu=1; mu < 4; ++mu) {
            gcoords[mu] += tmp_coord[mu];
        }
      
        cbb = cb;
        for(mu=1; mu < 4; ++mu) {
            cbb += gcoords[mu];
        }
        gcoords[0] += (cbb & 1);
    }

 
    inline
    int myLinearSiteIndex4D(const int gcoords[]) 
    {
        int mu;
        int subgrid_cb_coord[4];
        int cb;


        cb=0;
        for(mu=0; mu < Nd; ++mu) { 
            cb += gcoords[mu];
        }
        cb &=1;
    
        subgrid_cb_coord[0] = (gcoords[0]/2)% subgrid_cb_size[0];
        for(mu=1; mu < 4; mu++) { 
            subgrid_cb_coord[mu] = gcoords[mu] % subgrid_cb_size[mu];
        }

        return localSite4d(subgrid_cb_coord, subgrid_cb_size) + cb*subgrid_vol_cb;
    }


    
    
    inline
    int localSite4d(int coord[], int latt_size[])
    {
        int order = 0;
        int mmu;
      
        for(mmu=4-1; mmu >= 1; --mmu) {
            order = latt_size[mmu-1]*(coord[mmu] + order);
        }
        order += coord[0];
      
        return order;
    }

    

    inline 
    void crtesn4d(int ipos, const int latt_size[], int coord[] )
    {
        int Ndim=0; /* Start running x fastest */
        int i, ix;
      
        /* Calculate the Cartesian coordinates of the VALUE of IPOS where the 
         * value is defined by
         *
         *     for i = 0 to NDIM-1  {
         *        X_i  <- mod( IPOS, L(i) )
         *        IPOS <- int( IPOS / L(i) )
         *     }
         *
         * NOTE: here the coord(i) and IPOS have their origin at 0. 
         */
        for(i = Ndim; i < Ndim+4; ++i) {
            ix=i%4;  /* This lets me start with the time direction and then wraparound */
	
            coord[ix] = ipos % latt_size[ix];
            ipos = ipos / latt_size[ix];
        }
    }

    inline 
    void offs(int temp[], const int coord[], int mu, int isign)
    {
        int i;
      
        for(i=0; i < 4; ++i) {
            temp[i] = coord[i];
        }
      
        /* translate address to neighbour */
        temp[mu] = (temp[mu] + isign + 2*tot_size[mu]) % tot_size[mu];
    } 

    inline
    int parity(const int coord[])
    {
        int m;
        int sum = 0;
      
        for(m=0; m < 4; ++m) {
            sum += coord[m];
        }
      
        return sum % 2;
    }
};
  

} // namespace Chroma

#endif
