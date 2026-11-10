/* Stage 9: giant-planet migration and e/i damping as a REBOUND additional_forces function.
 * The coefficients are set from Python (trojans/pirani_forces.py) at every runner check:
 *   pf_mig[k]  = adot/(2a) of giant k            (a_m  = pf_mig v)
 *   pf_e[k]    = 1/t_e                           (a_e  = -2 (v.r) r / r^2 * pf_e)
 *   pf_i[k]    = 2/t_i                           (a_i  = -pf_i v_z z)
 * Heliocentric r, v; giants are particles 1..pf_n. Only the giants are touched. */
#include "rebound.h"

#define PF_NMAX 8
double pf_mig[PF_NMAX];
double pf_e[PF_NMAX];
double pf_i[PF_NMAX];
int pf_n = 0;

void pirani_forces(struct reb_simulation* r) {
    struct reb_particle* const ps = r->particles;
    const struct reb_particle sun = ps[0];
    for (int k = 0; k < pf_n; k++) {
        struct reb_particle* p = &ps[k + 1];
        const double rx = p->x - sun.x, ry = p->y - sun.y, rz = p->z - sun.z;
        const double vx = p->vx - sun.vx, vy = p->vy - sun.vy, vz = p->vz - sun.vz;
        const double r2 = rx * rx + ry * ry + rz * rz;
        const double g = -2.0 * (rx * vx + ry * vy + rz * vz) / r2 * pf_e[k];
        p->ax += pf_mig[k] * vx + g * rx;
        p->ay += pf_mig[k] * vy + g * ry;
        p->az += pf_mig[k] * vz + g * rz - pf_i[k] * vz;
    }
}
