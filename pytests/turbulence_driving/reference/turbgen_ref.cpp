// Reference driver for the astronomix port of TurbGen.
//
// Runs one parameter set through the unmodified algorithms of TurbGen.h (commit d9a7d74 of
// github.com/chfeder/turbulence_generator; the header is copied with the two-line instrumentation recorded in
// TurbGen_instrumented.patch, which logs every Gaussian deviate) and prints, with 17 significant digits,
// the mode list, the deviates, the Ornstein-Uhlenbeck state, the projected coefficients and the acceleration
// at the cell centres x_i = (i + 1/2) L / n of a small grid.
//
// usage: turbgen_ref key=value ...   keys of TurbGen::init_driving(map) are passed through verbatim
//        (ndim, length, target_vdisp, ampl_factor, ampl_auto_adjust, k_driv, k_min, k_max, sol_weight,
//        spect_form, power_law_exp, angles_exp, random_seed, nsteps_per_t_turb); driver keys:
//        out=file   grid=nx,ny,nz   times=t0,t1,... (in units of the update interval)   vturb=vx,vy,vz (optional)
// build: see build.sh (explicit g++ command, outputs outside the TurbGen source directory)

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "TurbGen_instrumented.h"

class TurbGenProbe : public TurbGen {
  public:
    TurbGenProbe() : TurbGen() {
        // members that a 1D or 2D initialisation leaves unset (TurbGen.h reads only ncmp entries)
        for (int d = 0; d < 3; d++) { L[d] = 1.0; ampl_factor[d] = 1.0; }
        power_law_exp = 0.0; angles_exp = 0.0;
    }
    FILE * out = stdout; // TurbGen itself prints to stdout, so the reference data go to their own file
    void print(const char * name, const std::vector<double> & values) {
        fprintf(out, "%s %zu\n", name, values.size());
        for (size_t i = 0; i < values.size(); i++) fprintf(out, "%.17g\n", values[i]);
    }
    void print_setup() {
        print("scalars", {(double)nmodes, (double)ncmp, t_decay, dt, OUvar, sol_weight_norm, kmin, kmax});
        std::vector<double> modes;
        for (int m = 0; m < nmodes; m++) for (int d = 0; d < (int)ndim; d++) modes.push_back(mode[d][m]);
        print("modes", modes);
        print("amplitudes", ampl);
    }
    void print_state(const double time, const bool updated) {
        print("time", {time, (double)updated, (double)step});
        print("deviates", deviate_log); deviate_log.clear();
        print("ou_phases", OUphases);
        std::vector<double> coefficients; // (mode, component, [aka, akb])
        for (int m = 0; m < nmodes; m++) for (int d = 0; d < ncmp; d++) {
            coefficients.push_back(aka[d][m]); coefficients.push_back(akb[d][m]);
        }
        print("coefficients", coefficients);
        print("ampl_factor", std::vector<double>(ampl_factor, ampl_factor + ncmp));
    }
    void print_field(const int n[3]) {
        // double-precision field from get_turb_vector, x outermost (the layout of the astronomix state)
        std::vector<double> field;
        double del[3], pos[3] = {0.0, 0.0, 0.0}, v[3];
        for (int d = 0; d < 3; d++) del[d] = L[d] / n[d];
        for (int i = 0; i < n[0]; i++) for (int j = 0; j < n[1]; j++) for (int k = 0; k < n[2]; k++) {
            pos[0] = (i + 0.5) * del[0]; pos[1] = (j + 0.5) * del[1]; pos[2] = (k + 0.5) * del[2];
            get_turb_vector(pos, v);
            for (int d = 0; d < ncmp; d++) field.push_back(v[d]);
        }
        print("field", field);
        // cross-check against the single-precision uniform-grid routine (x innermost there)
        long ncells = (long)n[0] * n[1] * n[2];
        std::vector<float> grid[3]; float * grid_ptr[3];
        for (int d = 0; d < 3; d++) { grid[d].resize(ncells); grid_ptr[d] = grid[d].data(); }
        double pos_beg[3], pos_end[3];
        for (int d = 0; d < 3; d++) { pos_beg[d] = 0.5 * del[d]; pos_end[d] = L[d] - 0.5 * del[d]; }
        get_turb_vector_unigrid(pos_beg, pos_end, n, grid_ptr);
        double max_diff = 0.0, max_abs = 0.0;
        for (int i = 0; i < n[0]; i++) for (int j = 0; j < n[1]; j++) for (int k = 0; k < n[2]; k++)
            for (int d = 0; d < ncmp; d++) {
                double a = field[((i * n[1] + j) * n[2] + k) * ncmp + d];
                double b = grid[d][(long)k * n[0] * n[1] + (long)j * n[0] + i];
                max_diff = std::max(max_diff, fabs(a - b)); max_abs = std::max(max_abs, fabs(a));
            }
        print("unigrid_check", {max_diff, max_abs});
    }
};

static std::vector<double> parse_list(const std::string & text) {
    std::vector<double> values; std::stringstream stream(text); std::string token;
    while (std::getline(stream, token, ',')) values.push_back(atof(token.c_str()));
    return values;
}

int main(int argc, char ** argv) {
    std::map<std::string, std::string> params;
    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a]; size_t eq = arg.find('=');
        params[arg.substr(0, eq)] = arg.substr(eq + 1);
    }
    std::vector<double> grid = parse_list(params["grid"]), times = parse_list(params["times"]);
    std::vector<double> vturb = params.count("vturb") ? parse_list(params["vturb"]) : std::vector<double>();
    const int n[3] = {(int)grid[0], (int)grid[1], (int)grid[2]};
    TurbGenProbe tg; tg.set_verbose(0);
    tg.out = fopen(params["out"].c_str(), "w");
    tg.init_driving(params, 0.0);
    tg.print_setup();
    tg.print_state(-1.0, false); // state after initialisation (before the first pattern)
    const double interval = tg.get_turnover_time() / tg.get_nsteps_per_turnover_time();
    for (size_t i = 0; i < times.size(); i++) {
        const double time = times[i] * interval;
        bool updated = vturb.empty() ? tg.check_for_update(time) : tg.check_for_update(time, vturb.data());
        tg.print_state(time, updated);
        tg.print_field(n);
    }
    fclose(tg.out);
    return 0;
}
