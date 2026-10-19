// Explicit device SoA, independent of Rust struct layout and padding.
typedef unsigned long long U64;
static constexpr float MAX_F = 3.4028234663852886e38f;

struct State {
    float v, activity, input, threshold, burst, adaptation, rebound, threshold_adaptation, current, structural;
    U64 updated, refractory, last_spike, spike_valid, spikes, homeostasis;
};
struct Params {
    float rest, reset, configured_threshold, tau, trace_tau, drive, burst_gain, burst_tau,
          adaptation_gain, adaptation_tau, threshold_gain, threshold_tau, rebound_gain, rebound_tau;
    U64 refractory_us;
};

__device__ float decay(float value, U64 dt, float tau) { return value * expf(-((float)dt) / tau); }

__device__ double response(float current, float current_tau_us, float tau_us, U64 dt) {
    if (current == 0.0f || dt == 0) return 0.0;
    double t = (double)dt / 1000000.0;
    double tc = (double)current_tau_us / 1000000.0;
    double tm = (double)tau_us / 1000000.0;
    if (fabs(tc - tm) / tm <= 1e-9) return (double)current * t * exp(-t / tm);
    return (double)current * tm * tc / (tc - tm) * (exp(-t / tc) - exp(-t / tm));
}

__device__ bool advance(State &s, const Params &p, U64 time) {
    if (time < s.updated) return false;
    U64 dt = time - s.updated;
    if (dt == 0) return true;
    bool active = p.drive != 0.0f || p.burst_gain != 0.0f || p.adaptation_gain != 0.0f
        || p.threshold_gain != 0.0f || p.rebound_gain != 0.0f;
    if (!active) {
        float equilibrium = p.rest + s.current * (p.tau / 1000000.0f);
        if (!isfinite(equilibrium)) return false;
        s.v = equilibrium + (s.v - equilibrium) * expf(-((float)dt) / p.tau);
    } else {
        double equilibrium = (double)p.rest + ((double)p.drive + (double)s.current) * (double)p.tau / 1000000.0;
        if (!isfinite(equilibrium) || fabs(equilibrium) > (double)MAX_F) return false;
        double t = (double)dt / 1000000.0;
        double tm = (double)p.tau / 1000000.0;
        double eq = (double)p.rest + ((double)p.drive + (double)s.current) * tm;
        double v = eq + ((double)s.v - eq) * exp(-t / tm);
        v += response(s.burst, p.burst_tau, p.tau, dt);
        v -= response(s.adaptation, p.adaptation_tau, p.tau, dt);
        v += response(s.rebound, p.rebound_tau, p.tau, dt);
        if (!isfinite(v) || fabs(v) > (double)MAX_F) return false;
        s.v = (float)v;
    }
    if (!isfinite(s.v)) return false;
    s.burst = decay(s.burst, dt, p.burst_tau);
    s.adaptation = decay(s.adaptation, dt, p.adaptation_tau);
    s.rebound = decay(s.rebound, dt, p.rebound_tau);
    s.threshold_adaptation = decay(s.threshold_adaptation, dt, p.threshold_tau);
    s.activity = decay(s.activity, dt, p.trace_tau);
    s.input = decay(s.input, dt, p.trace_tau);
    s.updated = time;
    return true;
}

__device__ bool checked_add(float a, float b, float &out) {
    double sum = (double)a + (double)b;
    if (!isfinite(sum) || fabs(sum) > (double)MAX_F) return false;
    out = (float)sum;
    return true;
}

// 0 = no spike, 1 = spike, 2 = numerical failure (host replays CPU error path).
__device__ unsigned integrate(State &s, const Params &p, U64 time, float sum, float mag, float inhibitory) {
    if (!advance(s, p, time)) return 2;
    s.input = (float)fmin((double)s.input + (double)mag, (double)MAX_F);
    if (time < s.refractory) return 0;
    float v = s.v + sum;
    if (!isfinite(v)) return 2;
    float rebound;
    if (v < s.threshold + s.threshold_adaptation) {
        if (!checked_add(s.rebound, p.rebound_gain * inhibitory, rebound)) return 2;
        s.v = v; s.rebound = rebound;
        return 0;
    }
    if (p.refractory_us > (~(U64)0) - time) return 2;
    float burst, adaptation, threshold_adaptation, effective;
    if (!checked_add(s.burst, p.burst_gain, burst)
        || !checked_add(s.adaptation, p.adaptation_gain, adaptation)
        || !checked_add(s.threshold_adaptation, p.threshold_gain, threshold_adaptation)
        || !checked_add(s.threshold, threshold_adaptation, effective)
        || !checked_add(s.rebound, p.rebound_gain * inhibitory, rebound)) return 2;
    s.v = p.reset; s.refractory = time + p.refractory_us;
    s.burst = burst; s.adaptation = adaptation; s.threshold_adaptation = threshold_adaptation; s.rebound = rebound;
    s.activity += 1.0f; s.last_spike = time; s.spike_valid = 1;
    if (s.spikes < (~(U64)0)) ++s.spikes;
    return 1;
}

extern "C" __global__ void nerva_compute(float *sf, U64 *st, const float *pf, const U64 *pt,
                                       const float *inputs, unsigned *flags, unsigned n, U64 time, unsigned mode) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    State s;
    s.v=sf[0*n+row]; s.activity=sf[1*n+row]; s.input=sf[2*n+row]; s.threshold=sf[3*n+row];
    s.burst=sf[4*n+row]; s.adaptation=sf[5*n+row]; s.rebound=sf[6*n+row];
    s.threshold_adaptation=sf[7*n+row]; s.current=sf[8*n+row]; s.structural=sf[9*n+row];
    s.updated=st[0*n+row]; s.refractory=st[1*n+row]; s.last_spike=st[2*n+row];
    s.spike_valid=st[3*n+row]; s.spikes=st[4*n+row]; s.homeostasis=st[5*n+row];
    Params p;
    p.rest=pf[0*n+row]; p.reset=pf[1*n+row]; p.configured_threshold=pf[2*n+row];
    p.tau=pf[3*n+row]; p.trace_tau=pf[4*n+row]; p.drive=pf[5*n+row];
    p.burst_gain=pf[6*n+row]; p.burst_tau=pf[7*n+row]; p.adaptation_gain=pf[8*n+row];
    p.adaptation_tau=pf[9*n+row]; p.threshold_gain=pf[10*n+row]; p.threshold_tau=pf[11*n+row];
    p.rebound_gain=pf[12*n+row]; p.rebound_tau=pf[13*n+row]; p.refractory_us=pt[row];
    flags[row] = mode == 0 ? (advance(s,p,time) ? 0 : 2)
        : integrate(s,p,time,inputs[row],inputs[n+row],inputs[2*n+row]);
    sf[0*n+row]=s.v; sf[1*n+row]=s.activity; sf[2*n+row]=s.input; sf[3*n+row]=s.threshold;
    sf[4*n+row]=s.burst; sf[5*n+row]=s.adaptation; sf[6*n+row]=s.rebound;
    sf[7*n+row]=s.threshold_adaptation; sf[8*n+row]=s.current; sf[9*n+row]=s.structural;
    st[0*n+row]=s.updated; st[1*n+row]=s.refractory; st[2*n+row]=s.last_spike;
    st[3*n+row]=s.spike_valid; st[4*n+row]=s.spikes; st[5*n+row]=s.homeostasis;
}
