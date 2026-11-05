// Nuisance fit (homography refinement + poly-4 warp + dot gain), per-contour high-pass, matched filter, despread, Golay, CRC.
#include <algorithm>
#include <numeric>
#include "internal.hpp"
#include "stego/demo.hpp"

namespace stego {

namespace {

struct NormalEq {
  double G[NPAD * NPAD];
  double b[NPAD];
  void zero() { std::memset(this, 0, sizeof *this); }
};

// G += sum_j w_j x_j x_j^T ; b += sum_j w_j x_j e_j  (rows act[j] of reg; w,e indexed by j). sign=+1 add, -1 remove.
void accumulate(const Asset& A, const int32_t* act, const float* w, const float* e, int n, NormalEq& ne, double sign = 1.0) {
  float Gf[NCOL][NPAD];
  float bf[NPAD];
  int j = 0;
  while (j < n) {
    std::memset(Gf, 0, sizeof Gf);
    std::memset(bf, 0, sizeof bf);
    int jend = std::min(n, j + 256);
    for (; j < jend; j++) {
      if (w[j] <= 0) continue;
      const float* __restrict x = &A.reg[(size_t)act[j] * NPAD];
      const float wj = w[j];
      for (int k = 0; k < NCOL; k++) {
        const float a = wj * x[k];
        float* __restrict g = Gf[k];
        for (int c = 0; c < NPAD; c++) g[c] += a * x[c];
      }
      const float we = wj * e[j];
      for (int c = 0; c < NPAD; c++) bf[c] += we * x[c];
    }
    for (int k = 0; k < NCOL; k++)
      for (int c = 0; c < NPAD; c++) ne.G[k * NPAD + c] += sign * Gf[k][c];
    for (int c = 0; c < NPAD; c++) ne.b[c] += sign * bf[c];
  }
}

void rhs_only(const Asset& A, const int32_t* act, const float* w, const float* e, int n, double* b) {
  float bf[NPAD] = {0};
  for (int c = 0; c < NPAD; c++) b[c] = 0;
  int j = 0;
  while (j < n) {
    std::memset(bf, 0, sizeof bf);
    int jend = std::min(n, j + 256);
    for (; j < jend; j++) {
      if (w[j] <= 0) continue;
      const float* __restrict x = &A.reg[(size_t)act[j] * NPAD];
      const float we = w[j] * e[j];
      for (int c = 0; c < NPAD; c++) bf[c] += we * x[c];
    }
    for (int c = 0; c < NPAD; c++) b[c] += bf[c];
  }
}

bool solve(const NormalEq& ne, double* coef) {
  double M[NCOL * NCOL], b[NCOL];
  double tr = 0;
  for (int i = 0; i < NCOL; i++) {
    for (int j = 0; j < NCOL; j++) M[i * NCOL + j] = ne.G[i * NPAD + j];
    b[i] = ne.b[i];
    tr += M[i * NCOL + i];
  }
  return chol_solve(M, b, NCOL, 1e-9 * tr / NCOL, coef);
}

inline float dot31(const float* x, const float* c) {
  float s = 0;
  for (int k = 0; k < NPAD; k++) s += x[k] * c[k];
  return s;
}

// median of v (destroyed order); matches np.median
float median_of(std::vector<float>& v) {
  size_t n = v.size();
  if (!n) return 0;
  size_t mid = n / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  float hi = v[mid];
  if (n & 1) return hi;
  float lo = *std::max_element(v.begin(), v.begin() + mid);
  return 0.5f * (lo + hi);
}

uint16_t crc16_bits(const uint8_t* bits, int n) {
  uint16_t reg = 0xFFFF;
  for (int i = 0; i < n; i++) {
    uint16_t top = ((reg >> 15) & 1) ^ bits[i];
    reg = (uint16_t)(reg << 1);
    if (top) reg ^= 0x1021;
  }
  return reg;
}
uint8_t crc8_bits(const uint8_t* bits, int n) {
  uint8_t reg = 0;
  for (int i = 0; i < n; i++) {
    uint8_t top = ((reg >> 7) & 1) ^ bits[i];
    reg = (uint8_t)(reg << 1);
    if (top) reg ^= 0x07;
  }
  return reg;
}

struct Stat { double z = 0, gain = 0, s2 = 0; };

// pilot statistics of ConvSpreader.decode (golay_crc16) / Spreader.decode (rep_crc8)
Stat pilot_stat(const Asset& A, const double* y, const double* w, const float* pil = nullptr) {
  Stat st;
  if (!pil) pil = A.pilot.data();
  const int nc = A.n_chips;
  if (A.code == "rep_crc8") {
    double num = 0, e2 = 0, sw = 0;
    for (int i = 0; i < nc; i++)
      if (w[i] > 0) { num += pil[i] * y[i]; e2 += y[i] * y[i]; sw += w[i]; }
    st.z = num / std::sqrt(e2 + 1e-30);
    st.gain = num / (sw * A.rho + 1e-30);
    return st;
  }
  double sw = 0, num = 0;
  int npm = 0;
  for (int i = 0; i < nc; i++)
    if (A.pilot_mask[i] && w[i] > 0) { sw += w[i]; num += pil[i] * y[i]; npm++; }
  sw += 1e-30;
  const double gain = num / sw;
  double s2 = 0;
  for (int i = 0; i < nc; i++)
    if (A.pilot_mask[i] && w[i] > 0) {
      double r = y[i] - gain * w[i] * pil[i];
      s2 += r * r / std::max(w[i], 1e-12);
    }
  s2 /= std::max(npm - 1, 1);
  st.gain = gain;
  st.s2 = s2;
  st.z = num / std::sqrt(s2 * sw + 1e-30);
  return st;
}


// Null calibration: the same statistic against K fixed pseudo-random +-1 decoy sequences placed on the same chips as the pilot
// (pilot-mask chips; all chips for rep_crc8). Decoys see the same static misfit / noise but carry no signal.
constexpr int N_DECOY = 32;
void null_stats(const Asset& A, const double* y, const double* w, double& mean, double& sd) {
  const int nc = A.n_chips;
  std::vector<float> dec(nc);
  double s1 = 0, s2 = 0;
  for (int k = 0; k < N_DECOY; k++) {
    uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t)(k + 1) + (uint64_t)nc;   // splitmix64 stream, deterministic
    for (int i = 0; i < nc; i++) {
      x += 0x9E3779B97F4A7C15ull;
      uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      z ^= z >> 31;
      dec[i] = (z & 1) ? 1.f : -1.f;
    }
    double zk = pilot_stat(A, y, w, dec.data()).z;
    s1 += zk; s2 += zk * zk;
  }
  mean = s1 / N_DECOY;
  sd = std::sqrt(std::max(s2 / N_DECOY - mean * mean, 1e-12) * N_DECOY / (N_DECOY - 1));
}

void despread(const Asset& A, const double* y, const double* w, Stat st, Result& out) {
  const int nc = A.n_chips, ns = A.n_sym;
  std::vector<double> c(ns, 0.0), en(ns, 0.0);
  if (A.code == "rep_crc8") {
    // residual after removing the pilot, per symbol
    for (int i = 0; i < nc; i++) {
      double resid = y[i] - st.gain * w[i] * A.rho * A.pilot[i];
      c[A.bit_of[i]] += A.pn[i] * resid;
      en[A.bit_of[i]] += w[i];
    }
    out.soft.assign(ns, 0.f);
    std::vector<uint8_t> bits(ns);
    for (int k = 0; k < ns; k++) {
      double sf = c[k] / (st.gain * (1 - A.rho) * en[k] + 1e-30);
      out.soft[k] = (float)sf;
      bits[k] = sf > 0;
    }
    out.bits.assign(bits.begin(), bits.begin() + A.n_bits);
    uint8_t cr = crc8_bits(out.bits.data(), A.n_bits), rx = 0;
    for (int i = 0; i < 8; i++) rx = (uint8_t)((rx << 1) | bits[A.n_bits + i]);
    out.crc_ok = cr == rx;
    return;
  }
  for (int i = 0; i < nc; i++)
    if (!A.pilot_mask[i]) {
      c[A.bit_of[i]] += A.pn[i] * y[i];
      en[A.bit_of[i]] += w[i];
    }
  out.soft.assign(ns, 0.f);
  if (std::fabs(st.gain) > 1e-12)
    for (int k = 0; k < ns; k++) out.soft[k] = (float)(c[k] / (st.gain * en[k] + 1e-30));
  // exact soft ML Golay(24,12), ns/24 blocks
  std::vector<uint8_t> info;
  for (int b = 0; b < ns / 24; b++) {
    float cc[24];
    for (int k = 0; k < 24; k++) cc[k] = (float)c[b * 24 + k];
    int bestm = 0;
    float best = -1e30f;
    for (int m = 0; m < 4096; m++) {
      const float* cw = &A.golay_pm[(size_t)m * 24];
      float s = 0;
      for (int k = 0; k < 24; k++) s += cc[k] * cw[k];
      if (s > best) { best = s; bestm = m; }
    }
    for (int k = 0; k < 12; k++) info.push_back((bestm >> (11 - k)) & 1);
  }
  out.bits.assign(info.begin(), info.begin() + A.n_bits);
  uint16_t cr = crc16_bits(out.bits.data(), A.n_bits), rx = 0;
  for (int i = 0; i < 16; i++) rx = (uint16_t)((rx << 1) | info[A.n_bits + i]);
  out.crc_ok = cr == rx;
}

// circular moving-average high-pass per contour, in place into hp (for all samples)
void highpass(const Asset& A, const float* e, const float* v, float* hp, std::vector<double>& scratch) {
  const int nco = (int)A.contour_n.size();
  for (int k = 0; k < nco; k++) {
    const int s0 = A.contour_start[k], n = A.contour_n[k], win = A.hp_win[k], h = win / 2;
    scratch.resize(2 * (size_t)(n + 2 * h + 1));
    double* cve = scratch.data();
    double* cv = cve + (n + 2 * h + 1);
    cve[0] = 0; cv[0] = 0;
    for (int i = 0; i < n + 2 * h; i++) {
      int j = ((i - h) % n + n) % n;
      cve[i + 1] = cve[i] + (double)v[s0 + j] * e[s0 + j];
      cv[i + 1] = cv[i] + v[s0 + j];
    }
    for (int i = 0; i < n; i++) {
      double a = cve[i + win] - cve[i], b = cv[i + win] - cv[i];
      hp[s0 + i] = (float)(e[s0 + i] - a / (b + 1e-9));
    }
  }
}

void matched(const Asset& A, const float* hp, const float* wv, double* y, double* cw, bool accum_cw) {
  std::fill(y, y + A.n_chips, 0.0);
  if (accum_cw) std::fill(cw, cw + A.n_chips, 0.0);
  for (int s = 0; s < A.ns; s++) {
    if (!(wv[s] > 0)) continue;
    const double wgt = A.w0[s];
    const double ga = wgt * A.wa[s], gb = wgt * A.wb[s];
    y[A.ca[s]] += ga * hp[s];
    y[A.cb[s]] += gb * hp[s];
    if (accum_cw) { cw[A.ca[s]] += ga * ga; cw[A.cb[s]] += gb * gb; }
  }
}

}  // namespace

Result decode_dark(const Asset& A, const Dark& dk, const Params& P, const double* Hin, double* reg_score, Timer& total) {
  Result R;
  Timer t;
  double H[9];
  std::memcpy(H, Hin, sizeof H);
  h_norm(H);
  const int ns = A.ns;
  const std::vector<int32_t>& act0 = A.fit_idx;
  const int n0 = (int)act0.size();
  std::vector<int32_t> act;
  std::vector<float> e(n0), wt(n0), wg(n0), w0v(n0);
  std::vector<uint8_t> okv(n0);
  NormalEq ne;
  double coef[NCOL];
  double meas_ms = 0, fit_ms = 0;
  std::vector<float> fx(n0), fy(n0);

  auto subset = [&](int stride) {
    act.clear();
    for (int i = 0; i < n0; i += stride) act.push_back(act0[i]);
  };

  int n = 0;
  for (int it = 0; it <= P.n_iter; it++) {
    const bool fin = it == P.n_iter;
    subset(fin ? 1 : std::max(1, P.iter_stride));
    n = (int)act.size();
    t.reset();
    measure_edges(A, dk.D.data(), dk.w, dk.h, H, P.W > 0 ? P.W : A.W, fin ? P.step : P.iter_step, act.data(), n, e.data(), okv.data());
    meas_ms += t.ms();
    t.reset();
    for (int j = 0; j < n; j++) wt[j] = okv[j] ? A.w0[act[j]] : 0.f;
    const int fs = fin ? std::max(1, P.fit_stride) : 1;   // rows that enter the normal equations
    for (int j = 0; j < n; j++) wg[j] = (j % fs == 0) ? wt[j] : 0.f;
    // IRLS: 2 passes, drop > 4 sd
    ne.zero();
    accumulate(A, act.data(), wg.data(), e.data(), n, ne);
    for (int pass = 0; pass < 2; pass++) {
      if (!solve(ne, coef)) break;
      float cf[NPAD] = {0};
      for (int k = 0; k < NCOL; k++) cf[k] = (float)coef[k];
      double sw = 0, sr = 0;
      std::vector<float>& r = fx;  // reuse as residual store
      for (int j = 0; j < n; j++) {
        r[j] = e[j] - dot31(&A.reg[(size_t)act[j] * NPAD], cf);
        if (wt[j] > 0) { sw += wt[j]; sr += (double)wt[j] * r[j] * r[j]; }
      }
      const double sd = std::sqrt(sr / (sw + 1e-12));
      std::vector<int32_t> da;
      std::vector<float> dw, de;
      for (int j = 0; j < n; j++)
        if (wt[j] > 0 && std::fabs(r[j]) > 4 * sd) {
          if (wg[j] > 0) { da.push_back(act[j]); dw.push_back(wg[j]); de.push_back(e[j]); }
          wt[j] = 0; wg[j] = 0;
        }
      if (!da.empty()) accumulate(A, da.data(), dw.data(), de.data(), (int)da.size(), ne, -1.0);
    }
    if (fin) { fit_ms += t.ms(); break; }
    // affine part of the fitted displacement field -> update H
    {
      double S[3][3] = {{0}}, B[3][2] = {{0}};
      for (int j = 0; j < n; j++) {
        if (wt[j] <= 0) continue;
        const int s = act[j];
        const float* ph = &A.phi[(size_t)s * 16];
        double fxv = 0, fyv = 0;
        for (int k = 0; k < NT; k++) { fxv += ph[k] * coef[k]; fyv += ph[k] * coef[NT + k]; }
        double xc[3] = {A.Px[s] - A.cen[0], A.Py[s] - A.cen[1], 1.0};
        for (int a = 0; a < 3; a++) {
          for (int b = 0; b < 3; b++) S[a][b] += wt[j] * xc[a] * xc[b];
          B[a][0] += wt[j] * xc[a] * fxv;
          B[a][1] += wt[j] * xc[a] * fyv;
        }
      }
      double M[9] = {S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2]};
      double Mi[9];
      if (h_inv(M, Mi)) {
        double Bs[3][2];
        for (int a = 0; a < 3; a++)
          for (int c = 0; c < 2; c++) Bs[a][c] = Mi[a * 3] * B[0][c] + Mi[a * 3 + 1] * B[1][c] + Mi[a * 3 + 2] * B[2][c];
        double G[2][2] = {{Bs[0][0], Bs[1][0]}, {Bs[0][1], Bs[1][1]}};
        double u[2] = {Bs[2][0] - (G[0][0] * A.cen[0] + G[0][1] * A.cen[1]), Bs[2][1] - (G[1][0] * A.cen[0] + G[1][1] * A.cen[1])};
        double A3[9] = {1 + G[0][0], G[0][1], u[0], G[1][0], 1 + G[1][1], u[1], 0, 0, 1};
        h_mul(H, A3, H);
        h_norm(H);
      }
    }
    fit_ms += t.ms();
  }
  t.reset();
  // ---- final: residuals, wv (full-size arrays) -------------------------------------------------------------------------
  std::vector<float> ef(ns, 0.f), wv(ns, 0.f), es(ns, 0.f), hp(ns, 0.f), hs(ns, 0.f);
  {
    float cf[NPAD] = {0};
    for (int k = 0; k < NCOL; k++) cf[k] = (float)coef[k];
    for (int j = 0; j < n; j++) {
      const int s = act[j];
      ef[s] = e[j] - dot31(&A.reg[(size_t)s * NPAD], cf);
      wv[s] = wt[j];
    }
    // synthetic pilot-only displacement passed through the same projection (same weights)
    std::vector<float> esj(n);
    for (int j = 0; j < n; j++) esj[j] = A.esyn[act[j]];
    NormalEq ns_ = ne;     // same weights -> same Gram; only the right-hand side differs
    rhs_only(A, act.data(), wg.data(), esj.data(), n, ns_.b);
    double cs[NCOL];
    solve(ns_, cs);
    float cfs[NPAD] = {0};
    for (int k = 0; k < NCOL; k++) cfs[k] = (float)cs[k];
    for (int j = 0; j < n; j++) es[act[j]] = esj[j] - dot31(&A.reg[(size_t)act[j] * NPAD], cfs);
  }
  fit_ms += t.ms();
  t.reset();
  std::vector<double> scratch, y(A.n_chips), cw(A.n_chips), ysyn(A.n_chips);
  highpass(A, ef.data(), wv.data(), hp.data(), scratch);
  // MAD clip
  if (P.clip_mad > 0) {
    std::vector<float> v;
    v.reserve(n);
    for (int j = 0; j < n; j++) if (wv[act[j]] > 0) v.push_back(hp[act[j]]);
    if (!v.empty()) {
      float med = median_of(v);
      for (auto& x : v) x = std::fabs(x - med);
      float mad = median_of(v);
      float lim = (float)(P.clip_mad * 1.4826 * mad);
      for (int j = 0; j < n; j++) { float& x = hp[act[j]]; x = std::min(std::max(x, -lim), lim); }
    }
  }
  matched(A, hp.data(), wv.data(), y.data(), cw.data(), true);
  highpass(A, es.data(), wv.data(), hs.data(), scratch);
  matched(A, hs.data(), wv.data(), ysyn.data(), cw.data(), false);
  double ret = std::max(pilot_stat(A, ysyn.data(), cw.data()).gain, 1e-3);
  Stat raw = pilot_stat(A, y.data(), cw.data());
  for (auto& v : y) v /= ret;
  Stat st = pilot_stat(A, y.data(), cw.data());
  double tail_ms = t.ms();
  t.reset();
  despread(A, y.data(), cw.data(), st, R);
  R.t.golay = t.ms();
  R.z_pilot = st.z;
  null_stats(A, y.data(), cw.data(), R.null_mean, R.null_sd);
  R.z_cal = (st.z - R.null_mean) / R.null_sd;
  R.pilot_s2 = st.s2;
  R.chip_y = y;
  R.chip_w = cw;
  R.amp_um = st.gain * 1000;
  R.amp_raw_um = raw.gain * 1000;
  R.hp_retention = ret;
  R.hex = bits_to_hex(R.bits);
  R.accept = R.crc_ok && (!A.has_zacc || R.z_cal > A.z_accept);
  int nv = 0;
  for (int s = 0; s < ns; s++) nv += wv[s] > 0;
  R.n_valid = nv;
  R.n_points = ns;
  std::memcpy(R.H, H, sizeof H);
  R.t.measure = meas_ms; R.t.fit = fit_ms; R.t.tail = tail_ms;
  (void)total; (void)reg_score;
  return R;
}

// Accumulated chip statistics (sum of inverse-variance weighted y / w of several decodes of the same block) -> full result.
Result finalize_chips(const Asset& A, const double* y, const double* w) {
  Result R;
  Stat st = pilot_stat(A, y, w);
  despread(A, y, w, st, R);
  R.z_pilot = st.z;
  null_stats(A, y, w, R.null_mean, R.null_sd);
  R.z_cal = (st.z - R.null_mean) / R.null_sd;
  R.pilot_s2 = st.s2;
  R.amp_um = R.amp_raw_um = st.gain * 1000;
  R.hex = bits_to_hex(R.bits);
  R.accept = R.crc_ok && (!A.has_zacc || R.z_cal > A.z_accept);
  return R;
}


// ---- demo token: guarded per-region statistics (port of geostego/demo.py region_stats)
// A region contributes only if enough pilot chips carry weight: chips with w <= GUARD_W_REL * median(w of pilot chips with w > 0) are dropped; the region is
// dropped (z = 0, no LLR) if fewer than GUARD_MIN_N or GUARD_MIN_FRAC of its pilot chips remain, or if the noise normaliser collapses (s2 < GUARD_S2_MIN).
constexpr double GUARD_W_REL = 0.2, GUARD_MIN_FRAC = 0.3, GUARD_S2_MIN = 1e-9;
constexpr int GUARD_MIN_N = 20;

static bool guarded_stat(const Asset& A, const double* y, const double* w, Stat& st, double& thr) {
  std::vector<double> pw;
  int npil = 0;
  for (int i = 0; i < A.n_chips; i++) if (A.pilot_mask[i]) { npil++; if (w[i] > 0) pw.push_back(w[i]); }
  if (pw.empty()) return false;
  std::sort(pw.begin(), pw.end());
  const size_t m = pw.size();
  const double med = (m % 2) ? pw[m / 2] : 0.5 * (pw[m / 2 - 1] + pw[m / 2]);
  thr = GUARD_W_REL * med;
  double sw = 0, num = 0;
  int n = 0;
  for (int i = 0; i < A.n_chips; i++)
    if (A.pilot_mask[i] && w[i] > thr) { sw += w[i]; num += A.pilot[i] * y[i]; n++; }
  if (n < GUARD_MIN_N || n < GUARD_MIN_FRAC * npil || sw <= 0) return false;
  const double gain = num / sw;
  double s2 = 0;
  for (int i = 0; i < A.n_chips; i++)
    if (A.pilot_mask[i] && w[i] > thr) { const double r = y[i] - gain * w[i] * A.pilot[i]; s2 += r * r / std::max(w[i], 1e-12); }
  s2 /= std::max(n - 1, 1);
  if (!(s2 >= GUARD_S2_MIN)) return false;
  st.gain = gain; st.s2 = s2; st.z = num / std::sqrt(s2 * sw);
  return true;
}

// ---- demo token: combined decision over regions (geostego/demo.py combine)
DemoResult combine_demo(const std::vector<DemoRegion>& regions, double z_accept, double zmin_use) {
  DemoResult out;
  int ns = 0;
  for (const auto& rg : regions) if (rg.asset) { ns = rg.asset->n_sym; break; }
  if (ns == 0) return out;
  std::vector<double> c(ns, 0.0);
  for (const auto& rg : regions) {
    if (!rg.asset || !rg.r || rg.r->chip_y.empty()) { out.zs.push_back(0.0); out.gains.push_back(0.0); continue; }
    const Asset& A = *rg.asset;
    const double* y = rg.r->chip_y.data();
    const double* w = rg.r->chip_w.data();
    Stat st; double thr = 0;
    if (!guarded_stat(A, y, w, st, thr)) { out.zs.push_back(0.0); out.gains.push_back(0.0); continue; }
    out.zs.push_back(st.z); out.gains.push_back(st.gain);
    if (st.z >= zmin_use && st.gain > 0 && A.n_sym == ns) {
      std::vector<double> cr(ns, 0.0);
      for (int i = 0; i < A.n_chips; i++)
        if (!A.pilot_mask[i] && w[i] > thr) cr[A.bit_of[i]] += A.pn[i] * y[i];
      const double k = 2.0 * st.gain / (st.s2 + 1e-30);
      for (int j = 0; j < ns; j++) c[j] += k * cr[j];
    }
  }
  double zsum = 0;
  for (double z : out.zs) zsum += z;
  out.z = out.zs.empty() ? 0.0 : zsum / std::sqrt((double)out.zs.size());
  const Asset* A0 = nullptr;
  for (const auto& rg : regions) if (rg.asset) { A0 = rg.asset; break; }
  std::vector<uint8_t> info;
  for (int b = 0; b < ns / 24; b++) {
    int bestm = 0; double best = -1e300;
    for (int m = 0; m < 4096; m++) {
      const float* cw = &A0->golay_pm[(size_t)m * 24];
      double sc = 0;
      for (int k = 0; k < 24; k++) sc += c[b * 24 + k] * cw[k];
      if (sc > best) { best = sc; bestm = m; }
    }
    for (int k = 0; k < 12; k++) info.push_back((bestm >> (11 - k)) & 1);
  }
  out.bits = info;
  const int nb = (int)info.size() - 16;
  uint16_t cr = crc16_bits(info.data(), nb), rx = 0;
  for (int i = 0; i < 16; i++) rx = (uint16_t)((rx << 1) | info[nb + i]);
  out.crc_ok = cr == rx;
  if (nb == 44) {
    uint64_t v = 0; int tag = 0;
    for (int i = 0; i < kTokenBits; i++) v = (v << 1) | info[i];
    for (int i = kTokenBits; i < kTokenBits + kTagBits; i++) tag = (tag << 1) | info[i];
    uint64_t lim = 1; for (int i = 0; i < kTokenLen; i++) lim *= 62;
    out.tag_ok = tag == kTag && v < lim;
    if (out.crc_ok && out.tag_ok) {
      static const char AL[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
      std::string t(kTokenLen, '0');
      for (int i = kTokenLen - 1; i >= 0; i--) { t[i] = AL[v % 62]; v /= 62; }
      out.token = t;
    }
  }
  out.accept = out.crc_ok && out.tag_ok && out.z > z_accept;
  return out;
}

}  // namespace stego
