// Registration without ECC: moments + low-res NCC rotation sweep (coarse pose), then an edge-based homography fit:
// the reference contour is sampled along its normals in a wide window (area method, same measurement as the decoder),
// the offsets are fitted by an 8-parameter homography update (Gauss-Newton, robust IRLS), and the window is narrowed
// 0.6 -> 0.3 -> 0.2 mm.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "internal.hpp"

namespace stego {

namespace {

struct Coarse {
  int k = 1, w = 0, h = 0;
  std::vector<float> img;
  inline float px(int x, int y) const { return (x < 0 || y < 0 || x >= w || y >= h) ? 0.f : img[(size_t)y * w + x]; }
  // bilinear, zero outside; (x,y) in original crop px
  inline float sample(float x, float y) const {
    float u = x / k - 0.5f, v = y / k - 0.5f;
    int ix = (int)std::floor(u), iy = (int)std::floor(v);
    float fx = u - ix, fy = v - iy;
    if (ix >= 0 && iy >= 0 && ix + 1 < w && iy + 1 < h) {
      const float* r = &img[(size_t)iy * w + ix];
      float a = r[0] + fx * (r[1] - r[0]), b = r[w] + fx * (r[w + 1] - r[w]);
      return a + fy * (b - a);
    }
    float a = px(ix, iy) * (1 - fx) + px(ix + 1, iy) * fx;
    float b = px(ix, iy + 1) * (1 - fx) + px(ix + 1, iy + 1) * fx;
    return a * (1 - fy) + b * fy;
  }
};

void gauss_inplace(std::vector<float>& img, int w, int h, double sigma) {
  int r = std::max(1, (int)std::ceil(3 * sigma));
  std::vector<float> k(2 * r + 1);
  double s = 0;
  for (int i = -r; i <= r; i++) { k[i + r] = (float)std::exp(-0.5 * i * i / (sigma * sigma)); s += k[i + r]; }
  for (auto& v : k) v = (float)(v / s);
  std::vector<float> tmp((size_t)w * h);
  // horizontal: interior vectorisable, edges clamped
  for (int y = 0; y < h; y++) {
    const float* in = &img[(size_t)y * w];
    float* out = &tmp[(size_t)y * w];
    for (int x = 0; x < w; x++) {
      float a = 0;
      if (x >= r && x < w - r) { for (int i = -r; i <= r; i++) a += k[i + r] * in[x + i]; }
      else for (int i = -r; i <= r; i++) a += k[i + r] * in[std::clamp(x + i, 0, w - 1)];
      out[x] = a;
    }
  }
  for (int y = 0; y < h; y++) {
    float* out = &img[(size_t)y * w];
    for (int x = 0; x < w; x++) out[x] = 0;
    for (int i = -r; i <= r; i++) {
      const float* in = &tmp[(size_t)std::clamp(y + i, 0, h - 1) * w];
      const float kk = k[i + r];
      for (int x = 0; x < w; x++) out[x] += kk * in[x];
    }
  }
}

void build_coarse(const Dark& dk, double s, Coarse& C) {
  int k = std::max(1, (int)std::lround(s / 4.0));
  C.k = k; C.w = dk.w / k; C.h = dk.h / k;
  C.img.assign((size_t)C.w * C.h, 0.f);
  const float inv = 1.0f / (k * k);
  for (int y = 0; y < C.h; y++) {
    float* o = &C.img[(size_t)y * C.w];
    for (int dy = 0; dy < k; dy++) {
      const float* r = &dk.D[(size_t)(y * k + dy) * dk.w];
      for (int x = 0; x < C.w; x++) {
        float a = 0;
        for (int dx = 0; dx < k; dx++) a += r[x * k + dx];
        o[x] += a;
      }
    }
    for (int x = 0; x < C.w; x++) o[x] *= inv;
  }
  gauss_inplace(C.img, C.w, C.h, 0.5 * s / k);
}

struct Pose { double th, s, tx, ty; };

double ncc_pose(const Asset& A, const Coarse& C, const Pose& p, bool sub = false) {
  const double c = std::cos(p.th) * p.s, sn = std::sin(p.th) * p.s;
  const float m00 = (float)c, m01 = (float)-sn, m10 = (float)sn, m11 = (float)c, tx = (float)p.tx, ty = (float)p.ty;
  const std::vector<float>& TX = sub ? A.ctx2 : A.ctx; const std::vector<float>& TY = sub ? A.cty2 : A.cty; const std::vector<float>& TV = sub ? A.ctv2 : A.ctv;
  const int n = (int)TV.size();
  double sab = 0, sb = 0, sbb = 0;
  for (int j = 0; j < n; j++) {
    float x = m00 * TX[j] + m01 * TY[j] + tx, y = m10 * TX[j] + m11 * TY[j] + ty;
    float b = C.sample(x, y);
    sab += TV[j] * b; sb += b; sbb += (double)b * b;
  }
  double var = sbb - sb * sb / n;
  if (var <= 1e-9) return -1;
  return sab / std::sqrt(var);
}

void pose_to_H(const Pose& p, double* H) {
  double c = std::cos(p.th) * p.s, s = std::sin(p.th) * p.s;
  H[0] = c; H[1] = -s; H[2] = p.tx; H[3] = s; H[4] = c; H[5] = p.ty; H[6] = 0; H[7] = 0; H[8] = 1;
}

// Robust 8-parameter homography update from edge offsets. Returns inlier fraction (|resid| < 0.08 mm) of measured samples.
double edge_stage(const Asset& A, const Dark& dk, double* H, double W, double step, int iters, int stride, double* max_step_out = nullptr) {
  const auto& rs = A.regset(W, stride);
  const int n = (int)rs.idx.size();
  if (getenv("STEGO_DEBUG")) fprintf(stderr, "edge stage W=%.2f stride=%d: %d samples\n", W, stride, n);
  if (n < 50) return 0;
  std::vector<float> e(n), w(n), r(n), arbuf;
  arbuf.reserve(n);
  std::vector<uint8_t> ok(n);
  double inl = 0;
  constexpr int NP = 9;
  for (int it = 0; it < iters; it++) {
    measure_edges(A, dk.D.data(), dk.w, dk.h, H, W, step, rs.idx.data(), n, e.data(), ok.data());
    int nok = 0;
    for (int i = 0; i < n; i++) {
      bool g = ok[i] && std::fabs(e[i]) < 0.9 * W;
      w[i] = g ? 1.f : 0.f;
      nok += g;
    }
    if (nok < 50) return 0;
    double coef[NP] = {0};
    const float* F = rs.feat.data();
    for (int pass = 0; pass < 3; pass++) {
      float Mf[NP][12];
      double M[NP * NP] = {0}, b[NP] = {0};
      int cnt = 0;
      std::memset(Mf, 0, sizeof Mf);
      float bf[12] = {0};
      for (int i = 0; i < n; i++) {
        if (w[i] <= 0) continue;
        const float* __restrict x = F + (size_t)i * 12;
        for (int a = 0; a < NP; a++) {
          const float wa = w[i] * x[a];
          float* __restrict m = Mf[a];
          for (int c = 0; c < 12; c++) m[c] += wa * x[c];
        }
        const float we = w[i] * e[i];
        for (int c = 0; c < 12; c++) bf[c] += we * x[c];
        if (++cnt == 512) {   // flush float partial sums
          for (int a = 0; a < NP; a++) for (int c = 0; c < NP; c++) M[a * NP + c] += Mf[a][c];
          for (int c = 0; c < NP; c++) b[c] += bf[c];
          std::memset(Mf, 0, sizeof Mf); std::memset(bf, 0, sizeof bf); cnt = 0;
        }
      }
      for (int a = 0; a < NP; a++) for (int c = 0; c < NP; c++) M[a * NP + c] += Mf[a][c];
      for (int c = 0; c < NP; c++) b[c] += bf[c];
      double tr = 0;
      for (int a = 0; a < NP; a++) tr += M[a * NP + a];
      double Mc[NP * NP];
      std::memcpy(Mc, M, sizeof M);
      if (!chol_solve(Mc, b, NP, 1e-9 * tr / NP, coef)) return 0;
      // residuals + robust reweight
      float cf[12] = {0};
      for (int a = 0; a < NP; a++) cf[a] = (float)coef[a];
      std::vector<float>& ar = arbuf;
      ar.clear();
      for (int i = 0; i < n; i++) {
        const float* x = F + (size_t)i * 12;
        float f = 0;
        for (int a = 0; a < 12; a++) f += x[a] * cf[a];
        r[i] = e[i] - f;
        if (ok[i] && std::fabs(e[i]) < 0.9 * W) ar.push_back(std::fabs(r[i]));
      }
      if (pass == 1) break;
      size_t mid = ar.size() / 2;
      std::nth_element(ar.begin(), ar.begin() + mid, ar.end());
      double sc = std::max(1.4826 * ar[mid], 0.01 * W);
      for (int i = 0; i < n; i++) {
        if (!ok[i] || std::fabs(e[i]) >= 0.9 * W) { w[i] = 0; continue; }
        double u = r[i] / (4.685 * sc);
        w[i] = std::fabs(u) < 1 ? (float)((1 - u * u) * (1 - u * u)) : 0.f;
      }
    }
    {
      int ni = 0, nm = 0;
      for (int i = 0; i < n; i++) if (ok[i] && std::fabs(e[i]) < 0.9 * W) { nm++; ni += std::fabs(r[i]) < 0.08; }
      inl = nm ? (double)ni / nm : 0;
    }
    // apply: p' = T(p) in centred coords, T = [[I+G, u],[g, 1]]; H <- H * C T C^-1
    double T[9] = {1 + coef[2], coef[3], coef[0], coef[4], 1 + coef[5], coef[1], coef[6], coef[7], 1};
    double Cm[9] = {1, 0, A.cen[0], 0, 1, A.cen[1], 0, 0, 1}, Ci[9] = {1, 0, -A.cen[0], 0, 1, -A.cen[1], 0, 0, 1};
    double t1[9], t2[9];
    h_mul(Cm, T, t1);
    h_mul(t1, Ci, t2);
    h_mul(H, t2, H);
    h_norm(H);
    if (max_step_out) *max_step_out = std::max(std::fabs(coef[0]), std::fabs(coef[1]));
  }
  return inl;
}

}  // namespace

RegResult register_dark(const Asset& A, const Dark& dk, const Params& P, const double* H_init) {
  RegResult out;
  Timer t;
  double H[9];
  double score = 0;
  auto run_edges = [&](double* Hc) {
    double sc = 0;
    for (size_t s = 0; s < P.reg_W.size(); s++)
      sc = edge_stage(A, dk, Hc, P.reg_W[s], P.reg_step[s], P.reg_iters[s], P.reg_stride);
    return sc;
  };
  if (H_init) {
    std::memcpy(H, H_init, sizeof H);
    h_norm(H);
    out.t.coarse = 0;
    t.reset();
    score = run_edges(H);
    out.t.edge = t.ms();
    std::memcpy(out.H, H, sizeof H);
    out.score = score;
    return out;
  }
  // ---- coarse pose ----------------------------------------------------------------------------------------------
  double m0 = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (int y = 0; y < dk.h; y++) {
    const float* r = &dk.D[(size_t)y * dk.w];
    double rs = 0, rsx = 0, rsxx = 0;
    for (int x = 0; x < dk.w; x++) { rs += r[x]; rsx += r[x] * (x + 0.5); rsxx += r[x] * (x + 0.5) * (x + 0.5); }
    const double py = y + 0.5;
    m0 += rs; sx += rsx; sy += rs * py; sxx += rsxx; syy += rs * py * py; sxy += rsx * py;
  }
  const double mx = sx / m0, my = sy / m0;
  const double vxx = sxx / m0 - mx * mx, vyy = syy / m0 - my * my, vxy = sxy / m0 - mx * my;
  const double phi_s = 0.5 * std::atan2(2 * vxy, vxx - vyy);
  const double s = std::sqrt(m0 / A.ref_area_mm2);
  Coarse C;
  const bool dbg = getenv("STEGO_DEBUG") != nullptr;
  if (dbg) fprintf(stderr, "moments %.2f ms\n", t.ms());
  build_coarse(dk, s, C);
  if (dbg) fprintf(stderr, "build_coarse %.2f ms (k=%d)\n", t.ms(), C.k);
  auto centred = [&](double th) {
    Pose p{th, s, 0, 0};
    double c = std::cos(th) * s, sn = std::sin(th) * s;
    p.tx = mx - (c * A.ref_c[0] - sn * A.ref_c[1]);
    p.ty = my - (sn * A.ref_c[0] + c * A.ref_c[1]);
    return p;
  };
  struct Cand { double sc; Pose p; };
  std::vector<Cand> cands;
  const double PI = 3.14159265358979323846;
  std::vector<double> sweep;
  if (A.square_block) {
    const int ns_ = 360 / P.rot_step_deg;
    for (int i = 0; i < ns_; i++) {
      Pose p = centred(i * P.rot_step_deg * PI / 180);
      sweep.push_back(ncc_pose(A, C, p, true));
    }
    for (int i = 0; i < ns_; i++) {   // parabolic interpolation of the NCC peak over the angle
      double l = sweep[(i + ns_ - 1) % ns_], c = sweep[i], r = sweep[(i + 1) % ns_];
      double d = l - 2 * c + r;
      double off = (d < 0 && c >= l && c >= r) ? 0.5 * (l - r) / d : 0.0;
      Pose p = centred((i + off) * P.rot_step_deg * PI / 180);
      cands.push_back({c, p});
    }
  } else {
    for (double th : {phi_s - A.ref_phi, phi_s - A.ref_phi + PI}) {
      Pose p = centred(th);
      cands.push_back({ncc_pose(A, C, p), p});
    }
  }
  if (dbg) fprintf(stderr, "sweep %.2f ms\n", t.ms());
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.sc > b.sc; });
  std::vector<Cand> picks;
  const int maxp = A.square_block ? P.n_cands : 1;
  for (auto& c : cands) {
    bool sep = true;
    for (auto& q : picks) {
      double d = std::fabs(std::remainder(c.p.th - q.p.th, 2 * PI));
      if (d <= 30 * PI / 180) sep = false;
    }
    if (sep) picks.push_back(c);
    if ((int)picks.size() == maxp) break;
  }
  // local refinement of each pick (greedy coordinate search on rotation, log-scale, translation)
  for (auto& c : picks) {
    double dth = 4 * PI / 180, dls = 0.03, dt = 0.5 * s;
    double best = c.sc;
    for (int round = 0; round < P.coarse_rounds; round++) {
      bool moved = true;
      int guard = 0;
      while (moved && guard++ < 6) {
        moved = false;
        for (int par = 0; par < 4; par++)
          for (int sg = -1; sg <= 1; sg += 2) {
            Pose q = c.p;
            if (par == 0) q.th += sg * dth;
            else if (par == 1) q.s *= std::exp(sg * dls);
            else if (par == 2) q.tx += sg * dt;
            else q.ty += sg * dt;
            double v = ncc_pose(A, C, q);
            if (v > best + 1e-6) { best = v; c.p = q; c.sc = v; moved = true; }
          }
      }
      dth *= 0.5; dls *= 0.5; dt *= 0.5;
    }
  }
  std::sort(picks.begin(), picks.end(), [](const Cand& a, const Cand& b) { return a.sc > b.sc; });
  if (dbg) fprintf(stderr, "refine %.2f ms\n", t.ms());
  out.t.coarse = t.ms();
  t.reset();
  double bestscore = -1;
  for (size_t ci = 0; ci < picks.size(); ci++) {
    double Hc[9];
    pose_to_H(picks[ci].p, Hc);
    double sc = run_edges(Hc);
    if (getenv("STEGO_DEBUG")) fprintf(stderr, "cand %zu th=%.1f s=%.2f ncc=%.3f -> edge inlier=%.3f\n", ci, picks[ci].p.th * 180 / PI, picks[ci].p.s, picks[ci].sc, sc);
    if (sc > bestscore) { bestscore = sc; std::memcpy(H, Hc, sizeof H); out.cand = (int)ci; }
    if (sc >= 0.5) break;
  }
  out.t.edge = t.ms();
  std::memcpy(out.H, H, sizeof H);
  out.score = bestscore;
  return out;
}

}  // namespace stego
