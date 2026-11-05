// Mark detector (port of geostego/marks_detect.py, no OpenCV).
//
// Stage A  centre + scale: zero-mean NCC (TM_CCOEFF_NORMED) of the darkness map, area-resampled to a working 2.5 px/mm for every candidate
//          scale, against a ROTATION-AVERAGED (radial profile) template of the mark (built from the asset's 20 px/mm reference raster).
//          Direct (shifted accumulate) or FFT correlation, whichever is cheaper; window sums from integral images.
// Stage B  rotation + scale: polar resampling around the stage-A centre; rotation = circular shift in angle, scale = radial stretch;
//          exhaustive over 1-2 degree shifts x 11 scales (replaces the oriented-template sweep of the Python version).
// Stage C  centre: oriented-template NCC over a small window, then Gauss-Newton on the zero-mean NCC (ECC objective), first as a
//          similarity, then as an affine warp (tilt). Result H = asset page mm -> image px (pixel i covers [i, i+1)).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include "internal.hpp"
#include "stego/marks.hpp"

namespace stego {

namespace {

constexpr double A_PPM = 2.5;
constexpr double PI = 3.14159265358979323846;

// ------------------------------------------------------------------------------------------------ small helpers
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// bilinear sample, edge replicate; (x, y) in pixel-index coordinates
inline float bilerp_c(const float* I, int w, int h, float x, float y) {
  x = clampf(x, 0.f, w - 1.001f);
  y = clampf(y, 0.f, h - 1.001f);
  const int ix = (int)x, iy = (int)y;
  const float fx = x - ix, fy = y - iy;
  const float* r = I + (size_t)iy * w + ix;
  const float a = r[0] + fx * (r[1] - r[0]), b = r[w] + fx * (r[w + 1] - r[w]);
  return a + fy * (b - a);
}
// bilinear sample, zero outside
inline float bilerp_z(const float* I, int w, int h, float x, float y) {
  const int ix = (int)std::floor(x), iy = (int)std::floor(y);
  const float fx = x - ix, fy = y - iy;
  auto px = [&](int xx, int yy) { return (xx < 0 || yy < 0 || xx >= w || yy >= h) ? 0.f : I[(size_t)yy * w + xx]; };
  const float a = px(ix, iy) + fx * (px(ix + 1, iy) - px(ix, iy));
  const float b = px(ix, iy + 1) + fx * (px(ix + 1, iy + 1) - px(ix, iy + 1));
  return a + fy * (b - a);
}

float dot_f(const float* a, const float* b, int n) {   // 8 independent accumulators: vectorises without -ffast-math
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5]; s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
  }
  float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  for (; i < n; i++) s += a[i] * b[i];
  return s;
}

inline int reflect101(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
  return i;
}

void gauss_blur(std::vector<float>& img, int w, int h, double sigma) {
  const int r = std::max(1, (int)std::lround(sigma * 4));
  std::vector<float> k(2 * r + 1);
  double s = 0;
  for (int i = -r; i <= r; i++) { k[i + r] = (float)std::exp(-0.5 * i * i / (sigma * sigma)); s += k[i + r]; }
  for (auto& v : k) v = (float)(v / s);
  std::vector<float> tmp((size_t)w * h);
  std::vector<int> idx(w + 2 * r);
  for (int x = -r; x < w + r; x++) idx[x + r] = reflect101(x, w);
  for (int y = 0; y < h; y++) {
    const float* in = &img[(size_t)y * w];
    float* out = &tmp[(size_t)y * w];
    for (int x = 0; x < w; x++) {
      float a = 0;
      if (x >= r && x < w - r) { for (int i = -r; i <= r; i++) a += k[i + r] * in[x + i]; }
      else for (int i = -r; i <= r; i++) a += k[i + r] * in[idx[x + i + r]];
      out[x] = a;
    }
  }
  for (int y = 0; y < h; y++) {
    float* out = &img[(size_t)y * w];
    for (int x = 0; x < w; x++) out[x] = 0;
    for (int i = -r; i <= r; i++) {
      const float* in = &tmp[(size_t)reflect101(y + i, h) * w];
      const float kk = k[i + r];
      for (int x = 0; x < w; x++) out[x] += kk * in[x];
    }
  }
}

// area (box-weighted) resample src (w x h) -> (ow x oh), like cv2.INTER_AREA for downscaling
void area_resize(const float* src, int w, int h, int ow, int oh, std::vector<float>& dst) {
  struct Tap { int i0, n; std::vector<float> w; };
  auto taps = [](int n_in, int n_out) {
    std::vector<Tap> t(n_out);
    const double sc = (double)n_in / n_out;
    for (int o = 0; o < n_out; o++) {
      double a = o * sc, b = (o + 1) * sc;
      int i0 = (int)std::floor(a), i1 = std::min(n_in, (int)std::ceil(b));
      t[o].i0 = i0; t[o].n = i1 - i0;
      t[o].w.resize(t[o].n);
      for (int i = i0; i < i1; i++) t[o].w[i - i0] = (float)((std::min(b, (double)i + 1) - std::max(a, (double)i)) / sc);
    }
    return t;
  };
  auto tx = taps(w, ow), ty = taps(h, oh);
  std::vector<float> tmp((size_t)h * ow);
  for (int y = 0; y < h; y++) {
    const float* r = src + (size_t)y * w;
    float* o = &tmp[(size_t)y * ow];
    for (int x = 0; x < ow; x++) {
      const Tap& t = tx[x];
      float a = 0;
      for (int i = 0; i < t.n; i++) a += t.w[i] * r[t.i0 + i];
      o[x] = a;
    }
  }
  dst.assign((size_t)ow * oh, 0.f);
  for (int y = 0; y < oh; y++) {
    float* o = &dst[(size_t)y * ow];
    const Tap& t = ty[y];
    for (int i = 0; i < t.n; i++) {
      const float* r = &tmp[(size_t)(t.i0 + i) * ow];
      const float wv = t.w[i];
      for (int x = 0; x < ow; x++) o[x] += wv * r[x];
    }
  }
}

// uint8 2x2 box (rounded); source stride in elements
void box2_u8(const uint8_t* s, int w, int h, size_t stride, std::vector<uint8_t>& d, int& ow, int& oh) {
  ow = w / 2; oh = h / 2;
  d.resize((size_t)ow * oh);
  for (int y = 0; y < oh; y++) {
    const uint8_t* a = s + (size_t)(2 * y) * stride;
    const uint8_t* b = a + stride;
    uint8_t* o = &d[(size_t)y * ow];
    for (int x = 0; x < ow; x++) o[x] = (uint8_t)((a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + 2) >> 2);
  }
}

// ------------------------------------------------------------------------------------------------ FFT (mixed radix, kissfft style)
struct cf { float r, i; };
inline cf operator*(cf a, cf b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }
inline cf operator+(cf a, cf b) { return {a.r + b.r, a.i + b.i}; }
inline cf operator-(cf a, cf b) { return {a.r - b.r, a.i - b.i}; }

class FFT1 {
 public:
  FFT1(int n, bool inv) : n_(n), inv_(inv), tw_(n) {
    for (int i = 0; i < n; i++) {
      double ph = -2 * PI * i / n;
      if (inv) ph = -ph;
      tw_[i] = {(float)std::cos(ph), (float)std::sin(ph)};
    }
    int m = n, p = 4;
    const double fl = std::floor(std::sqrt((double)m));
    while (m > 1) {
      while (m % p) {
        switch (p) { case 4: p = 2; break; case 2: p = 3; break; default: p += 2; }
        if (p > fl) p = m;
      }
      m /= p;
      fac_.push_back(p); fac_.push_back(m);
    }
    scratch_.resize(std::max(8, p_max()));
  }
  void run(const cf* in, cf* out) const { work(out, in, 1, fac_.data()); }

 private:
  int n_; bool inv_;
  std::vector<cf> tw_;
  std::vector<int> fac_;
  mutable std::vector<cf> scratch_;
  int p_max() const { int m = 8; for (size_t i = 0; i < fac_.size(); i += 2) m = std::max(m, fac_[i]); return m; }

  void work(cf* Fout, const cf* f, size_t fstride, const int* factors) const {
    const int p = *factors++, m = *factors++;
    cf* Fend = Fout + (size_t)p * m;
    if (m == 1) { do { *Fout = *f; f += fstride; } while (++Fout != Fend); }
    else { do { work(Fout, f, fstride * p, factors); f += fstride; } while ((Fout += m) != Fend); }
    Fout = Fend - (size_t)p * m;
    switch (p) {
      case 2: bf2(Fout, fstride, m); break;
      case 3: bf3(Fout, fstride, m); break;
      case 4: bf4(Fout, fstride, m); break;
      case 5: bf5(Fout, fstride, m); break;
      default: bfg(Fout, fstride, p, m);
    }
  }
  void bf2(cf* F, size_t fs, int m) const {
    cf* F2 = F + m;
    for (int k = 0; k < m; k++) {
      cf t = F2[k] * tw_[k * fs];
      F2[k] = F[k] - t;
      F[k] = F[k] + t;
    }
  }
  void bf4(cf* F, size_t fs, int m) const {
    for (int k = 0; k < m; k++) {
      cf s0 = F[k + m] * tw_[k * fs], s1 = F[k + 2 * m] * tw_[2 * k * fs], s2 = F[k + 3 * m] * tw_[3 * k * fs];
      cf s5 = F[k] - s1;
      F[k] = F[k] + s1;
      cf s3 = s0 + s2, s4 = s0 - s2;
      F[k + 2 * m] = F[k] - s3;
      F[k] = F[k] + s3;
      if (inv_) { F[k + m] = {s5.r - s4.i, s5.i + s4.r}; F[k + 3 * m] = {s5.r + s4.i, s5.i - s4.r}; }
      else { F[k + m] = {s5.r + s4.i, s5.i - s4.r}; F[k + 3 * m] = {s5.r - s4.i, s5.i + s4.r}; }
    }
  }
  void bf3(cf* F, size_t fs, int m) const {
    const size_t m2 = 2 * (size_t)m;
    const cf epi3 = tw_[fs * m];
    for (int k = 0; k < m; k++) {
      cf s1 = F[k + m] * tw_[k * fs], s2 = F[k + m2] * tw_[2 * k * fs];
      cf s3 = s1 + s2, s0 = {s1.r - s2.r, s1.i - s2.i};
      F[k + m] = {F[k].r - 0.5f * s3.r, F[k].i - 0.5f * s3.i};
      s0.r *= epi3.i; s0.i *= epi3.i;
      F[k] = F[k] + s3;
      F[k + m2] = {F[k + m].r + s0.i, F[k + m].i - s0.r};
      F[k + m] = {F[k + m].r - s0.i, F[k + m].i + s0.r};
    }
  }
  void bf5(cf* F, size_t fs, int m) const {
    const cf ya = tw_[fs * m], yb = tw_[fs * 2 * m];
    for (int u = 0; u < m; u++) {
      cf s0 = F[u];
      cf s1 = F[u + m] * tw_[u * fs], s2 = F[u + 2 * m] * tw_[2 * u * fs], s3 = F[u + 3 * m] * tw_[3 * u * fs], s4 = F[u + 4 * m] * tw_[4 * u * fs];
      cf s7 = s1 + s4, s10 = s1 - s4, s8 = s2 + s3, s9 = s2 - s3;
      F[u] = {s0.r + s7.r + s8.r, s0.i + s7.i + s8.i};
      cf s5 = {s0.r + s7.r * ya.r + s8.r * yb.r, s0.i + s7.i * ya.r + s8.i * yb.r};
      cf s6 = {s10.i * ya.i + s9.i * yb.i, -s10.r * ya.i - s9.r * yb.i};
      F[u + m] = s5 - s6; F[u + 4 * m] = s5 + s6;
      cf s11 = {s0.r + s7.r * yb.r + s8.r * ya.r, s0.i + s7.i * yb.r + s8.i * ya.r};
      cf s12 = {-s10.i * yb.i + s9.i * ya.i, s10.r * yb.i - s9.r * ya.i};
      F[u + 2 * m] = s11 + s12; F[u + 3 * m] = s11 - s12;
    }
  }
  void bfg(cf* F, size_t fs, int p, int m) const {
    cf* sc = scratch_.data();
    for (int u = 0; u < m; u++) {
      int k = u;
      for (int q1 = 0; q1 < p; q1++) { sc[q1] = F[k]; k += m; }
      k = u;
      for (int q1 = 0; q1 < p; q1++) {
        size_t twi = 0;
        F[k] = sc[0];
        for (int q = 1; q < p; q++) {
          twi += fs * k;
          if (twi >= (size_t)n_) twi %= n_;
          F[k] = F[k] + sc[q] * tw_[twi];
        }
        k += m;
      }
    }
  }
};

int good_size(int n) {   // smallest 2^a 3^b 5^c >= n
  for (int m = std::max(n, 2);; m++) {
    int r = m;
    while (r % 2 == 0) r /= 2;
    while (r % 3 == 0) r /= 3;
    while (r % 5 == 0) r /= 5;
    if (r == 1) return m;
  }
}

struct FFT2 {
  int nx, ny;
  FFT1 fx, fy;
  FFT2(int nx_, int ny_, bool inv) : nx(nx_), ny(ny_), fx(nx_, inv), fy(ny_, inv) {}
  void run(cf* d) const {
    std::vector<cf> t(std::max(nx, ny));
    for (int y = 0; y < ny; y++) { fx.run(d + (size_t)y * nx, t.data()); std::copy(t.begin(), t.begin() + nx, d + (size_t)y * nx); }
    std::vector<cf> col(ny), cout_(ny);
    for (int x = 0; x < nx; x++) {
      for (int y = 0; y < ny; y++) col[y] = d[(size_t)y * nx + x];
      fy.run(col.data(), cout_.data());
      for (int y = 0; y < ny; y++) d[(size_t)y * nx + x] = cout_[y];
    }
  }
};

// ------------------------------------------------------------------------------------------------ NCC maps
struct Integral {
  int w = 0, h = 0;
  std::vector<double> s1, s2;   // (w+1)(h+1)
  void build(const float* I, int w_, int h_) {
    w = w_; h = h_;
    s1.assign((size_t)(w + 1) * (h + 1), 0.0);
    s2.assign((size_t)(w + 1) * (h + 1), 0.0);
    for (int y = 0; y < h; y++) {
      double r1 = 0, r2 = 0;
      const float* r = I + (size_t)y * w;
      double* a1 = &s1[(size_t)(y + 1) * (w + 1)];
      double* a2 = &s2[(size_t)(y + 1) * (w + 1)];
      const double* p1 = &s1[(size_t)y * (w + 1)];
      const double* p2 = &s2[(size_t)y * (w + 1)];
      for (int x = 0; x < w; x++) {
        r1 += r[x]; r2 += (double)r[x] * r[x];
        a1[x + 1] = p1[x + 1] + r1;
        a2[x + 1] = p2[x + 1] + r2;
      }
    }
  }
  // sum over [x, x+n) x [y, y+n)
  inline void win(int x, int y, int n, double& a, double& b) const {
    const size_t W = w + 1;
    a = s1[(size_t)(y + n) * W + x + n] - s1[(size_t)y * W + x + n] - s1[(size_t)(y + n) * W + x] + s1[(size_t)y * W + x];
    b = s2[(size_t)(y + n) * W + x + n] - s2[(size_t)y * W + x + n] - s2[(size_t)(y + n) * W + x] + s2[(size_t)y * W + x];
  }
};

// numerators by shifted accumulation: out[y][x] = sum_{v,u} T0[v][u] * I[iy0+y+v][ix0+x+u]
void corr_direct(const float* I, int iw, const float* T0, int side, int ix0, int iy0, int nx, int ny, float* out) {
  std::fill(out, out + (size_t)nx * ny, 0.f);
  for (int y = 0; y < ny; y++) {
    float* o = out + (size_t)y * nx;
    for (int v = 0; v < side; v++) {
      const float* Ir = I + (size_t)(iy0 + y + v) * iw + ix0;
      const float* Tr = T0 + (size_t)v * side;
      for (int u = 0; u < side; u++) {
        const float t = Tr[u];
        const float* ir = Ir + u;
        for (int x = 0; x < nx; x++) o[x] += t * ir[x];
      }
    }
  }
}

// numerators by FFT over the whole image: out[y][x] = sum T0[v][u] I[y+v][x+u], x < iw-side+1, y < ih-side+1 (valid positions only)
void corr_fft(const float* I, int iw, int ih, const float* T0, int side, float* out) {
  const int NX = good_size(iw), NY = good_size(ih);
  std::vector<cf> buf((size_t)NX * NY, cf{0, 0});
  for (int y = 0; y < ih; y++) {
    cf* r = &buf[(size_t)y * NX];
    const float* s = I + (size_t)y * iw;
    for (int x = 0; x < iw; x++) r[x].r = s[x];
  }
  for (int y = 0; y < side; y++) {
    cf* r = &buf[(size_t)y * NX];
    const float* s = T0 + (size_t)y * side;
    for (int x = 0; x < side; x++) r[x].i = s[x];
  }
  FFT2 F(NX, NY, false);
  F.run(buf.data());
  for (int ky = 0; ky < NY; ky++) {
    const int my = (NY - ky) % NY;
    for (int kx = 0; kx < NX; kx++) {
      const int mx = (NX - kx) % NX;
      const size_t a = (size_t)ky * NX + kx, b = (size_t)my * NX + mx;
      if (b < a) continue;
      const cf X = buf[a], Xm = buf[b];
      // A = (X + conj(Xm))/2, B = (X - conj(Xm))/(2i)
      cf A = {0.5f * (X.r + Xm.r), 0.5f * (X.i - Xm.i)};
      cf d = {0.5f * (X.r - Xm.r), 0.5f * (X.i + Xm.i)};
      cf B = {d.i, -d.r};
      cf P = {A.r * B.r + A.i * B.i, A.i * B.r - A.r * B.i};   // A * conj(B)
      buf[a] = P;
      buf[b] = {P.r, -P.i};
    }
  }
  FFT2 Fi(NX, NY, true);
  Fi.run(buf.data());
  const float sc = 1.f / ((float)NX * NY);
  const int nx = iw - side + 1, ny = ih - side + 1;
  for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) out[(size_t)y * nx + x] = buf[(size_t)y * NX + x].r * sc;
}

void normalize_zero_mean(float* T, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; i++) s += T[i];
  const float m = (float)(s / n);
  double q = 0;
  for (size_t i = 0; i < n; i++) { T[i] -= m; q += (double)T[i] * T[i]; }
  const float inv = (float)(1.0 / std::sqrt(q + 1e-18));
  for (size_t i = 0; i < n; i++) T[i] *= inv;
}

// divide numerators by sqrt(sum (I-mean)^2) over the window
void ncc_denominator(const Integral& ig, int side, int ix0, int iy0, int nx, int ny, float* R) {
  const double N = (double)side * side;
  for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) {
      double a, b;
      ig.win(ix0 + x, iy0 + y, side, a, b);
      double var = b - a * a / N;
      float& r = R[(size_t)y * nx + x];
      r = var > 1e-8 ? (float)(r / std::sqrt(var)) : 0.f;
    }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ detector
struct MarkDetector::Impl {
  int side = 0, off_x = 0, off_y = 0;
  double page[2] = {0, 0};
  double blur_px = 0;
  std::vector<float> Tb;      // blurred padded template, side x side (ink = 1)
  std::vector<float> iso0;    // radial-profile template, blurred, zero mean, unit norm
  // polar rotation/scale search tables (angle spectra of the template at stretches k = K0 + DK i)
  static constexpr int NA = 180, NK = 25;
  static constexpr double K0 = 0.85, DK = 0.0125;
  int NJ = 0;
  double dr = 1.8, sw = 0;
  std::vector<double> cs, sn, wj;
  std::vector<cf> tspec;      // [NK][NJ][NA]: conj(FFT(w_j * T_k[j][.])) / norm_k
};

MarkDetector::MarkDetector(const Asset& A, double blur_mm) : p_(new Impl) {
  Impl& P = *p_;
  P.page[0] = A.page[0]; P.page[1] = A.page[1];
  P.blur_px = blur_mm * A_PPM;
  const int f = (int)std::lround(A.ref_ppm / A_PPM);   // 8
  const int tw = (A.rw + f - 1) / f, th = (A.rh + f - 1) / f;
  std::vector<float> T((size_t)tw * th, 0.f);
  for (int y = 0; y < th; y++)
    for (int x = 0; x < tw; x++) {
      double s = 0; int n = 0;
      for (int dy = 0; dy < f; dy++)
        for (int dx = 0; dx < f; dx++) {
          int xx = x * f + dx, yy = y * f + dy;
          if (xx < A.rw && yy < A.rh) { s += A.dref[(size_t)yy * A.rw + xx]; n++; }
        }
      T[(size_t)y * tw + x] = (float)(s / (4.0 * std::max(n, 1)));
    }
  const int side = std::max(tw, th) | 1;
  P.side = side; P.off_x = (side - tw) / 2; P.off_y = (side - th) / 2;
  std::vector<float> Tp((size_t)side * side, 0.f);
  for (int y = 0; y < th; y++)
    for (int x = 0; x < tw; x++) Tp[(size_t)(y + P.off_y) * side + x + P.off_x] = T[(size_t)y * tw + x];
  // rotation-averaged profile (bins of 1 px)
  const double c = (side - 1) / 2.0;
  const int nb = side / 2 + 1;
  std::vector<double> ps(nb, 0.0), pn(nb, 0.0);
  std::vector<int> rb((size_t)side * side);
  std::vector<uint8_t> inside((size_t)side * side);
  for (int y = 0; y < side; y++)
    for (int x = 0; x < side; x++) {
      double r = std::hypot(y - c, x - c);
      int b = std::min((int)r, side / 2);
      rb[(size_t)y * side + x] = b;
      inside[(size_t)y * side + x] = r <= side / 2.0;
      if (r <= side / 2.0) { ps[b] += Tp[(size_t)y * side + x]; pn[b] += 1; }
    }
  std::vector<float> prof(nb);
  for (int b = 0; b < nb; b++) prof[b] = (float)(ps[b] / std::max(pn[b], 1.0));
  P.iso0.resize((size_t)side * side);
  for (size_t i = 0; i < P.iso0.size(); i++) P.iso0[i] = inside[i] ? prof[rb[i]] : prof[nb - 1];
  gauss_blur(P.iso0, side, side, P.blur_px);
  normalize_zero_mean(P.iso0.data(), P.iso0.size());
  P.Tb = Tp;
  gauss_blur(P.Tb, side, side, P.blur_px);
  // polar tables
  const int NA = Impl::NA, NK = Impl::NK;
  P.NJ = std::max(8, (int)((side / 2.0 - 3.0) / P.dr));
  const int NJ = P.NJ;
  P.cs.resize(NA); P.sn.resize(NA); P.wj.resize(NJ);
  for (int i = 0; i < NA; i++) { P.cs[i] = std::cos(2 * PI * i / NA); P.sn[i] = std::sin(2 * PI * i / NA); }
  P.sw = 0;
  for (int j = 0; j < NJ; j++) { P.wj[j] = (j + 1) * P.dr; P.sw += P.wj[j] * NA; }
  P.tspec.assign((size_t)NK * NJ * NA, cf{0, 0});
  FFT1 fwd(NA, false);
  std::vector<float> Tw((size_t)NJ * NA);
  std::vector<cf> row(NA), spec(NA);
  const double ctc = (side - 1) / 2.0;
  for (int ki = 0; ki < NK; ki++) {
    const double k = Impl::K0 + Impl::DK * ki;
    double st = 0;
    for (int j = 0; j < NJ; j++) {
      const double r = P.wj[j] / k;
      for (int i = 0; i < NA; i++) {
        const float v = bilerp_z(P.Tb.data(), side, side, (float)(ctc + r * P.cs[i]), (float)(ctc + r * P.sn[i]));
        Tw[(size_t)j * NA + i] = v; st += P.wj[j] * v;
      }
    }
    const float mt = (float)(st / P.sw);
    double nt2 = 0;
    for (int j = 0; j < NJ; j++) for (int i = 0; i < NA; i++) { float& v = Tw[(size_t)j * NA + i]; v -= mt; nt2 += P.wj[j] * v * v; }
    const float sc = (float)(1.0 / std::sqrt(nt2 + 1e-18));
    for (int j = 0; j < NJ; j++) {
      for (int i = 0; i < NA; i++) row[i] = {Tw[(size_t)j * NA + i] * (float)P.wj[j] * sc, 0.f};
      fwd.run(row.data(), spec.data());
      cf* dst = &P.tspec[((size_t)ki * NJ + j) * NA];
      for (int f = 0; f < NA; f++) dst[f] = {spec[f].r, -spec[f].i};
    }
  }
}

MarkDetector::~MarkDetector() = default;

namespace {

struct Warp { double a, b, c, d, e, f, g = 0, h = 0; };   // Ds_idx = proj([a b c; d e f; g h 1] (T_idx - ct))   (+ ct-relative translation in c, f)

// Gauss-Newton (Levenberg-Marquardt damped) on the zero-mean NCC (ECC objective) between the template Tb (masked disc) and the frame warped by W.
// mode: 4 = similarity (s, theta, tx, ty), 6 = affine, 8 = homography. Returns the masked rho at the optimum, W updated.
double ecc_refine(const float* Ds, int w, int h, const float* Tb, int side, Warp& W, int mode, int iters) {
  const double ct = (side - 1) / 2.0;
  const double rmax = side / 2.0 - 1.0;
  std::vector<float> px, py, vhat;
  for (int y = 0; y < side; y++)
    for (int x = 0; x < side; x++) {
      double dx = x - ct, dy = y - ct;
      if (dx * dx + dy * dy <= rmax * rmax) { px.push_back((float)dx); py.push_back((float)dy); vhat.push_back(Tb[(size_t)y * side + x]); }
    }
  const int n = (int)px.size();
  normalize_zero_mean(vhat.data(), n);
  std::vector<float> Iw(n), gx(n), gy(n), wz(n), Xw(n), Yw(n), J((size_t)n * 8);
  const int np = mode;
  auto params_from = [&](const Warp& Wp, double* p) {
    if (mode == 8) { p[0] = Wp.a; p[1] = Wp.b; p[2] = Wp.c; p[3] = Wp.d; p[4] = Wp.e; p[5] = Wp.f; p[6] = Wp.g; p[7] = Wp.h; }
    else if (mode == 6) { p[0] = Wp.a; p[1] = Wp.b; p[2] = Wp.c; p[3] = Wp.d; p[4] = Wp.e; p[5] = Wp.f; }
    else { p[0] = std::sqrt(std::abs(Wp.a * Wp.e - Wp.b * Wp.d)); p[1] = std::atan2(0.5 * (Wp.d - Wp.b), 0.5 * (Wp.a + Wp.e)); p[2] = Wp.c; p[3] = Wp.f; }
  };
  auto warp_from = [&](const double* p) {
    Warp Wp{};
    if (mode == 8) Wp = {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]};
    else if (mode == 6) Wp = {p[0], p[1], p[2], p[3], p[4], p[5], 0, 0};
    else { double c = std::cos(p[1]) * p[0], s = std::sin(p[1]) * p[0]; Wp = {c, -s, p[2], s, c, p[3], 0, 0}; }
    return Wp;
  };
  // evaluates rho; fills Iw (and the gradient / warp coordinates if grad)
  auto eval = [&](const Warp& Wp, bool grad, double& nu) {
    bool ok = true;
    for (int k = 0; k < n; k++) {
      const double z = Wp.g * px[k] + Wp.h * py[k] + 1.0;
      if (z < 0.2) ok = false;
      const float x = (float)((Wp.a * px[k] + Wp.b * py[k] + Wp.c) / z), y = (float)((Wp.d * px[k] + Wp.e * py[k] + Wp.f) / z);
      Iw[k] = bilerp_c(Ds, w, h, x, y);
      if (grad) {
        gx[k] = 0.5f * (bilerp_c(Ds, w, h, x + 1, y) - bilerp_c(Ds, w, h, x - 1, y));
        gy[k] = 0.5f * (bilerp_c(Ds, w, h, x, y + 1) - bilerp_c(Ds, w, h, x, y - 1));
        wz[k] = (float)z; Xw[k] = x; Yw[k] = y;
      }
    }
    if (!ok) { nu = 1; return -2.0; }
    double m = 0;
    for (int k = 0; k < n; k++) m += Iw[k];
    m /= n;
    double q = 0, dp = 0;
    for (int k = 0; k < n; k++) { double u = Iw[k] - m; q += u * u; dp += u * vhat[k]; }
    nu = std::sqrt(q + 1e-18);
    return dp / nu;
  };
  double p[8];
  params_from(W, p);
  double nu;
  double rho = eval(warp_from(p), false, nu);
  double lam = 1e-3;
  for (int it = 0; it < iters; it++) {
    Warp Wc = warp_from(p);
    rho = eval(Wc, true, nu);
    // jacobian columns
    double s_ = mode == 4 ? p[0] : 0, th = mode == 4 ? p[1] : 0, ca = std::cos(th), sa = std::sin(th);
    for (int k = 0; k < n; k++) {
      float* j = &J[(size_t)k * 8];
      const float x = px[k], y = py[k];
      if (mode == 4) {
        j[0] = (float)(gx[k] * (ca * x - sa * y) + gy[k] * (sa * x + ca * y));
        j[1] = (float)(s_ * (gx[k] * (-sa * x - ca * y) + gy[k] * (ca * x - sa * y)));
        j[2] = gx[k]; j[3] = gy[k];
      } else {
        const float iz = 1.f / wz[k];
        const float gxz = gx[k] * iz, gyz = gy[k] * iz;
        j[0] = gxz * x; j[1] = gxz * y; j[2] = gxz; j[3] = gyz * x; j[4] = gyz * y; j[5] = gyz;
        if (mode == 8) { const float t = -(gxz * Xw[k] + gyz * Yw[k]); j[6] = t * x; j[7] = t * y; }
      }
    }
    // centre columns, project out u-hat, scale by 1/nu
    double m = 0;
    for (int k = 0; k < n; k++) m += Iw[k];
    m /= n;
    std::vector<float> uh(n);
    for (int k = 0; k < n; k++) uh[k] = (float)((Iw[k] - m) / nu);
    double jm[8] = {0}, ju[8] = {0};
    for (int k = 0; k < n; k++) for (int a = 0; a < np; a++) jm[a] += J[(size_t)k * 8 + a];
    for (int a = 0; a < np; a++) jm[a] /= n;
    for (int k = 0; k < n; k++) for (int a = 0; a < np; a++) ju[a] += uh[k] * (J[(size_t)k * 8 + a] - jm[a]);
    double Hm[64] = {0}, g[8] = {0};
    for (int k = 0; k < n; k++) {
      double jn[8];
      for (int a = 0; a < np; a++) jn[a] = (J[(size_t)k * 8 + a] - jm[a] - uh[k] * ju[a]) / nu;
      for (int a = 0; a < np; a++) {
        g[a] += jn[a] * vhat[k];
        for (int b = a; b < np; b++) Hm[a * 8 + b] += jn[a] * jn[b];
      }
    }
    for (int a = 0; a < np; a++) for (int b = 0; b < a; b++) Hm[a * 8 + b] = Hm[b * 8 + a];
    bool improved = false;
    for (int tr = 0; tr < 6 && !improved; tr++) {
      double A_[64], bb[8];
      for (int a = 0; a < np; a++) { bb[a] = g[a]; for (int b = 0; b < np; b++) A_[a * np + b] = Hm[a * 8 + b]; }
      for (int a = 0; a < np; a++) A_[a * np + a] += lam * (Hm[a * 8 + a] + 1e-9);
      double x[8];
      if (!chol_solve(A_, bb, np, 0.0, x)) { lam *= 10; continue; }
      double pn_[8];
      for (int a = 0; a < np; a++) pn_[a] = p[a] + x[a];
      double nu2;
      double rho2 = eval(warp_from(pn_), false, nu2);
      if (rho2 > rho) {
        double lin = 0, trn = 0;
        for (int a = 0; a < np; a++) {
          const bool is_t = mode == 4 ? (a >= 2) : (a == 2 || a == 5);
          const bool is_p = mode == 8 && a >= 6;
          const double v = is_p ? std::abs(x[a]) * 50 : std::abs(x[a]);
          if (is_t) trn = std::max(trn, v); else lin = std::max(lin, v);
        }
        for (int a = 0; a < np; a++) p[a] = pn_[a];
        const double gain = rho2 - rho;
        rho = rho2; improved = true; lam = std::max(lam / 3, 1e-7);
        if (gain < 2e-6 && trn < 0.01 && lin < 1e-4) it = iters;
      } else lam *= 4;
    }
    if (!improved) break;
  }
  W = warp_from(p);
  double nu3;
  rho = eval(W, false, nu3);
  return rho;
}

// full-square zero-mean NCC of Tb against the frame warped by W (the reported score; includes the corners of the template square)
double zncc_square(const float* Ds, int w, int h, const float* Tb, int side, const Warp& W) {
  const double ct = (side - 1) / 2.0;
  std::vector<float> t((size_t)side * side), v((size_t)side * side);
  for (int y = 0; y < side; y++)
    for (int x = 0; x < side; x++) {
      double dx = x - ct, dy = y - ct;
      const double z = W.g * dx + W.h * dy + 1.0;
      v[(size_t)y * side + x] = bilerp_c(Ds, w, h, (float)((W.a * dx + W.b * dy + W.c) / z), (float)((W.d * dx + W.e * dy + W.f) / z));
    }
  std::copy(Tb, Tb + (size_t)side * side, t.begin());
  normalize_zero_mean(t.data(), t.size());
  normalize_zero_mean(v.data(), v.size());
  return dot_f(t.data(), v.data(), (int)t.size());
}

void orient_template(const float* T, int side, double th, double k, float* out) {
  const double c = (side - 1) / 2.0, ca = std::cos(th), sa = std::sin(th);
  for (int y = 0; y < side; y++)
    for (int x = 0; x < side; x++) {
      const double dx = x - c, dy = y - c;
      out[(size_t)y * side + x] = bilerp_z(T, side, side, (float)(c + (ca * dx + sa * dy) / k), (float)(c + (-sa * dx + ca * dy) / k));
    }
}

}  // namespace

MarkResult MarkDetector::detect(const ImageView& im, const MarkParams& P) const {
  const Impl& M = *p_;
  Timer total, t;
  MarkResult R;
  const int side = M.side;
  const int dim = im.is_float ? 0 : 1;
  (void)dim;

  // --- scales and pyramid levels (uint8 box downsampling of the frame, darkness per level)
  std::vector<double> scales;
  for (double s = P.ppm_lo; s <= P.ppm_hi * 1.001; s *= P.step) scales.push_back(s);
  auto level_for = [&](double s) { int m = 1; while (s / (2.0 * m) >= A_PPM && m < 16) m *= 2; return m; };
  struct Level { int m = 0; std::vector<uint8_t> u8; int w = 0, h = 0; Dark dk; };
  std::map<int, Level> levels;
  auto get_level = [&](int m) -> Level& {
    auto it = levels.find(m);
    if (it != levels.end()) return it->second;
    Level L; L.m = m;
    if (m == 1) { darkness(im, L.dk); L.w = L.dk.w; L.h = L.dk.h; }
    else {
      const Level* prev = nullptr;
      if (m == 2) {
        // from the original view: convert floats to u8 on the fly
        const size_t stride = im.stride ? im.stride : im.w;
        if (!im.is_float) box2_u8((const uint8_t*)im.data, im.w, im.h, stride, L.u8, L.w, L.h);
        else {
          std::vector<uint8_t> tmp((size_t)im.w * im.h);
          const float* p = (const float*)im.data;
          for (int y = 0; y < im.h; y++) for (int x = 0; x < im.w; x++) tmp[(size_t)y * im.w + x] = (uint8_t)std::min(255.f, std::max(0.f, std::nearbyint(p[y * stride + x])));
          box2_u8(tmp.data(), im.w, im.h, im.w, L.u8, L.w, L.h);
        }
      } else {
        prev = &levels.at(m / 2);
        box2_u8(prev->u8.data(), prev->w, prev->h, prev->w, L.u8, L.w, L.h);
      }
      ImageView v; v.data = L.u8.data(); v.w = L.w; v.h = L.h; v.stride = L.w; v.is_float = false;
      darkness(v, L.dk);
    }
    return levels.emplace(m, std::move(L)).first->second;
  };
  // make sure the chain of levels exists in order (level m needs level m/2's u8)
  auto ensure = [&](int m) { for (int q = 2; q <= m; q *= 2) get_level(q); return &get_level(m); };

  const double ct = (side - 1) / 2.0;
  struct Best { double mx = -2; double s = 0; double lx = 0, ly = 0; int w = 0, h = 0; std::vector<float> Ds; double fx = 0, fy = 0; Integral ig; } best;
  R.t.prep += t.ms();
  for (double s : scales) {
    t.reset();
    const int m = level_for(s);
    Level* L = ensure(m);
    const double g = A_PPM / (s / m);
    const int ow = (int)std::lround(L->w * g), oh = (int)std::lround(L->h * g);
    R.t.prep += t.ms();
    if (std::min(ow, oh) < side + 4) continue;
    t.reset();
    std::vector<float> Ds;
    area_resize(L->dk.D.data(), L->w, L->h, ow, oh, Ds);
    gauss_blur(Ds, ow, oh, M.blur_px);
    double t_rs = t.ms();
    Integral ig;
    ig.build(Ds.data(), ow, oh);
    double t_ig = t.ms();
    const int nx = ow - side + 1, ny = oh - side + 1;
    std::vector<float> Rm((size_t)nx * ny);
    const int NX = good_size(ow), NY = good_size(oh);
    const double c_direct = (double)nx * ny * side * side;
    const double c_fft = 14.0 * NX * NY * std::log2((double)NX * NY);
    if (c_direct < c_fft) corr_direct(Ds.data(), ow, M.iso0.data(), side, 0, 0, nx, ny, Rm.data());
    else { corr_fft(Ds.data(), ow, oh, M.iso0.data(), side, Rm.data()); R.t.n_fft++; }
    double t_cr = t.ms();
    ncc_denominator(ig, side, 0, 0, nx, ny, Rm.data());
    if (getenv("STEGO_DEBUG")) fprintf(stderr, "  scale %.1f Ds %dx%d resize+blur %.1f integral %.1f corr(%s) %.1f denom %.1f\n", s, ow, oh, t_rs, t_ig - t_rs, c_direct < c_fft ? "direct" : "fft", t_cr - t_ig, t.ms() - t_cr);
    int bi = 0; float bv = -2;
    if (!P.exclude.empty()) {   // ignore peaks whose centre (view px) lies inside an excluded circle
      for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
          const double ux = (x + ct + 0.5) / (((double)ow) / ((double)L->w * m)), uy = (y + ct + 0.5) / (((double)oh) / ((double)L->h * m));
          for (const auto& e : P.exclude)
            if ((ux - e[0]) * (ux - e[0]) + (uy - e[1]) * (uy - e[1]) < e[2] * e[2]) { Rm[(size_t)y * nx + x] = -2.f; break; }
        }
    }
    for (size_t i = 0; i < Rm.size(); i++) if (Rm[i] > bv) { bv = Rm[i]; bi = (int)i; }
    R.t.n_scales++;
    R.t.stage_a += t.ms();
    if (bv > best.mx) {
      const int bx = bi % nx, by = bi / nx;
      // parabolic sub-pixel refinement
      auto para = [](float a, float b, float c) { double d = a - 2.0 * b + c; return std::abs(d) < 1e-9 ? 0.0 : std::clamp(0.5 * (a - c) / d, -0.5, 0.5); };
      double sx = 0, sy = 0;
      if (bx > 0 && bx < nx - 1) sx = para(Rm[(size_t)by * nx + bx - 1], bv, Rm[(size_t)by * nx + bx + 1]);
      if (by > 0 && by < ny - 1) sy = para(Rm[(size_t)(by - 1) * nx + bx], bv, Rm[(size_t)(by + 1) * nx + bx]);
      best.mx = bv; best.s = s; best.lx = bx + sx; best.ly = by + sy; best.w = ow; best.h = oh;
      best.Ds = std::move(Ds); best.ig = std::move(ig);
      best.fx = (double)ow / ((double)L->w * m); best.fy = (double)oh / ((double)L->h * m);
    }
  }
  if (best.mx < -1.5) { R.t.total = total.ms(); return R; }
  R.iso = best.mx;
  R.scale_ppm = best.s;
  const float* Ds = best.Ds.data();
  const int dw = best.w, dh = best.h;

  // --- stage B: polar rotation/scale search (angular FFT cross-correlation) over a grid of centres around the stage-A estimate
  t.reset();
  double c0x = best.lx + ct, c0y = best.ly + ct;
  double theta = 0, kk = 1.0;
  struct PHyp { double th, k, v; };
  std::vector<PHyp> hyps_out;
  {
    const int NA = Impl::NA, NK = Impl::NK, NJ = M.NJ;
    FFT1 fwd(NA, false), inv(NA, true);
    std::vector<cf> Pf((size_t)NJ * NA), row(NA), acc(NA), cc(NA);
    std::vector<float> Pr((size_t)NJ * NA);
    double inv_np = 1;
    auto sample_P = [&](double cx, double cy) {
      double sp = 0;
      for (int j = 0; j < NJ; j++) {
        float* rw = &Pr[(size_t)j * NA];
        const double r = M.wj[j];
        for (int i = 0; i < NA; i++) { rw[i] = bilerp_c(Ds, dw, dh, (float)(cx + r * M.cs[i]), (float)(cy + r * M.sn[i])); sp += r * rw[i]; }
      }
      const float mp = (float)(sp / M.sw);
      double np2 = 0;
      for (int j = 0; j < NJ; j++) {
        float* rw = &Pr[(size_t)j * NA];
        for (int i = 0; i < NA; i++) { rw[i] -= mp; np2 += M.wj[j] * rw[i] * rw[i]; }
      }
      inv_np = 1.0 / std::sqrt(np2 + 1e-18);
      for (int j = 0; j < NJ; j++) {
        const float* rw = &Pr[(size_t)j * NA];
        for (int i = 0; i < NA; i++) row[i] = {rw[i], 0.f};
        fwd.run(row.data(), &Pf[(size_t)j * NA]);
      }
    };
    // correlation scores over all shifts for stretch index ki -> cc (real part); returns best shift
    auto corr_k = [&](int ki, double& bv, std::vector<double>* keep) {
      std::fill(acc.begin(), acc.end(), cf{0, 0});
      for (int j = 0; j < NJ; j++) {
        const cf* T = &M.tspec[((size_t)ki * NJ + j) * NA];
        const cf* Pj = &Pf[(size_t)j * NA];
        for (int f = 0; f < NA; f++) { acc[f].r += T[f].r * Pj[f].r - T[f].i * Pj[f].i; acc[f].i += T[f].r * Pj[f].i + T[f].i * Pj[f].r; }
      }
      inv.run(acc.data(), cc.data());
      int bi = 0; bv = -1e30;
      for (int s = 0; s < NA; s++) { const double v = cc[s].r * inv_np / NA; if (v > bv) { bv = v; bi = s; } if (keep) (*keep)[s] = v; }
      return bi;
    };
    double cx = c0x, cy = c0y;
    double bsc = -1e30; int bks = 0;
    // coarse: centres +-8 px step 2, stretches every 4th index
    for (double dy = -8; dy <= 8.01; dy += 2)
      for (double dx = -8; dx <= 8.01; dx += 2) {
        sample_P(c0x + dx, c0y + dy);
        for (int ki = 0; ki < NK; ki += 4) {
          double v; corr_k(ki, v, nullptr);
          if (v > bsc) { bsc = v; cx = c0x + dx; cy = c0y + dy; bks = ki; }
        }
      }
    // medium: centres +-2 px step 1 around the best, stretch indices bks-3..bks+3 step 1.5 ~ all within +-0.05
    const double ccx = cx, ccy = cy;
    for (double dy = -2; dy <= 2.01; dy += 1)
      for (double dx = -2; dx <= 2.01; dx += 1) {
        sample_P(ccx + dx, ccy + dy);
        for (int ki = std::max(0, bks - 4); ki <= std::min(NK - 1, bks + 4); ki += 2) {
          double v; corr_k(ki, v, nullptr);
          if (v > bsc) { bsc = v; cx = ccx + dx; cy = ccy + dy; bks = ki; }
        }
      }
    // fine: centres +-0.5 step 0.5 around the best, all stretches within +-0.04
    const double fcx = cx, fcy = cy;
    for (double dy = -0.5; dy <= 0.51; dy += 0.5)
      for (double dx = -0.5; dx <= 0.51; dx += 0.5) {
        sample_P(fcx + dx, fcy + dy);
        for (int ki = std::max(0, bks - 3); ki <= std::min(NK - 1, bks + 3); ki++) {
          double v; corr_k(ki, v, nullptr);
          if (v > bsc) { bsc = v; cx = fcx + dx; cy = fcy + dy; bks = ki; }
        }
      }
    // final: parabolic refinement of shift and stretch at the best centre; collect the best distinct rotation peaks as hypotheses
    sample_P(cx, cy);
    std::vector<double> ks(NK, -1e30);
    std::vector<std::vector<double>> allk(NK);
    auto para = [](double a, double b, double c) { double d = a - 2 * b + c; return std::abs(d) < 1e-12 ? 0.0 : std::clamp(0.5 * (a - c) / d, -0.5, 0.5); };
    std::vector<int> kbs_i(NK, 0);
    for (int ki = std::max(0, bks - 2); ki <= std::min(NK - 1, bks + 2); ki++) {
      allk[ki].assign(NA, 0.0);
      double v; kbs_i[ki] = corr_k(ki, v, &allk[ki]);
      ks[ki] = v;
    }
    struct Hyp { double th, k, v; };
    std::vector<Hyp> cand;
    for (int ki = std::max(0, bks - 1); ki <= std::min(NK - 1, bks + 1); ki++) {
      const auto& a = allk[ki];
      for (int sh = 0; sh < NA; sh++) {
        const double v = a[sh];
        if (v >= a[(sh + NA - 1) % NA] && v > a[(sh + 1) % NA]) {
          const double dsh = para(a[(sh + NA - 1) % NA], v, a[(sh + 1) % NA]);
          const double dk = (ki > std::max(0, bks - 2) && ki < std::min(NK - 1, bks + 2)) ? para(ks[ki - 1], ks[ki], ks[ki + 1]) : 0.0;
          cand.push_back({(sh + dsh) * 2 * PI / NA, Impl::K0 + Impl::DK * (ki + (ki == bks ? dk : 0.0)), v});
        }
      }
    }
    std::sort(cand.begin(), cand.end(), [](const Hyp& a, const Hyp& b) { return a.v > b.v; });
    for (const Hyp& h : cand) {
      bool dup = false;
      for (const PHyp& q : hyps_out) {
        double d = std::fmod(std::abs(h.th - q.th), 2 * PI);
        d = std::min(d, 2 * PI - d);
        if (d < 8.0 * PI / 180) dup = true;
      }
      if (!dup) hyps_out.push_back({h.th, h.k, h.v});
      if ((int)hyps_out.size() >= P.n_hyp) break;
    }
    if (hyps_out.empty()) hyps_out.push_back({0.0, 1.0, 0.0});
    theta = hyps_out[0].th; kk = hyps_out[0].k;
    c0x = cx; c0y = cy;
  }
  R.t.polar = t.ms();
  if (getenv("STEGO_DEBUG")) fprintf(stderr, "mark: scale %.2f iso %.3f c=(%.1f,%.1f) polar theta=%.1f k=%.3f\n", best.s, best.mx, c0x, c0y, theta * 180 / PI, kk);

  // --- stage C: for each rotation hypothesis: centre search with the oriented template + a few GN similarity iterations; best -> full GN
  t.reset();
  auto start_pose = [&](double th, double k, double* ncc_out) {
    std::vector<float> To((size_t)side * side);
    orient_template(M.Tb.data(), side, th, k, To.data());
    normalize_zero_mean(To.data(), To.size());
    const int rad = P.pad;
    const int cx0 = (int)std::lround(c0x - ct), cy0 = (int)std::lround(c0y - ct);
    const int x0 = std::max(0, cx0 - rad), y0 = std::max(0, cy0 - rad);
    const int x1 = std::min(dw - side, cx0 + rad), y1 = std::min(dh - side, cy0 + rad);
    int bx = std::clamp(cx0, 0, std::max(0, dw - side)), by = std::clamp(cy0, 0, std::max(0, dh - side));
    if (x1 >= x0 && y1 >= y0) {
      const int nx = x1 - x0 + 1, ny = y1 - y0 + 1;
      std::vector<float> Rm((size_t)nx * ny);
      corr_direct(Ds, dw, To.data(), side, x0, y0, nx, ny, Rm.data());
      ncc_denominator(best.ig, side, x0, y0, nx, ny, Rm.data());
      float bv = -2;
      for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) if (Rm[(size_t)y * nx + x] > bv) { bv = Rm[(size_t)y * nx + x]; bx = x0 + x; by = y0 + y; }
      if (ncc_out) *ncc_out = bv;
    } else if (ncc_out) *ncc_out = -1;
    // Ds_idx = kR (T_idx - ct) + (bx + ct, by + ct)
    const double c = std::cos(th) * k, sn = std::sin(th) * k;
    return Warp{c, -sn, bx + ct, sn, c, by + ct};
  };
  Warp W{};
  {
    // screen all polar peaks by the oriented-template NCC (as the Python sweep ranks them), GN the best few
    struct Sc { double ncc; PHyp h; Warp W; };
    std::vector<Sc> scr;
    for (const PHyp& h : hyps_out) {
      double nccv;
      Warp Wh = start_pose(h.th, h.k, &nccv);
      scr.push_back({nccv, h, Wh});
    }
    std::sort(scr.begin(), scr.end(), [](const Sc& a, const Sc& b) { return a.ncc > b.ncc; });
    double bestrho = -2;
    const int ngn = std::min<int>(P.n_gn, (int)scr.size());
    for (int i = 0; i < ngn; i++) {
      Warp Wh = scr[i].W;
      const double rho = ecc_refine(Ds, dw, dh, M.Tb.data(), side, Wh, 4, ngn > 1 ? 10 : 40);
      if (getenv("STEGO_DEBUG")) fprintf(stderr, "mark: hyp theta=%.1f k=%.3f polar=%.3f ncc=%.3f -> rho %.3f\n", scr[i].h.th * 180 / PI, scr[i].h.k, scr[i].h.v, scr[i].ncc, rho);
      if (rho > bestrho) { bestrho = rho; W = Wh; }
    }
    if (ngn > 1) ecc_refine(Ds, dw, dh, M.Tb.data(), side, W, 4, 40);
  }
  double sc_sim = zncc_square(Ds, dw, dh, M.Tb.data(), side, W);
  R.sim_score = sc_sim;
  R.t.fine = t.ms();
  t.reset();
  double sc = sc_sim;
  if (P.affine) {
    Warp W2 = W;
    ecc_refine(Ds, dw, dh, M.Tb.data(), side, W2, 6, 60);
    double sc2 = zncc_square(Ds, dw, dh, M.Tb.data(), side, W2);
    R.aff_score = sc2;
    const double d1 = W.a * W.e - W.b * W.d, d2 = W2.a * W2.e - W2.b * W2.d;
    if (sc2 > sc && d1 > 0 && d2 > 0 && std::abs(d2 / d1 - 1) < 0.35) { W = W2; sc = sc2; }
    if (P.homography) {
      Warp W3 = W;
      ecc_refine(Ds, dw, dh, M.Tb.data(), side, W3, 8, 60);
      const double sc3 = zncc_square(Ds, dw, dh, M.Tb.data(), side, W3);
      R.hom_score = sc3;
      const double d3 = W3.a * W3.e - W3.b * W3.d, d0 = W.a * W.e - W.b * W.d;
      if (sc3 > sc && d3 > 0 && d0 > 0 && std::abs(d3 / d0 - 1) < 0.35 && std::abs(W3.g) * side < 0.9 && std::abs(W3.h) * side < 0.9) { W = W3; sc = sc3; }
    }
  }
  R.t.ecc = t.ms();

  // --- assemble H: page mm -> T_idx -> Ds_idx -> Ds_u -> full-view u
  // T_idx = A_PPM * mm - 0.5 + off ; Ds_idx = Lin (T_idx - ct) + t ; u = (Ds_idx + 0.5) / f
  double T3[9] = {A_PPM, 0, M.off_x - 0.5, 0, A_PPM, M.off_y - 0.5, 0, 0, 1};
  // Ds_idx = proj(Wc (T_idx - ct)) with Wc = [a b c; d e f; g h 1]  ->  W3 = Wc * Tr(-ct)
  double Wc[9] = {W.a, W.b, W.c, W.d, W.e, W.f, W.g, W.h, 1};
  double Trc[9] = {1, 0, -ct, 0, 1, -ct, 0, 0, 1};
  double W3[9];
  h_mul(Wc, Trc, W3);
  double S3[9] = {1 / best.fx, 0, 0.5 / best.fx, 0, 1 / best.fy, 0.5 / best.fy, 0, 0, 1};
  double tmp[9], H[9];
  h_mul(W3, T3, tmp);
  h_mul(S3, tmp, H);
  std::memcpy(R.H, H, sizeof H);
  R.found = true;
  R.score = sc;
  {
    const double cx_ = M.page[0] / 2, cy_ = M.page[1] / 2;
    const double w_ = H[6] * cx_ + H[7] * cy_ + H[8];
    const double px_ = (H[0] * cx_ + H[1] * cy_ + H[2]) / w_, py_ = (H[3] * cx_ + H[4] * cy_ + H[5]) / w_;
    R.centre[0] = px_; R.centre[1] = py_;
    // local Jacobian of the projective map at the page centre
    const double J00 = (H[0] - px_ * H[6]) / w_, J01 = (H[1] - px_ * H[7]) / w_, J10 = (H[3] - py_ * H[6]) / w_, J11 = (H[4] - py_ * H[7]) / w_;
    R.ppm = std::sqrt(std::abs(J00 * J11 - J01 * J10));
    R.rot_deg = std::fmod(std::atan2(J10, J00) * 180 / PI + 360.0, 360.0);
  }
  // rotation hypotheses (n-fold symmetric geometry): H * Rot_mm(2 pi k / n) about the symmetry centre
  const int nsym = std::max(1, P.sym);
  const double scx = P.sym_cx >= 0 ? P.sym_cx : M.page[0] / 2, scy = P.sym_cy >= 0 ? P.sym_cy : M.page[1] / 2;
  for (int k = 0; k < nsym; k++) {
    const double a = 2 * PI * k / nsym, ca = std::cos(a), sa = std::sin(a);
    double Rm[9] = {ca, -sa, scx - ca * scx + sa * scy, sa, ca, scy - sa * scx - ca * scy, 0, 0, 1};
    double Hk[9];
    h_mul(H, Rm, Hk);
    std::array<double, 9> arr;
    std::copy(Hk, Hk + 9, arr.begin());
    R.hyps.push_back(arr);
  }
  R.t.total = total.ms();
  return R;
}

double mark_fft_selftest(int nx, int ny) {
  std::vector<cf> a((size_t)nx * ny), ref((size_t)nx * ny);
  unsigned s = 12345;
  for (auto& v : a) { s = s * 1664525u + 1013904223u; v.r = (s >> 8) / 16777216.f - 0.5f; s = s * 1664525u + 1013904223u; v.i = (s >> 8) / 16777216.f - 0.5f; }
  for (int ky = 0; ky < ny; ky++)
    for (int kx = 0; kx < nx; kx++) {
      double sr = 0, si = 0;
      for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
          double ph = -2 * PI * ((double)kx * x / nx + (double)ky * y / ny);
          const cf v = a[(size_t)y * nx + x];
          sr += v.r * std::cos(ph) - v.i * std::sin(ph);
          si += v.r * std::sin(ph) + v.i * std::cos(ph);
        }
      ref[(size_t)ky * nx + kx] = {(float)sr, (float)si};
    }
  FFT2 F(nx, ny, false);
  F.run(a.data());
  double e = 0;
  for (size_t i = 0; i < a.size(); i++) e = std::max(e, (double)std::max(std::abs(a[i].r - ref[i].r), std::abs(a[i].i - ref[i].i)));
  return e;
}

}  // namespace stego
