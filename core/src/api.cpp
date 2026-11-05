#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "internal.hpp"

namespace stego {

Result decode(const Asset& A, const ImageView& im, const Params& P, const double* H_init) {
  Timer total, t;
  Dark dk;
  darkness(im, dk);
  double dark_ms = t.ms();
  double H[9];
  RegResult rr;
  if (H_init && !P.refine_init) {
    std::memcpy(H, H_init, sizeof H);
    rr.score = -1;
  } else {
    rr = register_dark(A, dk, P, H_init);
    std::memcpy(H, rr.H, sizeof H);
  }
  Timer td;
  Result R = decode_dark(A, dk, P, H, nullptr, total);
  if (getenv("STEGO_DEBUG")) fprintf(stderr, "decode_dark %.2f ms (measure %.2f fit %.2f tail %.2f golay %.2f)\n", td.ms(), R.t.measure, R.t.fit, R.t.tail, R.t.golay);
  R.t.darkness = dark_ms;
  R.t.coarse = rr.t.coarse;
  R.t.edge = rr.t.edge;
  R.reg_score = rr.score;
  R.reg_cand = rr.cand;
  R.t.total = total.ms();
  return R;
}

RegResult register_block(const Asset& A, const ImageView& im, const Params& P, const double* H_init) {
  Dark dk;
  darkness(im, dk);
  return register_dark(A, dk, P, H_init);
}

namespace {
inline float getpx(const ImageView& im, int x, int y) {
  const size_t st = im.stride ? im.stride : im.w;
  return im.is_float ? ((const float*)im.data)[y * st + x] : (float)((const uint8_t*)im.data)[y * st + x];
}
}  // namespace

Result decode_multi(const Asset& A, const ImageView* crops, int K, const Params& P) {
  Timer total, t;
  std::vector<std::vector<double>> Hs(K, std::vector<double>(9));
  double dark_ms = 0, coarse_ms = 0, edge_ms = 0;
  Dark dk0;
  for (int k = 0; k < K; k++) {
    Dark dk;
    Dark& d = k == 0 ? dk0 : dk;
    t.reset();
    darkness(crops[k], d);
    dark_ms += t.ms();
    RegResult rr;
    if (k > 0) {
      rr = register_dark(A, d, P, Hs[0].data());
      if (rr.score < 0.3) rr = register_dark(A, d, P, nullptr);   // lost: full search
    } else {
      rr = register_dark(A, d, P, nullptr);
    }
    coarse_ms += rr.t.coarse; edge_ms += rr.t.edge;
    std::memcpy(Hs[k].data(), rr.H, sizeof rr.H);
  }
  // average in crop-0 geometry
  const int w = crops[0].w, h = crops[0].h;
  std::vector<float> acc((size_t)w * h, 0.f), cnt((size_t)w * h, 0.f);
  t.reset();
  double H0i[9];
  h_inv(Hs[0].data(), H0i);
  for (int k = 0; k < K; k++) {
    double Q[9];
    h_mul(Hs[k].data(), H0i, Q);
    h_norm(Q);
    const ImageView& im = crops[k];
    for (int y = 0; y < h; y++) {
      float* a = &acc[(size_t)y * w];
      float* c = &cnt[(size_t)y * w];
      const double yy = y + 0.5;
      for (int x = 0; x < w; x++) {
        const double xx = x + 0.5;
        const double d = Q[6] * xx + Q[7] * yy + Q[8];
        const float u = (float)((Q[0] * xx + Q[1] * yy + Q[2]) / d - 0.5), v = (float)((Q[3] * xx + Q[4] * yy + Q[5]) / d - 0.5);
        if (u < -0.5f || v < -0.5f || u > im.w - 0.5f || v > im.h - 0.5f) continue;
        const float uc = std::min(std::max(u, 0.f), im.w - 1.001f), vc = std::min(std::max(v, 0.f), im.h - 1.001f);
        const int ix = (int)uc, iy = (int)vc;
        const float fx = uc - ix, fy = vc - iy;
        const float p00 = getpx(im, ix, iy), p01 = getpx(im, ix + 1, iy), p10 = getpx(im, ix, iy + 1), p11 = getpx(im, ix + 1, iy + 1);
        const float top = p00 + fx * (p01 - p00), bot = p10 + fx * (p11 - p10);
        a[x] += top + fy * (bot - top);
        c[x] += 1.f;
      }
    }
  }
  for (size_t i = 0; i < acc.size(); i++) acc[i] = cnt[i] > 0 ? acc[i] / cnt[i] : 255.f;
  double warp_ms = t.ms();
  ImageView mean;
  mean.data = acc.data(); mean.w = w; mean.h = h; mean.is_float = true;
  t.reset();
  Dark dm;
  darkness(mean, dm);
  dark_ms += t.ms();
  RegResult rr = register_dark(A, dm, P, Hs[0].data());
  Result R = decode_dark(A, dm, P, rr.H, nullptr, total);
  R.t.darkness = dark_ms;
  R.t.coarse = coarse_ms;
  R.t.edge = edge_ms + rr.t.edge;
  R.t.fit += 0;
  R.reg_score = rr.score;
  R.t.total = total.ms();
  (void)warp_ms;
  return R;
}

}  // namespace stego
