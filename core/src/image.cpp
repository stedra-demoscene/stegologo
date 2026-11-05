#include <algorithm>
#include "internal.hpp"

namespace stego {

namespace {
// median of histogram bins [lo,hi) with `scale` bins per grey level; exact (np.median) for scale==1 integer data
double hist_median(const std::vector<uint32_t>& hist, int lo, int hi, int scale) {
  uint64_t n = 0;
  for (int i = lo; i < hi; i++) n += hist[i];
  if (n == 0) return 0;
  auto kth = [&](uint64_t k) {  // value of k-th (0-based) sorted element
    uint64_t c = 0;
    for (int i = lo; i < hi; i++) {
      if (c + hist[i] > k) {
        if (scale == 1) return (double)i;
        return (i + (hist[i] ? (double)(k - c + 0.5) / hist[i] : 0.5)) / scale;
      }
      c += hist[i];
    }
    return (double)(hi - 1) / scale;
  };
  if (n & 1) return kth(n / 2);
  return 0.5 * (kth(n / 2 - 1) + kth(n / 2));
}
}  // namespace

void darkness(const ImageView& im, Dark& out) {
  const int w = im.w, h = im.h;
  const size_t stride = im.stride ? im.stride : w;
  out.w = w; out.h = h;
  out.D.resize((size_t)w * h);
  const int SC = im.is_float ? 4 : 1;
  const int NB = 256 * SC;
  std::vector<uint32_t> hist(NB, 0), h8(256, 0);
  if (im.is_float) {
    const float* p = (const float*)im.data;
    for (int y = 0; y < h; y++) {
      const float* r = p + y * stride;
      for (int x = 0; x < w; x++) {
        float v = std::min(std::max(r[x], 0.f), 255.99f);
        hist[(int)(v * SC)]++;
        h8[(int)std::min(255.f, std::nearbyint(v))]++;
      }
    }
  } else {
    const uint8_t* p = (const uint8_t*)im.data;
    for (int y = 0; y < h; y++) {
      const uint8_t* r = p + y * stride;
      uint32_t* hh = hist.data();
      for (int x = 0; x < w; x++) hh[r[x]]++;
    }
    h8 = hist;
  }
  // Otsu (as geostego.decode.otsu, on the rounded uint8 histogram)
  double tot = 0;
  for (int i = 0; i < 256; i++) tot += h8[i];
  double mu_all = 0;
  for (int i = 0; i < 256; i++) mu_all += (h8[i] / tot) * i;
  double w0 = 0, mu = 0, best = -1;
  int thr = 0;
  for (int i = 0; i < 256; i++) {
    double p = h8[i] / tot;
    w0 += p; mu += p * i;
    double v = (mu_all * w0 - mu) * (mu_all * w0 - mu) / (w0 * (1 - w0) + 1e-12);
    if (v > best) { best = v; thr = i; }
  }
  double paper, ink;
  if (im.is_float) {
    int tb = thr * SC + 1;   // pixels with value > thr (raw) are paper
    paper = hist_median(hist, tb, NB, SC);
    ink = hist_median(hist, 0, tb, SC);
  } else {
    paper = hist_median(hist, thr + 1, 256, 1);
    ink = hist_median(hist, 0, thr + 1, 1);
  }
  float inv = (float)(1.0 / std::max(paper - ink, 1.0));
  float pp = (float)paper;
  float* D = out.D.data();
  if (im.is_float) {
    const float* p = (const float*)im.data;
    for (int y = 0; y < h; y++) {
      const float* r = p + y * stride;
      float* d = D + (size_t)y * w;
      for (int x = 0; x < w; x++) d[x] = std::min(std::max((pp - r[x]) * inv, 0.f), 1.f);
    }
  } else {
    const uint8_t* p = (const uint8_t*)im.data;
    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = std::min(std::max((pp - (float)i) * inv, 0.f), 1.f);
    for (int y = 0; y < h; y++) {
      const uint8_t* r = p + y * stride;
      float* d = D + (size_t)y * w;
      for (int x = 0; x < w; x++) d[x] = lut[r[x]];
    }
  }
  out.paper = paper; out.ink = ink;
}

}  // namespace stego
