// Live mark path: detect the pose, crop around the mark (a view into the frame, no copy), decode against candidate assets.
#include <algorithm>
#include "internal.hpp"
#include "stego/marks.hpp"

namespace stego {

namespace {
struct Cand { int asset, hyp; Result r; bool refined; };
inline bool better(const Result& a, const Result& b) {
  if (a.accept != b.accept) return a.accept;
  return a.z_cal > b.z_cal;
}
}  // namespace

MarkDecodeResult decode_at(const std::vector<const Asset*>& assets, const ImageView& full, const MarkResult& det, const MarkDecodeOpts& o) {
  Timer t;
  MarkDecodeResult out;
  out.det = det;
  if (!det.found || assets.empty()) { out.ms_total = t.ms(); return out; }
  const Asset& A0 = *assets[0];
  // crop = bounding box of the page rectangle under H, plus the margin
  double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
  for (int c = 0; c < 4; c++) {
    const double mx = (c & 1) ? A0.page[0] : 0.0, my = (c & 2) ? A0.page[1] : 0.0;
    const double d = det.H[6] * mx + det.H[7] * my + det.H[8];
    const double x = (det.H[0] * mx + det.H[1] * my + det.H[2]) / d, y = (det.H[3] * mx + det.H[4] * my + det.H[5]) / d;
    xmin = std::min(xmin, x); xmax = std::max(xmax, x); ymin = std::min(ymin, y); ymax = std::max(ymax, y);
  }
  const double mg = o.crop_margin_mm * det.ppm;
  const int x0 = std::max(0, (int)std::floor(xmin - mg)), y0 = std::max(0, (int)std::floor(ymin - mg));
  const int x1 = std::min(full.w, (int)std::ceil(xmax + mg)), y1 = std::min(full.h, (int)std::ceil(ymax + mg));
  out.crop[0] = x0; out.crop[1] = y0; out.crop[2] = x1; out.crop[3] = y1;
  if (x1 - x0 < 32 || y1 - y0 < 32) { out.ms_total = t.ms(); return out; }
  const size_t stride = full.stride ? full.stride : full.w;
  ImageView cv;
  cv.is_float = full.is_float; cv.w = x1 - x0; cv.h = y1 - y0; cv.stride = (int)stride;
  cv.data = full.is_float ? (const void*)((const float*)full.data + (size_t)y0 * stride + x0) : (const void*)((const uint8_t*)full.data + (size_t)y0 * stride + x0);
  double T[9] = {1, 0, -(double)x0, 0, 1, -(double)y0, 0, 0, 1};
  std::vector<Cand> cands;
  Timer td;
  const int nh = std::max<size_t>(1, det.hyps.size());
  for (int h = 0; h < nh; h++) {
    double Hh[9], Hc[9];
    if (det.hyps.empty()) std::memcpy(Hh, det.H, sizeof Hh); else std::copy(det.hyps[h].begin(), det.hyps[h].end(), Hh);
    h_mul(T, Hh, Hc);
    for (int a = 0; a < (int)assets.size(); a++) {
      Result r = decode(*assets[a], cv, o.dec, Hc);
      out.n_decodes++;
      cands.push_back({a, h, std::move(r), false});
    }
  }
  size_t bi = 0;
  for (size_t i = 1; i < cands.size(); i++) if (better(cands[i].r, cands[bi].r)) bi = i;
  if (o.refine && !cands[bi].r.accept) {
    // the best few candidates (by z_cal) get a second chance from their start pose with the edge fit enabled
    std::vector<size_t> idx(cands.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return cands[a].r.z_cal > cands[b].r.z_cal; });
    Params p2 = o.dec; p2.refine_init = true;
    const size_t nref = std::min<size_t>(2, idx.size());
    for (size_t q = 0; q < nref; q++) {
      Cand& c = cands[idx[q]];
      double Hh[9], Hc[9];
      if (det.hyps.empty()) std::memcpy(Hh, det.H, sizeof Hh); else std::copy(det.hyps[c.hyp].begin(), det.hyps[c.hyp].end(), Hh);
      h_mul(T, Hh, Hc);
      Result r = decode(*assets[c.asset], cv, p2, Hc);
      out.n_decodes++;
      if (better(r, c.r)) { c.r = std::move(r); c.refined = true; }
    }
    bi = 0;
    for (size_t i = 1; i < cands.size(); i++) if (better(cands[i].r, cands[bi].r)) bi = i;
  }
  out.dec = cands[bi].r;
  out.asset = cands[bi].asset; out.hyp = cands[bi].hyp; out.refined = cands[bi].refined;
  out.ms_decode = td.ms();
  out.ms_total = t.ms();
  return out;
}

MarkDecodeResult detect_and_decode(const MarkDetector& det, const std::vector<const Asset*>& assets, const ImageView& full, const MarkDecodeOpts& o) {
  Timer t;
  MarkResult r = det.detect(full, o.det);
  MarkDecodeResult out = decode_at(assets, full, r, o);
  out.ms_total = t.ms();
  return out;
}

}  // namespace stego
