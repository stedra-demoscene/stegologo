// Demo cards: register the whole design from a keypoint seed, decode the 3 regions under that H, combine.
#include <algorithm>
#include "internal.hpp"
#include "stego/demo.hpp"

namespace stego {

DemoDecodeResult demo_decode(const Asset& comp, const std::vector<const Asset*>& regions, const std::vector<std::array<double, 2>>& shifts,
                             const ImageView& full, const double* H_seed, bool fast, double crop_margin_mm) {
  Timer t_all, t;
  DemoDecodeResult out;
  double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
  for (int c = 0; c < 4; c++) {
    const double mx = (c & 1) ? comp.page[0] : 0.0, my = (c & 2) ? comp.page[1] : 0.0;
    const double d = H_seed[6] * mx + H_seed[7] * my + H_seed[8];
    const double x = (H_seed[0] * mx + H_seed[1] * my + H_seed[2]) / d, y = (H_seed[3] * mx + H_seed[4] * my + H_seed[5]) / d;
    xmin = std::min(xmin, x); xmax = std::max(xmax, x); ymin = std::min(ymin, y); ymax = std::max(ymax, y);
  }
  const double dx = H_seed[0] * 1 + H_seed[3] * 0, dy = H_seed[1] * 0 + H_seed[4];
  const double ppm = std::sqrt(std::abs(H_seed[0] * H_seed[4] - H_seed[1] * H_seed[3]));
  (void)dx; (void)dy;
  const double mg = crop_margin_mm * ppm;
  const int x0 = std::max(0, (int)std::floor(xmin - mg)), y0 = std::max(0, (int)std::floor(ymin - mg));
  const int x1 = std::min(full.w, (int)std::ceil(xmax + mg)), y1 = std::min(full.h, (int)std::ceil(ymax + mg));
  out.crop[0] = x0; out.crop[1] = y0; out.crop[2] = x1; out.crop[3] = y1;
  if (x1 - x0 < 64 || y1 - y0 < 64) { out.ms_total = t_all.ms(); return out; }
  out.ok_crop = true;
  const size_t stride = full.stride ? full.stride : full.w;
  ImageView cv;
  cv.is_float = full.is_float; cv.w = x1 - x0; cv.h = y1 - y0; cv.stride = (int)stride;
  cv.data = full.is_float ? (const void*)((const float*)full.data + (size_t)y0 * stride + x0) : (const void*)((const uint8_t*)full.data + (size_t)y0 * stride + x0);
  double Tc[9] = {1, 0, -(double)x0, 0, 1, -(double)y0, 0, 0, 1};
  double Hs[9];
  h_mul(Tc, H_seed, Hs);
  Params preg;
  RegResult rr = register_block(comp, cv, preg, Hs);
  if (rr.score < 0.5) {
    Params pw;
    pw.reg_W = {2.4, 1.2, 0.6, 0.3}; pw.reg_iters = {4, 3, 3, 2}; pw.reg_step = {0.2, 0.1, 0.05, 0.03};
    RegResult r2 = register_block(comp, cv, pw, Hs);
    if (r2.score > rr.score) { rr = r2; out.wide = true; }
  }
  out.reg_score = rr.score;
  out.ms_reg = t.ms();
  t.reset();
  Params pd;
  if (fast) { pd.step = 0.02; pd.iter_step = 0.04; pd.fit_stride = 4; pd.iter_stride = 8; }
  std::vector<DemoRegion> regs;
  out.regions.resize(regions.size());
  for (size_t k = 0; k < regions.size(); k++) {
    double Tk[9] = {1, 0, shifts[k][0], 0, 1, shifts[k][1], 0, 0, 1}, Hk[9];
    h_mul(rr.H, Tk, Hk);
    Params pk = pd;
    if (!fast && regions[k]->W < 0.15) { pk.fit_stride = 1; pk.iter_stride = 1; pk.iter_step = pk.step; pk.n_iter = 4; }   // thin-line regions (small cards): full sampling, as the Python decoder
    out.regions[k] = decode(*regions[k], cv, pk, Hk);
  }
  for (size_t k = 0; k < regions.size(); k++) regs.push_back({regions[k], &out.regions[k]});
  out.comb = combine_demo(regs);
  double Tb[9] = {1, 0, (double)x0, 0, 1, (double)y0, 0, 0, 1};
  h_mul(Tb, rr.H, out.H);
  out.ms_dec = t.ms();
  out.ms_total = t_all.ms();
  return out;
}

}  // namespace stego
