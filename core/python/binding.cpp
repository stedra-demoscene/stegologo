// pybind11 module geostego_core: thin wrapper over the C++ core (numpy in, dict out).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "internal.hpp"
#include "stego/marks.hpp"
#include "stego/demo.hpp"

namespace py = pybind11;
using namespace stego;

namespace {
struct Img {
  py::array arr;
  ImageView v;
  Img(py::array a) {
    if (a.ndim() == 3) a = a[py::make_tuple(py::ellipsis(), 0)];
    if (a.ndim() != 2) throw std::runtime_error("expected a 2-D image");
    if (a.dtype().is(py::dtype::of<uint8_t>())) { arr = py::array::ensure(a, py::array::c_style); v.is_float = false; }
    else { arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(a); v.is_float = true; }
    v.data = arr.data(); v.h = (int)arr.shape(0); v.w = (int)arr.shape(1); v.stride = v.w;
  }
};

py::dict to_dict(const Result& R) {
  py::dict d;
  d["hex"] = R.hex; d["crc_ok"] = R.crc_ok; d["accept"] = R.accept; d["z_pilot"] = R.z_pilot; d["amp_um"] = R.amp_um;
  d["amp_raw_um"] = R.amp_raw_um; d["hp_retention"] = R.hp_retention; d["n_valid"] = R.n_valid; d["n_points"] = R.n_points;
  d["soft"] = py::array_t<float>(R.soft.size(), R.soft.data());
  d["bits"] = py::array_t<uint8_t>(R.bits.size(), R.bits.data());
  d["H"] = py::array_t<double>({3, 3}, R.H);
  d["pilot_s2"] = R.pilot_s2; d["z_cal"] = R.z_cal; d["null_mean"] = R.null_mean; d["null_sd"] = R.null_sd; d["z_raw"] = R.z_pilot;
  d["chip_y"] = py::array_t<double>(R.chip_y.size(), R.chip_y.data());
  d["chip_w"] = py::array_t<double>(R.chip_w.size(), R.chip_w.data());
  d["reg_score"] = R.reg_score; d["reg_cand"] = R.reg_cand;
  py::dict t;
  t["darkness"] = R.t.darkness; t["coarse"] = R.t.coarse; t["edge"] = R.t.edge; t["measure"] = R.t.measure;
  t["fit"] = R.t.fit; t["tail"] = R.t.tail; t["golay"] = R.t.golay; t["total"] = R.t.total;
  d["timing_ms"] = t;
  return d;
}

Params make_params(const py::kwargs& kw) {
  Params p;
  for (auto it : kw) {
    std::string k = py::cast<std::string>(it.first);
    py::handle v = it.second;
    if (k == "W") p.W = py::cast<double>(v);
    else if (k == "step") p.step = py::cast<double>(v);
    else if (k == "iter_step") p.iter_step = py::cast<double>(v);
    else if (k == "n_iter") p.n_iter = py::cast<int>(v);
    else if (k == "fit_stride") p.fit_stride = py::cast<int>(v);
    else if (k == "hp_chips") p.hp_chips = py::cast<double>(v);
    else if (k == "clip_mad") p.clip_mad = py::cast<double>(v);
    else if (k == "iter_stride") p.iter_stride = py::cast<int>(v);
    else if (k == "reg_W") p.reg_W = py::cast<std::vector<double>>(v);
    else if (k == "reg_iters") p.reg_iters = py::cast<std::vector<int>>(v);
    else if (k == "reg_step") p.reg_step = py::cast<std::vector<double>>(v);
    else if (k == "reg_stride") p.reg_stride = py::cast<int>(v);
    else if (k == "rot_step_deg") p.rot_step_deg = py::cast<int>(v);
    else if (k == "coarse_rounds") p.coarse_rounds = py::cast<int>(v);
    else if (k == "n_cands") p.n_cands = py::cast<int>(v);
    else if (k == "refine_init") p.refine_init = py::cast<bool>(v);
    else throw std::runtime_error("unknown parameter " + k);
  }
  return p;
}


MarkParams make_mark_params(const py::kwargs& kw) {
  MarkParams p;
  for (auto it : kw) {
    std::string k = py::cast<std::string>(it.first);
    py::handle v = it.second;
    if (k == "ppm_lo") p.ppm_lo = py::cast<double>(v);
    else if (k == "ppm_hi") p.ppm_hi = py::cast<double>(v);
    else if (k == "ppm_range") { auto r = py::cast<std::pair<double, double>>(v); p.ppm_lo = r.first; p.ppm_hi = r.second; }
    else if (k == "step") p.step = py::cast<double>(v);
    else if (k == "sym") p.sym = py::cast<int>(v);
    else if (k == "affine") p.affine = py::cast<bool>(v);
    else if (k == "homography") p.homography = py::cast<bool>(v);
    else if (k == "n_hyp") p.n_hyp = py::cast<int>(v);
    else if (k == "n_gn") p.n_gn = py::cast<int>(v);
    else if (k == "pad") p.pad = py::cast<int>(v);
    else if (k == "sym_cx") p.sym_cx = py::cast<double>(v);
    else if (k == "sym_cy") p.sym_cy = py::cast<double>(v);
    else if (k == "exclude") { for (auto e : py::cast<std::vector<std::array<double, 3>>>(v)) p.exclude.push_back(e); }
    else throw std::runtime_error("unknown parameter " + k);
  }
  return p;
}

py::dict mark_to_dict(const MarkResult& r) {
  py::dict o;
  o["found"] = r.found; o["score"] = r.score; o["iso"] = r.iso; o["sim_score"] = r.sim_score; o["aff_score"] = r.aff_score; o["hom_score"] = r.hom_score;
  o["H"] = py::array_t<double>({3, 3}, r.H);
  o["centre_px"] = py::make_tuple(r.centre[0], r.centre[1]);
  o["ppm"] = r.ppm; o["rot_deg"] = r.rot_deg; o["scale_ppm"] = r.scale_ppm;
  std::vector<double> hh;
  for (auto& h : r.hyps) hh.insert(hh.end(), h.begin(), h.end());
  o["hyps"] = py::array_t<double>({(py::ssize_t)r.hyps.size(), (py::ssize_t)3, (py::ssize_t)3}, hh.data());
  py::dict t;
  t["prep"] = r.t.prep; t["stage_a"] = r.t.stage_a; t["polar"] = r.t.polar; t["fine"] = r.t.fine; t["ecc"] = r.t.ecc; t["total"] = r.t.total;
  t["n_scales"] = r.t.n_scales; t["n_fft"] = r.t.n_fft;
  o["timing_ms"] = t;
  return o;
}

bool get_H(const py::object& o, double* H) {
  if (o.is_none()) return false;
  auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(o);
  if (!a || a.size() != 9) throw std::runtime_error("H must be 3x3");
  std::memcpy(H, a.data(), 9 * sizeof(double));
  return true;
}
}  // namespace

PYBIND11_MODULE(geostego_core, m) {
  m.doc() = "Native decoder core (registration + nuisance fit + despread)";
  py::class_<Asset, std::shared_ptr<Asset>>(m, "Asset")
      .def_property_readonly("page", [](const Asset& a) { return py::make_tuple(asset_page_w(a), asset_page_h(a)); });
  m.def("load_asset", [](const std::string& json, const std::string& bin) { return std::const_pointer_cast<Asset>(load_asset(json, bin)); }, py::arg("json_path"), py::arg("bin_path") = "");
  m.def("decode", [](const Asset& a, py::array img, py::object H, py::kwargs kw) {
    Img im(img);
    Params p = make_params(kw);
    double h[9];
    bool has = get_H(H, h);
    Result R;
    {
      py::gil_scoped_release rel;
      R = decode(a, im.v, p, has ? h : nullptr);
    }
    return to_dict(R);
  }, py::arg("asset"), py::arg("image"), py::arg("H") = py::none());
  m.def("decode_multi", [](const Asset& a, std::vector<py::array> imgs, py::kwargs kw) {
    std::vector<Img> keep;
    std::vector<ImageView> vs;
    keep.reserve(imgs.size());
    for (auto& i : imgs) { keep.emplace_back(i); vs.push_back(keep.back().v); }
    Params p = make_params(kw);
    Result R;
    {
      py::gil_scoped_release rel;
      R = decode_multi(a, vs.data(), (int)vs.size(), p);
    }
    return to_dict(R);
  }, py::arg("asset"), py::arg("images"));
  m.def("finalize_chips", [](const Asset& a, py::array_t<double, py::array::c_style | py::array::forcecast> y,
                             py::array_t<double, py::array::c_style | py::array::forcecast> w) {
    return to_dict(finalize_chips(a, y.data(), w.data()));
  }, py::arg("asset"), py::arg("chip_y"), py::arg("chip_w"));
  m.def("register", [](const Asset& a, py::array img, py::object H, py::kwargs kw) {
    Img im(img);
    Params p = make_params(kw);
    double h[9];
    bool has = get_H(H, h);
    RegResult r;
    {
      py::gil_scoped_release rel;
      r = register_block(a, im.v, p, has ? h : nullptr);
    }
    py::dict d;
    d["H"] = py::array_t<double>({3, 3}, r.H);
    d["score"] = r.score; d["cand"] = r.cand; d["coarse_ms"] = r.t.coarse; d["edge_ms"] = r.t.edge;
    return d;
  }, py::arg("asset"), py::arg("image"), py::arg("H") = py::none());

  py::class_<MarkDetector>(m, "MarkDetector")
      .def(py::init([](std::shared_ptr<Asset> a, double blur_mm) { return std::make_unique<MarkDetector>(*a, blur_mm); }), py::arg("asset"), py::arg("blur_mm") = 0.45)
      .def("detect", [](const MarkDetector& d, py::array img, py::kwargs kw) {
        Img im(img);
        MarkParams p = make_mark_params(kw);
        MarkResult r;
        {
          py::gil_scoped_release rel;
          r = d.detect(im.v, p);
        }
        return mark_to_dict(r);
      }, py::arg("image"));
  m.def("decode_mark", [](const MarkDetector& d, std::vector<std::shared_ptr<Asset>> assets, py::array img, py::object H, py::kwargs kw) {
    Img im(img);
    MarkDecodeOpts o;
    py::kwargs mk;
    for (auto it : kw) {
      std::string k = py::cast<std::string>(it.first);
      if (k == "crop_margin_mm") o.crop_margin_mm = py::cast<double>(it.second);
      else if (k == "refine") o.refine = py::cast<bool>(it.second);
      else if (k == "wide") { if (py::cast<bool>(it.second)) { o.dec.reg_W = {2.4, 1.2, 0.6, 0.3}; o.dec.reg_iters = {4, 3, 3, 2}; o.dec.reg_step = {0.2, 0.1, 0.05, 0.03}; } }
      else if (k == "fast") { if (py::cast<bool>(it.second)) { o.dec.step = 0.02; o.dec.iter_step = 0.04; o.dec.fit_stride = 4; o.dec.iter_stride = 8; } }
      else mk[it.first] = it.second;
    }
    o.det = make_mark_params(mk);
    std::vector<const Asset*> as;
    for (auto& a : assets) as.push_back(a.get());
    MarkDecodeResult R;
    double h9[9];
    const bool has = get_H(H, h9);
    {
      py::gil_scoped_release rel;
      if (has) { MarkResult mr; mr.found = true; std::memcpy(mr.H, h9, sizeof h9); mr.hyps.push_back({h9[0], h9[1], h9[2], h9[3], h9[4], h9[5], h9[6], h9[7], h9[8]});
                 mr.ppm = std::sqrt(std::abs(h9[0] * h9[4] - h9[1] * h9[3])); R = decode_at(as, im.v, mr, o); }
      else R = detect_and_decode(d, as, im.v, o);
    }
    py::dict out;
    out["det"] = mark_to_dict(R.det);
    out["dec"] = to_dict(R.dec);
    out["asset"] = R.asset; out["hyp"] = R.hyp; out["refined"] = R.refined; out["n_decodes"] = R.n_decodes;
    out["crop"] = py::make_tuple(R.crop[0], R.crop[1], R.crop[2], R.crop[3]);
    out["ms_decode"] = R.ms_decode; out["ms_total"] = R.ms_total;
    return out;
  }, py::arg("detector"), py::arg("assets"), py::arg("image"), py::arg("H") = py::none());
  m.def("combine_demo", [](std::vector<std::shared_ptr<Asset>> assets, std::vector<py::object> results, double z_accept, double zmin_use) {
    std::vector<Result> rs(results.size());
    std::vector<DemoRegion> regs(results.size());
    for (size_t i = 0; i < results.size(); i++) {
      regs[i].asset = assets[i].get();
      if (results[i].is_none()) continue;
      py::dict d = py::cast<py::dict>(results[i]);
      auto y = py::cast<std::vector<double>>(d["chip_y"]);
      auto w = py::cast<std::vector<double>>(d["chip_w"]);
      rs[i].chip_y = y; rs[i].chip_w = w;
      regs[i].r = &rs[i];
    }
    DemoResult R = combine_demo(regs, z_accept, zmin_use);
    py::dict o;
    o["accept"] = R.accept; o["crc_ok"] = R.crc_ok; o["tag_ok"] = R.tag_ok; o["token"] = R.token; o["z"] = R.z; o["zs"] = R.zs; o["gains"] = R.gains;
    o["bits"] = py::array_t<uint8_t>(R.bits.size(), R.bits.data());
    return o;
  }, py::arg("assets"), py::arg("results"), py::arg("z_accept") = kDemoZAccept, py::arg("zmin_use") = 2.0);
  m.def("demo_decode", [](std::shared_ptr<Asset> comp, std::vector<std::shared_ptr<Asset>> regions, std::vector<std::array<double, 2>> shifts, py::array img,
                          py::object H, bool fast) {
    Img im(img);
    double h9[9];
    if (!get_H(H, h9)) throw std::runtime_error("H seed required");
    std::vector<const Asset*> rs;
    for (auto& r : regions) rs.push_back(r.get());
    DemoDecodeResult R;
    {
      py::gil_scoped_release rel;
      R = demo_decode(*comp, rs, shifts, im.v, h9, fast);
    }
    py::dict o;
    o["accept"] = R.comb.accept; o["crc_ok"] = R.comb.crc_ok; o["tag_ok"] = R.comb.tag_ok; o["token"] = R.comb.token; o["z"] = R.comb.z;
    o["zs"] = R.comb.zs; o["gains"] = R.comb.gains;
    o["H"] = py::array_t<double>({3, 3}, R.H);
    o["reg_score"] = R.reg_score; o["wide"] = R.wide; o["crop"] = py::make_tuple(R.crop[0], R.crop[1], R.crop[2], R.crop[3]);
    o["ms_reg"] = R.ms_reg; o["ms_dec"] = R.ms_dec; o["ms_total"] = R.ms_total;
    py::list rl;
    for (auto& r : R.regions) rl.append(to_dict(r));
    o["regions"] = rl;
    return o;
  }, py::arg("comp"), py::arg("regions"), py::arg("shifts"), py::arg("image"), py::arg("H_seed"), py::arg("fast") = false);
  m.def("mark_fft_selftest", &mark_fft_selftest);
}
